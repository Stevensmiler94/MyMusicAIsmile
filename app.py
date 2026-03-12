import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai
from audio_recorder_streamlit import audio_recorder
import json

# Configurazione Pagina
st.set_page_config(page_title="AI Music Command Center", layout="wide", page_icon="🎛️")

# --- STRUTTURA DATI & PERSISTENZA ---
def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# Auto-riparazione struttura progetti esistenti
for p_nome in list(st.session_state.progetti.keys()):
    if not isinstance(st.session_state.progetti[p_nome], dict) or "songwriting" not in st.session_state.progetti[p_nome]:
        st.session_state.progetti[p_nome] = crea_struttura_progetto()

# --- SIDEBAR: STUDIO MANAGER ---
st.sidebar.title("🏢 Studio Manager")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
st.session_state.progetto_attivo = st.sidebar.selectbox("Scenario Attivo", list(st.session_state.progetti.keys()))

if st.sidebar.button("➕ Nuovo Scenario"):
    nuovo_nome = f"Progetto {len(st.session_state.progetti) + 1}"
    st.session_state.progetti[nuovo_nome] = crea_struttura_progetto()
    st.session_state.progetto_attivo = nuovo_nome
    st.rerun()

# --- FUNZIONE SALVATAGGIO/CARICAMENTO JSON ---
st.sidebar.divider()
st.sidebar.subheader("💾 Backup Locale")
progetto_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Progetto su PC", progetto_data, file_name=f"{st.session_state.progetto_attivo}.json")

carica_json = st.sidebar.file_uploader("📤 Carica Progetto da PC", type=["json"])
if carica_json:
    try:
        data_caricata = json.load(carica_json)
        st.session_state.progetti[data_caricata["nome"]] = data_caricata["dati"]
        st.session_state.progetto_attivo = data_caricata["nome"]
        st.rerun()
    except: st.error("Errore nel file JSON")

# --- FUNZIONE AUDIO (FIX DEFINITIVO BPM PYTHON 3.14) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    
    # Estrazione BPM: Metodo ultra-sicuro senza indici [0]
    tempo_result = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = np.mean(tempo_result) # Media per estrarre il valore numerico puro
    bpm = float(bpm_val)
    
    # Loudness e Crest Factor
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    
    # Scala
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
    return y, sr, bpm, lufs, crest, key

# --- INTERFACCIA PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}: Command Center")
tab1, tab2, tab3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk (Spietato)", "🏆 Comparison"])

# --- TAB 1: SONGWRITING ---
with tab1:
    st.header("Creatività & Melodia")
    vocal_memo = audio_recorder(text="Registra idea", icon_size="2x", key="v_rec")
    if vocal_memo: st.audio(vocal_memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    p_s = st.chat_input("Esempio: Scrivi un testo emozionale e disegna il Piano Roll", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        sys_s = "Sei un paroliere Progressive House. Scrivi testi in inglese e disegna il Piano Roll con schemi | X | - |."
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_s}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = resp['choices'][0]['message']['content']
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

# --- TAB 2: MIXING DESK ---
with tab2:
    st.header("Analisi Mix Spietata")
    mix_file = st.file_uploader("Carica Mix/Stem", type=["wav", "mp3"], key="u_mix")
    if mix_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(mix_file)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS"); c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", int(round(bpm))); c4.metric("Scala", key)
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        p_m = st.chat_input("Chiedi: Perché il mio kick non sposta aria?", key="in_mix")
        if p_m and user_key:
            openai.api_key = user_key
            full_p = f"[DATI: {lufs:.1f} LUFS, {crest:.1f}dB, BPM {int(bpm)}]. {p_m}"
            sys_m = "Sei un Mixing Engineer CATTIVO. USA I DATI per insultare o lodare il mix con parametri Ableton precisi."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_m}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": full_p}])
            ans = resp['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- TAB 3: COMPARISON ---
with tab3:
    st.header("Tu vs Festival Pro")
    cm1_f = st.file_uploader("Tuo Mix", type=["wav", "mp3"], key="c_my")
    cm2_f = st.file_uploader("Ref Pro", type=["wav", "mp3"], key="c_ref")
    if cm1_f and cm2_f:
        s1 = get_audio_stats(cm1_f); s2 = get_audio_stats(cm2_f)
        st.write(f"📊 Diff Loudness: {s1[3]-s2[3]:.1f} | Diff Crest: {s1[4]-s2[4]:.1f}")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_c = st.chat_input("Confronta kick e basso tra i due", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            full_pc = f"[MIO: {s1[3]:.1f} LUFS, {s1[4]:.1f}dB] vs [PRO: {s2[3]:.1f} LUFS, {s2[4]:.1f}dB]. {p_c}"
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Confronta i dati tecnici delle due tracce."}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": full_pc}])
            ans = resp['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
