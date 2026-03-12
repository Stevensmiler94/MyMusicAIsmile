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

# --- INIZIALIZZAZIONE STRUTTURA DATI ---
def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# Auto-riparazione struttura
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

# Backup su PC
st.sidebar.divider()
st.sidebar.subheader("💾 Backup Locale")
progetto_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Scenario su PC", progetto_data, file_name=f"{st.session_state.progetto_attivo}.json")

carica_json = st.sidebar.file_uploader("📤 Carica Scenario (.json)", type=["json"])
if carica_json:
    try:
        data_caricata = json.load(carica_json)
        st.session_state.progetti[data_caricata["nome"]] = data_caricata["dati"]
        st.session_state.progetto_attivo = data_caricata["nome"]
        st.rerun()
    except: st.error("Errore JSON")

# --- FUNZIONE AUDIO (FIX BPM DEFINITIVO) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    
    # Rilevamento BPM Robusto
    tempo_res = librosa.beat.beat_track(y=y, sr=sr)
    # Estraiamo il valore scalare indipendentemente dal tipo (tuple o array)
    bpm = float(np.atleast_1d(tempo_res)[0])
    
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
    return y, sr, bpm, lufs, crest, key

# --- INTERFACCIA PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}: Command Center")
tab1, tab2, tab3 = st.tabs(["📝 Songwriting & Melodia", "🎚️ Mixing Desk (Spietato)", "🏆 Festival Comparison"])

# --- TAB 1: SONGWRITING ---
with tab1:
    st.header("Creatività & Testi")
    vocal_memo = audio_recorder(text="Registra idea", icon_size="2x", key="vocal_rec")
    if vocal_memo: st.audio(vocal_memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    p_s = st.chat_input("Chiedi testo o piano roll...", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        sys_s = "Sei un paroliere Progressive House. Scrivi testi emozionali e disegna Piano Roll | X | - |."
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_s}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else resp.choices[0].message.content
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

# --- TAB 2: MIXING DESK ---
with tab2:
    st.header("Analisi Mix Spietata")
    mix_file = st.file_uploader("Carica il tuo Mix", type=["wav", "mp3"], key="u_mix")
    if mix_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(mix_file)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS"); c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", int(bpm)); c4.metric("Scala", key)
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        p_m = st.chat_input("Chiedi all'ingegnere...", key="in_m")
        if p_m and user_key:
            openai.api_key = user_key
            full_p = f"[DATI: {lufs:.1f} LUFS, {crest:.1f}dB Crest]. {p_m}"
            sys_m = "Sei un Mixing Engineer CATTIVO. USA I DATI per insultare o lodare il mix con parametri Ableton precisi."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_m}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": full_p}])
            ans = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else resp.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- TAB 3: COMPARISON ---
with tab3:
    st.header("Tu vs Festival Pro")
    c_m_file = st.file_uploader("Tuo brano", type=["wav", "mp3"], key="c_my")
    c_r_file = st.file_uploader("Ref Pro", type=["wav", "mp3"], key="c_ref")
    if c_m_file and c_r_file:
        bpm1, lufs1, crest1, key1 = get_audio_stats(c_m_file)[2:]
        bpm2, lufs2, crest2, key2 = get_audio_stats(c_r_file)[2:]
        st.write(f"📊 Diff Loudness: {lufs1-lufs2:.1f} | Diff Crest: {crest1-crest2:.1f}")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_c = st.chat_input("Confronta kick e basso...", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            sys_c = "Confronta i dati tecnici. Spiega perché la PRO suona meglio in Ableton."
            full_pc = f"[MIO: {lufs1:.1f} LUFS, {crest1:.1f}dB] [PRO: {lufs2:.1f} LUFS, {crest2:.1f}dB]. {p_c}"
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_c}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": full_pc}])
            ans = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else resp.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
