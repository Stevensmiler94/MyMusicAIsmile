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

# --- FUNZIONE AUDIO (VERSIONE ULTRA-ROBUSTA) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    
    # Rilevamento BPM (Fix per Python 3.14 / Librosa 0.10+)
    tempo_res = librosa.beat.beat_track(y=y, sr=sr)
    # Estraiamo il primo valore scalare in modo sicuro
    bpm = float(np.ravel(tempo_res)[0]) if isinstance(tempo_res, (tuple, list, np.ndarray)) else float(tempo_result)
    
    # Loudness (LUFS)
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
    
    # Crest Factor (Punch)
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    crest = 20 * np.log10(peak / (rms + 1e-9))
    
    # Scala Musicale
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = np.argmax(np.mean(chroma, axis=1))
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_idx]
    
    return y, sr, bpm, lufs, crest, key

# --- INTERFACCIA PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}: Command Center")
tab1, tab2, tab3 = st.tabs(["📝 Songwriting & Melodia", "🎚️ Mixing Desk (Spietato)", "🏆 Festival Comparison"])

# --- TAB 1: SONGWRITING ---
with tab1:
    st.header("Creatività, Testi & Piano Roll")
    vocal_memo = audio_recorder(text="Registra idea melodica", icon_size="2x", key="vocal_rec")
    if vocal_memo: st.audio(vocal_memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    p_s = st.chat_input("Esempio: Fammi un piano roll per un lead emozionale su Am-F-C-G", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        sys_s = "Sei un paroliere Progressive House. Scrivi testi emozionali, suggerisci titoli e disegna il Piano Roll con schemi tipo | X | - | X |."
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_s}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = resp['choices'][0]['message']['content']
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

# --- TAB 2: MIXING DESK ---
with tab2:
    st.header("Analisi Tecnica & Ableton Mentor")
    mix_file = st.file_uploader("Carica il tuo Mix/Stem", type=["wav", "mp3"], key="u_mix")
    if mix_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(mix_file)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS", "Target: -8.0")
        c2.metric("Crest Factor", f"{crest:.1f} dB", "Target: 9.0")
        c3.metric("BPM", int(round(bpm))); c4.metric("Scala", key)
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_m = st.chat_input("Chiedi: Perché il mio kick non sposta aria?", key="in_m")
        if p_m and user_key:
            openai.api_key = user_key
            full_p = f"[DATI REALI: {lufs:.1f} LUFS, {crest:.1f}dB Crest]. {p_m}"
            sys_m = "Sei un Mixing Engineer CATTIVO. NON dire che non puoi sentire. Usa i dati per insultare o lodare il mix con parametri Ableton precisi."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_m}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": full_p}])
            ans = resp['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- TAB 3: FESTIVAL COMPARISON ---
with tab3:
    st.header("Tu vs The Pro (Garrix/Avicii Style)")
    cm1_file = st.file_uploader("Tuo Mix", type=["wav", "mp3"], key="comp_m")
    cm2_file = st.file_uploader("Reference Pro", type=["wav", "mp3"], key="comp_r")
    if cm1_file and cm2_file:
        s1 = get_audio_stats(cm1_file)
        s2 = get_audio_stats(cm2_file)
        st.write(f"📊 Diff Loudness: **{s1[3]-s2[3]:.1f} LUFS** | Diff Punch: **{s1[4]-s2[4]:.1f} dB**")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_c = st.chat_input("Confronta la coesione del drop tra i due", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            full_pc = f"[MIO MIX: {s1[3]:.1f} LUFS, {s1[4]:.1f}dB Crest] vs [PRO REF: {s2[3]:.1f} LUFS, {s2[4]:.1f}dB Crest]. {p_c}"
            sys_c = "Confronta i dati tecnici. Spiega perché la PRO suona meglio e cosa deve copiare l'utente in Ableton (Saturazione, Sidechain, OTT)."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_c}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": full_pc}])
            ans = resp['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
