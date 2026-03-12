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

# Riparazione automatica struttura
for p in list(st.session_state.progetti.keys()):
    if not isinstance(st.session_state.progetti[p], dict) or "songwriting" not in st.session_state.progetti[p]:
        st.session_state.progetti[p] = crea_struttura_progetto()

# --- SIDEBAR: STUDIO MANAGER ---
st.sidebar.title("🏢 Studio Manager")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
st.session_state.progetto_attivo = st.sidebar.selectbox("Scenario Attivo", list(st.session_state.progetti.keys()))

if st.sidebar.button("➕ Nuovo Scenario"):
    nuovo_nome = f"Progetto {len(st.session_state.progetti) + 1}"
    st.session_state.progetti[nuovo_nome] = crea_struttura_progetto()
    st.session_state.progetto_attivo = nuovo_nome
    st.rerun()

# --- SALVATAGGIO / CARICAMENTO JSON ---
st.sidebar.divider()
st.sidebar.subheader("💾 Backup su PC")
exp_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Progetto su PC", exp_data, file_name=f"{st.session_state.progetto_attivo}.json")

carica_json = st.sidebar.file_uploader("📤 Carica Progetto da PC", type=["json"])
if carica_json:
    try:
        d = json.load(carica_json)
        st.session_state.progetti[d["nome"]] = d["dati"]
        st.session_state.progetto_attivo = d["nome"]
        st.rerun()
    except: st.error("Errore JSON")

# --- FUNZIONE AUDIO (FIX DEFINITIVO BPM PYTHON 3.14) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    
    # Rilevamento BPM: Gestione esplicita della tupla (BPM, frames)
    tempo_result = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo_result, tuple):
        bpm = float(tempo_result[0])
    else:
        bpm = float(tempo_result)
    
    # Loudness e Crest Factor
    meter = pdn.Meter(sr)
    l_data = y.reshape(-1, 1) if y.ndim == 1 else y.T
    lufs = meter.integrated_loudness(l_data)
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    
    # Scala
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
    return y, sr, bpm, lufs, crest, key

# --- INTERFACCIA ---
st.title(f"🚀 {st.session_state.progetto_attivo}: Command Center")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison"])

with t1:
    st.header("Creatività & Melodia")
    memo = audio_recorder(text="Registra idea", icon_size="2x", key="v_rec")
    if memo: st.audio(memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    p_s = st.chat_input("Chiedi testo o piano roll...", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        sys = "Sei un paroliere Progressive House. Scrivi testi in inglese e disegna il Piano Roll | X | - |."
        r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = r['choices'][0]['message']['content'] if isinstance(r, dict) else r.choices[0].message.content
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

with t2:
    st.header("Analisi Mix Spietata")
    m_file = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if m_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(m_file)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS"); c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", int(round(bpm))); c4.metric("Scala", key)
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_m = st.chat_input("Chiedi: Perché il mio kick non sposta aria?", key="in_m")
        if p_m and user_key:
            openai.api_key = user_key
            full = f"[DATI: {lufs:.1f} LUFS, {crest:.1f}dB, {int(bpm)} BPM]. {p_m}"
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un Mixing Engineer CATTIVO. USA I DATI per insultare o lodare il mix con parametri Ableton."}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": full}])
            ans = r['choices'][0]['message']['content'] if isinstance(r, dict) else r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

with t3:
    st.header("Tu vs Festival Pro")
    f1 = st.file_uploader("Tuo Mix", type=["wav", "mp3"], key="c_my")
    f2 = st.file_uploader("Ref Pro", type=["wav", "mp3"], key="c_ref")
    if f1 and f2:
        s1 = get_audio_stats(f1); s2 = get_audio_stats(f2)
        st.write(f"📊 Diff Loudness: {s1[3]-s2[3]:.1f} | Diff Crest: {s1[4]-s2[4]:.1f}")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        p_c = st.chat_input("Confronta kick e basso tra i due", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            full = f"[MIO: {s1[3]:.1f} LUFS, {s1[4]:.1f}dB] vs [PRO: {s2[3]:.1f} LUFS, {s2[4]:.1f}dB]. {p_c}"
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Confronta i dati tecnici delle due tracce."}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": full}])
            ans = r['choices'][0]['message']['content'] if isinstance(r, dict) else r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
