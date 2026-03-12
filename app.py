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

# --- INIZIALIZZAZIONE ---
def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# Riparazione automatica
for p in list(st.session_state.progetti.keys()):
    if not isinstance(st.session_state.progetti[p], dict) or "songwriting" not in st.session_state.progetti[p]:
        st.session_state.progetti[p] = crea_struttura_progetto()

# --- SIDEBAR ---
st.sidebar.title("🏢 Studio Manager")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
st.session_state.progetto_attivo = st.sidebar.selectbox("Scenario", list(st.session_state.progetti.keys()))

if st.sidebar.button("➕ Nuovo Scenario"):
    nome = f"Progetto {len(st.session_state.progetti) + 1}"
    st.session_state.progetti[nome] = crea_struttura_progetto()
    st.session_state.progetto_attivo = nome
    st.rerun()

# Export/Import
st.sidebar.divider()
exp_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Progetto su PC", exp_data, file_name=f"{st.session_state.progetto_attivo}.json")

# --- FUNZIONE AUDIO (FIX DEFINITIVO BPM) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    # Forza l'output in un array per estrarre il primo valore in modo sicuro
    tempo_result = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo_result)[0])
    # Loudness/Crest
    meter = pdn.Meter(sr)
    l_data = y.reshape(-1, 1) if y.ndim == 1 else y.T
    lufs = meter.integrated_loudness(l_data)
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    # Scala
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
    return y, sr, bpm, lufs, crest, key

# --- UI ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison"])

with t1:
    st.header("Creatività & Melodia")
    memo = audio_recorder(text="Registra idea", icon_size="2x", key="v_rec")
    if memo: st.audio(memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    p_s = st.chat_input("Esempio: Fammi un piano roll per un lead emozionale", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un paroliere Progressive House."}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = r.choices[0].message.content if hasattr(r, 'choices') else r['choices'][0]['message']['content']
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
            with st.chat_message(m["role"]): st.write(m["content"])
        p_m = st.chat_input("Perché il mio kick non spinge?", key="in_m")
        if p_m and user_key:
            openai.api_key = user_key
            ctx = f"[DATI: {lufs:.1f} LUFS, {crest:.1f}dB, {int(bpm)} BPM]. "
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un Mixing Engineer CATTIVO. USA I DATI per insultare o lodare il mix."}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": ctx + p_m}])
            ans = r.choices[0].message.content if hasattr(r, 'choices') else r['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

with t3:
    st.header("Tu vs Festival Pro")
    f1 = st.file_uploader("Tuo Mix", type=["wav", "mp3"], key="c_my")
    f2 = st.file_uploader("Ref Pro", type=["wav", "mp3"], key="c_ref")
    if f1 and f2:
        y1, sr1, bpm1, lufs1, crest1, key1 = get_audio_stats(f1)
        y2, sr2, bpm2, lufs2, crest2, key2 = get_audio_stats(f2)
        st.write(f"📊 Diff Loudness: {lufs1-lufs2:.1f} | Diff Crest: {crest1-crest2:.1f}")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        p_c = st.chat_input("Confronta i due file", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            ctx = f"[MIO: {lufs1:.1f} LUFS, {crest1:.1f}dB] vs [PRO: {lufs2:.1f} LUFS, {crest2:.1f}dB]. "
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Confronta tecnicamente le due tracce."}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": ctx + p_c}])
            ans = r.choices[0].message.content if hasattr(r, 'choices') else r['choices'][0]['message']['content']
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
