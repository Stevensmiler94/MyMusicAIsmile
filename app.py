import streamlit as st
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pdn
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import json
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="AI Music Studio Pro", layout="wide", page_icon="🎙️")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

# --- INIZIALIZZAZIONE SESSIONE ---
if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- LOGICA CACHING AUDIO ---
@st.cache_data
def get_audio_stats(file_bytes):
    y, sr = librosa.load(file_bytes, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    meter = pdn.Meter(sr)
    l_data = y.reshape(-1, 1) if y.ndim == 1 else y.T
    lufs = meter.integrated_loudness(l_data)
    rms = np.sqrt(np.mean(y**2))
    crest = 20 * np.log10(np.max(np.abs(y)) / (rms + 1e-9))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = np.argmax(np.mean(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return {"y": y, "sr": sr, "bpm": bpm, "lufs": lufs, "crest": crest, "key": keys[key_idx]}

# --- SIDEBAR: GESTIONE PROGETTI & IMPORT/EXPORT ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.subheader("📂 Gestione Scenari")
    
    # IMPORT JSON
    uploaded_json = st.file_uploader("Importa Progetto (.json)", type="json")
    if uploaded_json:
        try:
            data_load = json.load(uploaded_json)
            if "nome" in data_load and "dati" in data_load:
                st.session_state.progetti[data_load["nome"]] = data_load["dati"]
                st.session_state.progetto_attivo = data_load["nome"]
                st.success(f"Caricato: {data_load['nome']}")
        except:
            st.error("Errore nel file JSON.")

    # SELEZIONE SCENARIO
    scenarios = list(st.session_state.progetti.keys())
    st.session_state.progetto_attivo = st.selectbox("Scenario Corrente", scenarios)
    
    if st.button("➕ Nuovo Scenario"):
        name = f"Progetto {len(st.session_state.progetti) + 1}"
        st.session_state.progetti[name] = crea_struttura_progetto()
        st.session_state.progetto_attivo = name
        st.rerun()

    # EXPORT JSON
    st.divider()
    current_data = st.session_state.progetti[st.session_state.progetto_attivo]
    export_json = json.dumps({"nome": st.session_state.progetto_attivo, "dati": current_data}, indent=4)
    st.download_button("📥 Scarica Progetto Corrente", export_json, file_name=f"{st.session_state.progetto_attivo}.json")

# --- FUNZIONI AI & GRAFICI ---
def call_ai(system_prompt, messages, user_input):
    if not api_key: return "Inserisci la API Key nella sidebar!"
    try:
        client = OpenAI(api_key=api_key)
        full_msgs = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}]
        res = client.chat.completions.create(model="gpt-4o-mini", messages=full_msgs)
        return res.choices[0].message.content
    except Exception as e: return f"Errore: {str(e)}"

def plot_visuals(y, sr):
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 3)); librosa.display.waveshow(y, sr=sr, ax=ax, color="skyblue")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(10, 3)); S = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=sr, ax=ax)
        st.pyplot(fig)

# --- UI PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing", "🏆 Mastering Comparison"])

with t1:
    st.header("Creatività")
    memo = audio_recorder(text="Registra idea")
    if memo: st.audio(memo)
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p := st.chat_input("Idea musicale..."):
        ans = call_ai("Sei un esperto di songwriting.", st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"], p)
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

with t2:
    st.header("Analisi Mix")
    f = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="mix_up")
    if f:
        d = get_audio_stats(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Crest Factor", f"{d['crest']:.1f} dB")
        c3.metric("BPM", int(d['bpm']))
        c4.metric("Scala", d['key'])
        plot_visuals(d['y'], d['sr'])
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p := st.chat_input("Analizza mix..."):
            ctx = f"[DATI: {d['lufs']:.1f} LUFS, {d['crest']:.1f}dB]. "
            ans = call_ai("Sei un fonico esperto.", st.session_state.progetti[st.session_state.progetto_attivo]["mixing"], ctx + p)
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

with t3:
    st.header("Mastering Benchmark")
    c_a, c_b = st.columns(2)
    f1 = c_a.file_uploader("Il tuo Mix", type=["wav", "mp3"], key="f1")
    f2 = c_b.file_uploader("Reference", type=["wav", "mp3"], key="f2")
    if f1 and f2:
        d1, d2 = get_audio_stats(f1), get_audio_stats(f2)
        st.info(f"📊 Delta: {d1['lufs']-d2['lufs']:.1f} LUFS | Crest: {d1['crest']-d2['crest']:.1f} dB")
        if st.button("🪄 Genera Strategia Mastering"):
            p_m = f"Confronto: Mio ({d1['lufs']:.1f} LUFS) vs Ref ({d2['lufs']:.1f} LUFS). Suggerisci catena plugin."
            st.success(call_ai("Sei un Mastering Engineer.", [], p_m))
