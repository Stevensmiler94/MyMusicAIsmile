import streamlit as st
import librosa
import librosa.display
import numpy as np
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

# --- SIDEBAR & GESTIONE PROGETTI ---
st.sidebar.title("🏢 Studio Manager")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

# 1. Caricamento Progetto da PC (IMPORT)
uploaded_json = st.sidebar.file_uploader("📂 Carica Progetto (.json)", type="json")
if uploaded_json:
    try:
        data = json.load(uploaded_json)
        st.session_state.progetti[data["nome"]] = data["dati"]
        st.sidebar.success(f"Caricato: {data['nome']}")
    except Exception as e:
        st.sidebar.error("Errore nel caricamento file.")

st.sidebar.divider()

# Selezione Scenario
st.session_state.progetto_attivo = st.sidebar.selectbox("Scenario Attivo", list(st.session_state.progetti.keys()))

# 2. Nuovo Scenario
if st.sidebar.button("➕ Nuovo Scenario"):
    nome_nuovo = f"Progetto {len(st.session_state.progetti) + 1}"
    st.session_state.progetti[nome_nuovo] = crea_struttura_progetto()
    st.session_state.progetto_attivo = nome_nuovo
    st.rerun()

# 3. Cancella Chat / Reset
if st.sidebar.button("🗑️ Reset Chat Scenario"):
    st.session_state.progetti[st.session_state.progetto_attivo] = crea_struttura_progetto()
    st.rerun()

st.sidebar.divider()

# 4. Download Progetto (EXPORT)
exp_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Progetto su PC", exp_data, file_name=f"{st.session_state.progetto_attivo}.json")

# --- FUNZIONE AUDIO (FIX BPM & COMPATIBILITÀ) ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    
    # Fix Librosa 0.10+: beat_track restituisce (tempo, beat_frames)
    tempo_result = librosa.beat.beat_track(y=y, sr=sr)
    # Se tempo_result è una tupla, prendiamo il primo elemento (il tempo)
    bpm = float(tempo_result[0]) if isinstance(tempo_result, tuple) else float(tempo_result)
    
    # Loudness/Crest
    meter = pdn.Meter(sr)
    l_data = y.reshape(-1, 1) if y.ndim == 1 else y.T
    lufs = meter.integrated_loudness(l_data)
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    
    # Scala
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = np.argmax(np.mean(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return y, sr, bpm, lufs, crest, keys[key_idx]

# --- UI PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison"])

# TAB 1: SONGWRITING
with t1:
    st.header("Creatività & Melodia")
    memo = audio_recorder(text="Registra idea", icon_size="2x", key="v_rec")
    if memo: st.audio(memo)
    
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    
    p_s = st.chat_input("Esempio: Fammi un piano roll per un lead emozionale", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        messages = [{"role": "system", "content": "Sei un paroliere Progressive House."}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}]
        r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
        ans = r.choices[0].message.content
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

# TAB 2: MIXING
with t2:
    st.header("Analisi Mix Spietata")
    m_file = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if m_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(m_file)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", int(round(bpm)))
        c4.metric("Scala", key)
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        p_m = st.chat_input("Perché il mio kick non spinge?", key="in_m")
        if p_m and user_key:
            openai.api_key = user_key
            ctx = f"[DATI: {lufs:.1f} LUFS, {crest:.1f}dB, {int(bpm)} BPM]. "
            messages = [{"role": "system", "content": "Sei un Mixing Engineer CATTIVO. USA I DATI per insultare o lodare il mix."}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": ctx + p_m}]
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
            ans = r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

# TAB 3: COMPARISON
with t3:
    st.header("Tu vs Festival Pro")
    col_a, col_b = st.columns(2)
    with col_a: f1 = st.file_uploader("Tuo Mix", type=["wav", "mp3"], key="c_my")
    with col_b: f2 = st.file_uploader("Ref Pro", type=["wav", "mp3"], key="c_ref")
    
    if f1 and f2:
        y1, sr1, bpm1, lufs1, crest1, key1 = get_audio_stats(f1)
        y2, sr2, bpm2, lufs2, crest2, key2 = get_audio_stats(f2)
        st.info(f"📊 Diff Loudness: {lufs1-lufs2:.1f} LUFS | Diff Crest: {crest1-crest2:.1f} dB")
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        p_c = st.chat_input("Confronta i due file", key="in_c")
        if p_c and user_key:
            openai.api_key = user_key
            ctx = f"[MIO: {lufs1:.1f} LUFS, {crest1:.1f}dB] vs [PRO: {lufs2:.1f} LUFS, {crest2:.1f}dB]. "
            messages = [{"role": "system", "content": "Confronta tecnicamente le due tracce."}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": ctx + p_c}]
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
            ans = r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()
