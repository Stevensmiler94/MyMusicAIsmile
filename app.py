import streamlit as st
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pdn
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import json
import matplotlib.pyplot as plt

# Configurazione Pagina
st.set_page_config(page_title="AI Music Studio Pro", layout="wide", page_icon="🎙️")

# --- CORE LOGIC & CACHING ---
@st.cache_data
def get_audio_stats(file_bytes):
    """Analisi tecnica dell'audio con caching per performance ottimali."""
    y, sr = librosa.load(file_bytes, duration=30)
    
    # BPM Extraction (Librosa 0.10+)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    
    # Loudness (LUFS)
    meter = pdn.Meter(sr)
    l_data = y.reshape(-1, 1) if y.ndim == 1 else y.T
    lufs = meter.integrated_loudness(l_data)
    
    # Crest Factor (Dinamica)
    rms = np.sqrt(np.mean(y**2))
    crest = 20 * np.log10(np.max(np.abs(y)) / (rms + 1e-9))
    
    # Key Detection
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = np.argmax(np.mean(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return {
        "y": y, "sr": sr, "bpm": bpm, "lufs": lufs, 
        "crest": crest, "key": keys[key_idx]
    }

def call_ai(api_key, system_prompt, messages, user_input):
    """Gestione sicura delle chiamate OpenAI con error handling."""
    if not api_key:
        st.error("Inserisci la tua OpenAI API Key nella sidebar!")
        return None
    try:
        client = OpenAI(api_key=api_key)
        full_msgs = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}]
        response = client.chat.completions.create(model="gpt-4o-mini", messages=full_msgs)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Errore API: {str(e)}")
        return None

# --- UI COMPONENTS ---
def plot_audio_visuals(y, sr):
    """Genera Waveform e Spettrogramma."""
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Waveform (Ampiezza)**")
        fig_w, ax_w = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_w, color="skyblue")
        st.pyplot(fig_w)
    with col2:
        st.write("**Mel-Spectrogram (Frequenze)**")
        fig_s, ax_s = plt.subplots(figsize=(10, 3))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax_s)
        fig_s.colorbar(img, ax=ax_s, format='%+2.0f dB')
        st.pyplot(fig_s)

# --- INIZIALIZZAZIONE SESSIONE ---
if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": {"songwriting": [], "mixing": [], "comparison": []}}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    scenarios = list(st.session_state.progetti.keys())
    st.session_state.progetto_attivo = st.selectbox("Scenario Corrente", scenarios)
    
    if st.button("➕ Nuovo Scenario"):
        name = f"Progetto {len(st.session_state.progetti) + 1}"
        st.session_state.progetti[name] = {"songwriting": [], "mixing": [], "comparison": []}
        st.session_state.progetto_attivo = name
        st.rerun()

    if st.button("🗑️ Reset Chat"):
        st.session_state.progetti[st.session_state.progetto_attivo] = {"songwriting": [], "mixing": [], "comparison": []}
        st.rerun()

# --- TABS PRINCIPALI ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Mastering Comparison"])

# --- TAB 1: SONGWRITING ---
with t1:
    st.header("Creatività & Melodia")
    memo = audio_recorder(text="Registra idea vocale", icon_size="2x")
    if memo: st.audio(memo)
    
    chat_container = st.container()
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with chat_container.chat_message(m["role"]): st.write(m["content"])
    
    if prompt := st.chat_input("Chiedi consigli su testo o armonia..."):
        ans = call_ai(api_key, "Sei un esperto di songwriting e teoria musicale.", 
                      st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"], prompt)
        if ans:
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": prompt})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- TAB 2: MIXING DESK ---
with t2:
    st.header("Analisi Tecnica")
    audio_file = st.file_uploader("Carica Mix", type=["wav", "mp3"])
    
    if audio_file:
        with st.spinner("Analizzando audio..."):
            data = get_audio_stats(audio_file)
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{data['lufs']:.1f} LUFS")
        c2.metric("Crest Factor", f"{data['crest']:.1f} dB")
        c3.metric("BPM", int(data['bpm']))
        c4.metric("Scala", data['key'])
        
        plot_audio_visuals(data['y'], data['sr'])
        
        # Chat Mixing
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        if mix_prompt := st.chat_input("Cosa ne pensi del mio mix?"):
            ctx = f"[TECNICO: {data['lufs']:.1f} LUFS, Crest {data['crest']:.1f}dB, {int(data['bpm'])} BPM]. "
            ans = call_ai(api_key, "Sei un fonico esperto. Analizza i dati e dai consigli pratici (EQ, Comp, Gain Stage).", 
                          st.session_state.progetti[st.session_state.progetto_attivo]["mixing"], ctx + mix_prompt)
            if ans:
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": mix_prompt})
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
                st.rerun()

# --- TAB 3: COMPARISON & MASTERING ---
with t3:
    st.header("Mastering Benchmark")
    col_a, col_b = st.columns(2)
    f_my = col_a.file_uploader("Il tuo Mix", type=["wav", "mp3"], key="my")
    f_ref = col_b.file_uploader("Traccia Reference (Pro)", type=["wav", "mp3"], key="ref")
    
    if f_my and f_ref:
        d_my = get_audio_stats(f_my)
        d_ref = get_audio_stats(f_ref)
        
        delta_l = d_my['lufs'] - d_ref['lufs']
        delta_c = d_my['crest'] - d_ref['crest']
        
        st.divider()
        st.subheader("Analisi Comparativa")
        st.info(f"📊 **Loudness Delta:** {delta_l:.1f} LUFS | **Crest Delta:** {delta_c:.1f} dB")
        
        # Analisi automatica AI
        if st.button("🪄 Genera Strategia di Mastering"):
            prompt_master = (f"Confronta questi dati: Mio Mix ({d_my['lufs']:.1f} LUFS, {d_my['crest']:.1f}dB) "
                            f"vs Reference ({d_ref['lufs']:.1f} LUFS, {d_ref['crest']:.1f}dB). "
                            "Suggerisci una catena di mastering specifica.")
            
            ans = call_ai(api_key, "Sei un Mastering Engineer. Sii molto tecnico e preciso.", [], prompt_master)
            if ans: st.success(ans)
