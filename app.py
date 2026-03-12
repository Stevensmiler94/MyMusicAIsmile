import streamlit as st
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pdn
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="AI Music Studio Pro: Producer Edition", layout="wide", page_icon="🎧")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- UTILS AUDIO: FILTRI E ANALISI PRODUCER ---
def apply_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

@st.cache_data
def get_pro_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=45, mono=False) # 45 sec per beccare buildup/drop
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # 1. Analisi "AIR" (Frequenze > 12kHz)
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    air_idx = np.where(freqs >= 12000)
    air_energy = np.mean(S[air_idx, :]) / (np.mean(S) + 1e-9)
    
    # 2. Analisi Tensione (Varianza Energia RMS)
    rms = librosa.feature.rms(y=y_mono)[0]
    tension_score = np.std(rms) * 100 # Alta varianza = buildup/drop dinamico
    
    # 3. Sidechain/Kick Presence (Peak to RMS ratio)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    
    # 4. EQ Bands
    def get_e(f1, f2):
        idx = np.where((freqs >= f1) & (freqs <= f2))
        return np.mean(S[idx, :])
    l, m, h = get_e(20, 250), get_e(250, 4000), get_e(4000, 20000)
    
    # 5. LUFS & Phase
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    phase = np.corrcoef(y_stereo)[0,1]
    
    return {"y": y_mono, "sr": sr, "lufs": lufs, "phase": phase, "crest": crest,
            "air": air_energy, "tension": tension_score, "bands": {"Bass": l, "Mid": m, "High": h}}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    genere = st.selectbox("Genre Preset:", ["Progressive House (Garrix/Avicii Style)", "Techno", "Pop/Urban", "Generic"])
    
    # Prompt di sistema dinamico in base al genere
    sys_prompts = {
        "Progressive House (Garrix/Avicii Style)": "Sei un produttore Progressive House stile Martin Garrix e Avicii. Focus su: Euforia, Sidechain aggressivo, Lead brillanti (>12kHz) e Drop energetici.",
        "Techno": "Sei un produttore Techno. Focus su: Kick profondo, Rumble, Drive analogico e ipnotismo.",
        "Pop/Urban": "Sei un produttore Pop/Trap. Focus su: Vocali cristalline, 808 puliti e transienti veloci.",
        "Generic": "Sei un fonico esperto e bilanciato."
    }
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Progetto", type="json")
    if up_json:
        d_l = json.load(up_json)
        st.session_state.progetti[d_l["nome"]] = d_l["dati"]; st.rerun()

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta JSON", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name="studio.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo} | Mode: {genere}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing & Energy", "🏆 Mastering"])

with t2:
    f = st.file_uploader("Carica Mix", type=["wav", "mp3"])
    if f:
        d = get_pro_stats(f)
        
        # Dashboard Metriche Pro
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air (>12k)", f"{d['air']*100:.1f}%")
        c3.metric("Tension/Energy Drop", f"{d['tension']:.1f}")
        c4.metric("Kick Punch (Crest)", f"{d['crest']:.1f} dB")

        # Visualizzazioni
        st.subheader("🎧 Isolamento Spettrale & Monitor")
        target = st.radio("Ascolta Banda:", ["Tutto", "Sub/Bassi", "Medi", "Alti (Air)"], horizontal=True)
        f_audio = d['y']
        if target == "Sub/Bassi": f_audio = apply_filter(d['y'], 20, 200, d['sr'])
        elif target == "Medi": f_audio = apply_filter(d['y'], 500, 3000, d['sr'])
        elif target == "Alti (Air)": f_audio = apply_filter(d['y'], 8000, 15000, d['sr'])
        st.audio(f_audio, sample_rate=d['sr'])

        # AI CHECKLIST SPECIFICA PER IL GENERE
        if st.button("🪄 Analisi Strategica Garrix/Avicii"):
            if not api_key: st.error("Manca API Key")
            else:
                client = OpenAI(api_key=api_key)
                # Inviamo i dati di Air e Tension all'IA
                prompt = (f"Dati: Air {d['air']*100:.1f}%, Tension {d['tension']:.1f}, "
                          f"Crest {d['crest']:.1f}dB, LUFS {d['lufs']:.1f}. "
                          f"Genere: {genere}. Analizza il mix e dai 3 consigli pro.")
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompts[genere]}, {"role":"user","content":prompt}])
                st.success(res.choices[0].message.content)

        # Chat Mixing
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_m := st.chat_input("Chiedi un consiglio tecnico..."):
            client = OpenAI(api_key=api_key)
            ans = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompts[genere]}]+st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]+[{"role":"user","content":p_m}])
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"user","content":p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"assistant","content":ans.choices[0].message.content})
            st.rerun()

with t1:
    st.header("Emotional Songwriting")
    audio_recorder(text="Canta una melodia")
    # ... (Stessa logica chat del Tab 2 ma con prompt songwriting)
