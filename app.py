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
st.set_page_config(page_title="AI Studio Pro: Platinum Edition", layout="wide", page_icon="🎹")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- ENGINE AUDIO ---
def apply_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low, high = max(0.01, lowcut / nyq), min(0.99, highcut / nyq)
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

@st.cache_data
def get_platinum_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_psd = np.mean(S, axis=1)
    
    # Air & Kick-Bass Masking
    idx_air = np.where(freqs >= 10000)
    air_val = np.mean(S[idx_air, :]) / (np.mean(S) + 1e-9) if len(idx_air) > 0 else 0.0
    low_idx = np.where(freqs <= 100)
    low_energy_ts = np.mean(S[low_idx, :], axis=0)
    kb_score = np.std(low_energy_ts) * 100 
    
    # Stereo & Stats
    correlation = np.corrcoef(y_stereo, y_stereo)
    width = (1 - correlation) * 100 
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    rms = librosa.feature.rms(y=y_mono)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    
    return {"y": y_mono, "sr": sr, "lufs": lufs, "crest": crest, "air": air_val, 
            "psd": avg_psd, "freqs": freqs, "width": width, "kick_bass": kb_score}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genere Target:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    sys_instruction = f"Rispondi SEMPRE in ITALIANO. Sei un produttore esperto di {genere}."

    st.divider()
    up_json = st.file_uploader("📂 Importa Scenario", type="json")
    if up_json:
        try:
            d_l = json.load(up_json)
            st.session_state.progetti[d_l["nome"]] = d_l["dati"]
            st.session_state.progetto_attivo = d_l["nome"]; st.rerun()
        except: st.error("Errore file")

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name="studio.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo} | Mode: {genere}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison & EQ"])

# --- TAB 1: SONGWRITING ---
with t1:
    st.header("Ideazione & Testi")
    audio_recorder(text="Registra idea vocale")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p_s := st.chat_input("Chiedi un consiglio (Songwriting)..."):
        if api_key:
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p_s}])
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p_s})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":r.choices[0].message.content})
            st.rerun()

# --- TAB 2: MIXING DESK ---
with t2:
    st.header("Analisi Platinum")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        d = get_platinum_stats(f_mix)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.2f}%")
        c3.metric("Stereo Width", f"{d['width']:.1f}%")
        c4.metric("KB Space", f"{d['kick_bass']:.1f}")

        st.subheader("🎧 Isolamento Spettrale")
        target = st.radio("Ascolta:", ["Tutto", "Bassi (<200Hz)", "Alti (>8kHz)"], horizontal=True)
        f_p = d['y']
        if target == "Bassi (<200Hz)": f_p = apply_filter(d['y'], 20, 200, d['sr'])
        elif target == "Alti (>8kHz)": f_p = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_p, sample_rate=d['sr'])

        # Chat Mixing
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_m := st.chat_input("Fai una domanda sul mix..."):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = f"[DATI: Air {d['air']:.2%}, Width {d['width']:.1f}%, KB-Space {d['kick_bass']:.1f}] "
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]+[{"role":"user","content":ctx + p_m}])
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"user","content":p_m})
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"assistant","content":r.choices[0].message.content})
                st.rerun()

# --- TAB 3: COMPARISON ---
with t3:
    st.header("EQ Match & Comparison Chat")
    cl, cr = st.columns(2)
    f1, f2 = cl.file_uploader("Mio Mix", key="c1"), cr.file_uploader("Ref", key="c2")
    if f1 and f2:
        d1, d2 = get_platinum_stats(f1), get_platinum_stats(f2)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio", color="cyan", lw=2)
        ax.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Ref", color="orange", alpha=0.6, lw=2)
        ax.set_xlim(20, 20000); ax.legend(); plt.grid(True, which="both", alpha=0.1)
        st.pyplot(fig)
        
        st.metric("Delta Kick-Bass Space", f"{d1['kick_bass']-d2['kick_bass']:.1f}")

        # Chat Comparison
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_c := st.chat_input("Cosa manca al mio drop rispetto alla reference?"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = f"[CONFRONTO: MIO Air {d1['air']:.2%}, KB {d1['kick_bass']:.1f} vs REF Air {d2['air']:.2%}, KB {d2['kick_bass']:.1f}] "
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]+[{"role":"user","content":ctx + p_c}])
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"user","content":p_c})
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"assistant","content":r.choices[0].message.content})
                st.rerun()
