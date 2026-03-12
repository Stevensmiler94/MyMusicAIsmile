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
st.set_page_config(page_title="AI Studio Pro Ultimate", layout="wide", page_icon="🎙️")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- UTILS AUDIO: FILTRI ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# --- ENGINE ANALISI AVANZATA ---
@st.cache_data
def get_ultimate_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # 1. Fase Stereo & Punch
    correlation = np.corrcoef(y_stereo[0], y_stereo[1])[0,1]
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    punch = np.mean(onset_env) * 10 
    
    # 2. EQ Energy
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    def get_e(f1, f2):
        idx = np.where((freqs >= f1) & (freqs <= f2))
        return np.mean(S[idx, :]) if len(idx[0]) > 0 else 0
    l, m, h = get_e(20, 250), get_e(250, 4500), get_e(4500, 20000)
    tot = l + m + h
    
    # 3. LUFS
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y_stereo.T)
    
    return {"y": y_mono, "y_s": y_stereo, "sr": sr, "lufs": lufs, "phase": correlation, 
            "punch": punch, "bands": {"Bassi": l/tot, "Medi": m/tot, "Alti": h/tot}}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if st.button("🔄 Reset Totale App"):
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Progetto", type="json")
    if up_json:
        d_l = json.load(up_json)
        st.session_state.progetti[d_l["nome"]] = d_l["dati"]
        st.session_state.progetto_attivo = d_l["nome"]

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name=f"{st.session_state.progetto_attivo}.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Mastering"])

with t1:
    st.header("Ideazione")
    audio_recorder(text="Registra idea vocale")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p := st.chat_input("Scrivi qui..."):
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un esperto songwriter."}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p}]).choices.message.content
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":res})
        st.rerun()

with t2:
    st.header("Analisi & Monitor")
    f_mix = st.file_uploader("Carica traccia", type=["wav", "mp3"], key="main_u")
    
    if f_mix:
        d = get_ultimate_stats(f_mix)
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🔄 Riesamina Traccia (Refresh)"):
            st.cache_data.clear()
            st.rerun()
        if col_btn2.button("🧹 Svuota Chat Mixing"):
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] = []
            st.rerun()

        # Monitor Sezione
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Stereo Phase", f"{d['phase']:.2f}")
        c3.metric("Punch Score", f"{d['punch']:.1f}")
        c4.metric("EQ Focus", max(d['bands'], key=d['bands'].get))

        # --- FUNZIONE FILTRO ASCOLTO ---
        st.subheader("🎧 Monitor di Isolamento Spettrale")
        target_band = st.radio("Isola Frequenze:", ["Tutto", "Bassi (20-250Hz)", "Medi (250-4000Hz)", "Alti (4000Hz+)"], horizontal=True)
        
        filtered_audio = d['y']
        if target_band == "Bassi (20-250Hz)": filtered_audio = apply_filter(d['y'], 20, 250, d['sr'])
        elif target_band == "Medi (250-4000Hz)": filtered_audio = apply_filter(d['y'], 250, 4000, d['sr'])
        elif target_band == "Alti (4000Hz+)": filtered_audio = apply_filter(d['y'], 4000, 10000, d['sr'])
        
        st.audio(filtered_audio, sample_rate=d['sr'])
        st.caption(f"Stai ascoltando: {target_band}")

        # Visuals & AI
        st.bar_chart(d['bands'])
        if st.button("🪄 Genera Checklist Correzione Profonda"):
            client = OpenAI(api_key=api_key)
            prompt = f"Analisi: {d['lufs']:.1f} LUFS, Fase {d['phase']:.2f}, Punch {d['punch']:.1f}, EQ: {d['bands']}. Genera checklist correttiva."
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un Master Mixing Engineer."},{"role":"user","content":prompt}]).choices.message.content
            st.success(res)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_m := st.chat_input("Analisi mix..."):
            client = OpenAI(api_key=api_key)
            ctx = f"[PHASE: {d['phase']:.2f}, PUNCH: {d['punch']:.1f}]. "
            ans = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un fonico esperto."}]+st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]+[{"role":"user","content":ctx+p_m}]).choices.message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"user","content":p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"assistant","content":ans})
            st.rerun()

with t3:
    st.header("Mastering Benchmark")
    ca, cb = st.columns(2)
    f1, f2 = ca.file_uploader("Mio Mix", key="m1"), cb.file_uploader("Ref Pro", key="m2")
    if f1 and f2:
        d1, d2 = get_ultimate_stats(f1), get_ultimate_stats(f2)
        st.info(f"📊 Delta Loudness: {d1['lufs']-d2['lufs']:.1f} LUFS | Delta Fase: {d1['phase']-d2['phase']:.2f}")
        if st.button("🚀 Ottieni Strategia Mastering"):
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un Mastering Engineer."},{"role":"user","content":f"Mio: {d1['lufs']:.1f} LUFS, {d1['phase']:.2f} fase. Ref: {d2['lufs']:.1f} LUFS, {d2['phase']:.2f} fase."}]).choices.message.content
            st.write(res)
