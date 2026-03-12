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
st.set_page_config(page_title="AI Studio Pro: Garrix/Avicii Edition", layout="wide", page_icon="🎧")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- UTILS AUDIO ---
def apply_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low, high = max(0.01, lowcut / nyq), min(0.99, highcut / nyq)
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

@st.cache_data
def get_pro_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # Spettro per EQ Curve
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_psd = np.mean(S, axis=1)
    
    # Air & Tension
    air_energy = np.mean(S[np.where(freqs >= 12000), :]) / (np.mean(S) + 1e-9)
    rms = librosa.feature.rms(y=y_mono)
    tension = np.std(rms) * 100
    
    # Stats base
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    
    return {"y": y_mono, "sr": sr, "lufs": lufs, "crest": crest, "air": air_energy, 
            "tension": tension, "psd": avg_psd, "freqs": freqs}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genre Preset:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Progetto", type="json")
    if up_json:
        d_load = json.load(up_json)
        st.session_state.progetti[d_load["nome"]] = d_load["dati"]
        st.session_state.progetto_attivo = d_load["nome"]; st.rerun()

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name=f"{st.session_state.progetto_attivo}.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison & EQ Curve"])

with t2:
    st.header("Analisi Energetica")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="mix")
    if f_mix:
        d = get_pro_stats(f_mix)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.1f}%")
        c3.metric("Tension Score", f"{d['tension']:.1f}")
        c4.metric("Punch (Crest)", f"{d['crest']:.1f} dB")
        
        st.write("**Monitor Isolamento**")
        target = st.radio("Filtro:", ["Tutto", "Bassi", "Alti (Air)"], horizontal=True)
        f_a = d['y']
        if target == "Bassi": f_a = apply_filter(d['y'], 20, 250, d['sr'])
        elif target == "Alti (Air)": f_a = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_a, sample_rate=d['sr'])

with t3:
    st.header("Visual EQ Comparison")
    col_l, col_r = st.columns(2)
    f1 = col_l.file_uploader("Il tuo Mix", type=["wav", "mp3"], key="c1")
    f2 = col_r.file_uploader("Reference Pro", type=["wav", "mp3"], key="c2")
    
    if f1 and f2:
        d1, d2 = get_pro_stats(f1), get_pro_stats(f2)
        
        # Grafico EQ Sovrapposto
        st.subheader("📈 Analisi Spettrale Comparativa")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio Mix", color="cyan")
        ax.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Reference Pro", color="orange", alpha=0.6)
        ax.set_xlim(20, 20000); ax.legend(); ax.grid(True, which="both", ls="-", alpha=0.2)
        st.pyplot(fig)
        
        # AI Insight
        if st.button("🚀 Analizza Differenze con AI"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = (f"Mio: {d1['lufs']:.1f} LUFS, Air {d1['air']:.2%}, Tension {d1['tension']:.1f}. "
                       f"Ref: {d2['lufs']:.1f} LUFS, Air {d2['air']:.2%}, Tension {d2['tension']:.1f}.")
                ans = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un produttore Progressive House. Confronta i dati e suggerisci interventi su EQ e Sidechain."}, {"role":"user","content":ctx}])
                st.success(ans.choices[0].message.content)

with t1:
    st.header("Songwriting Chat")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p := st.chat_input("Consigli melodia..."):
        if api_key:
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un esperto songwriter."}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p}])
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":res.choices[0].message.content}); st.rerun()
