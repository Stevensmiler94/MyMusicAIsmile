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
st.set_page_config(page_title="AI Studio Pro: Master Edition", layout="wide", page_icon="🎧")

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
def get_ultimate_stats(file_bytes):
    # Caricamento Stereo per ampiezza e fase
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # 1. Analisi Spettrale & FIX AIR (Frequenze > 10kHz)
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_psd = np.mean(S, axis=1)
    idx_air = np.where(freqs >= 10000)[0]
    air_val = np.mean(S[idx_air, :]) / (np.mean(S) + 1e-9) if len(idx_air) > 0 else 0.0
    
    # 2. Analisi Stereo Width (Correlazione di fase)
    # +1 = Mono perfetto, 0 = Stereo Largo, -1 = Fuori Fase
    correlation = np.corrcoef(y_stereo[0], y_stereo[1])[0, 1]
    stereo_width = (1 - correlation) * 100 # Approssimazione larghezza percepitiva
    
    # 3. Tension (Energia RMS) & Punch (Crest Factor)
    rms = librosa.feature.rms(y=y_mono)
    tension = np.std(rms) * 100
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    
    return {
        "y": y_mono, "sr": sr, "lufs": lufs, "crest": crest, "air": air_val, 
        "tension": tension, "psd": avg_psd, "freqs": freqs, "phase": correlation, "width": stereo_width
    }

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genere Target:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    
    # Istruzione obbligatoria per l'italiano
    sys_prompt_base = f"Rispondi SEMPRE in ITALIANO. Sei un produttore esperto di {genere}."
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Scenario", type="json")
    if up_json:
        d = json.load(up_json)
        st.session_state.progetti[d["nome"]] = d["dati"]
        st.session_state.progetto_attivo = d["nome"]; st.rerun()

    st.session_state.progetto_attivo = st.selectbox("Progetto", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name="studio.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo} | {genere}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison & Stereo"])

# --- TAB 2: MIXING ---
with t2:
    st.header("Analisi Mix & Energia")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        with st.spinner("Analisi in corso..."):
            d = get_ultimate_stats(f_mix)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.2f}%")
        c3.metric("Stereo Width", f"{d['width']:.1f}%")
        c4.metric("Punch", f"{d['crest']:.1f} dB")
        
        st.subheader("🎧 Monitor di Isolamento")
        target = st.radio("Ascolta:", ["Tutto", "Bassi", "Alti (Air)"], horizontal=True)
        f_play = d['y']
        if target == "Bassi": f_play = apply_filter(d['y'], 20, 250, d['sr'])
        elif target == "Alti (Air)": f_play = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_play, sample_rate=d['sr'])

        if st.button("🪄 Analisi Strategica (ITA)"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt = (f"Dati: Air {d['air']:.4f}, Width {d['width']:.1f}%, "
                          f"Crest {d['crest']:.1f}dB, LUFS {d['lufs']:.1f}. Analizza in italiano.")
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompt_base}, {"role":"user","content":prompt}])
                st.success(r.choices.message.content)

# --- TAB 3: COMPARISON ---
with t3:
    st.header("Visual EQ & Stereo Match")
    cl, cr = st.columns(2)
    f1, f2 = cl.file_uploader("Mio Mix", type=["wav", "mp3"], key="c1"), cr.file_uploader("Reference", type=["wav", "mp3"], key="c2")
    if f1 and f2:
        d1, d2 = get_ultimate_stats(f1), get_ultimate_stats(f2)
        
        # Grafico EQ
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio Mix", color="cyan", lw=2)
        ax.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Reference", color="orange", alpha=0.6, lw=2)
        ax.set_xlim(20, 20000); ax.legend(); plt.grid(True, which="both", alpha=0.1)
        st.pyplot(fig)
        
        st.write("### 📊 Confronto Tecnico")
        m1, m2, m3 = st.columns(3)
        m1.metric("Delta Width (Stereo)", f"{d1['width']-d2['width']:.1f}%")
        m2.metric("Delta Air (Euphoria)", f"{(d1['air']-d2['air'])*100:.1f}%")
        m3.metric("Delta Punch", f"{d1['crest']-d2['crest']:.1f} dB")

        if st.button("🚀 Strategia Mastering A/B (ITA)"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = (f"MIO: {d1['lufs']:.1f}LUFS, Width {d1['width']:.1f}%, Air {d1['air']:.4f}. "
                       f"REF: {d2['lufs']:.1f}LUFS, Width {d2['width']:.1f}%, Air {d2['air']:.4f}.")
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompt_base + " Rispondi in italiano."}, {"role":"user","content":ctx}])
                st.info(r.choices.message.content)

# --- TAB 1: SONGWRITING ---
with t1:
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p := st.chat_input("Chiedi un consiglio in italiano..."):
        if api_key:
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Rispondi sempre in italiano come esperto songwriter."}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p}])
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":r.choices.message.content}); st.rerun()
