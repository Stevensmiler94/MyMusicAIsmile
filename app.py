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
st.set_page_config(page_title="AI Studio Pro: Ultimate Edition", layout="wide", page_icon="🎧")

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
    # Carichiamo 30 secondi per l'analisi
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # Analisi Spettrale per EQ Curve
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_psd = np.mean(S, axis=1)
    
    # Air (>12kHz) & Tension (RMS StdDev)
    air_energy = np.mean(S[np.where(freqs >= 12000), :]) / (np.mean(S) + 1e-9)
    rms = librosa.feature.rms(y=y_mono)
    tension = np.std(rms) * 100
    
    # Stats base (Loudness & Crest)
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y_stereo.T)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    
    return {"y": y_mono, "sr": sr, "lufs": lufs, "crest": crest, "air": air_energy, 
            "tension": tension, "psd": avg_psd, "freqs": freqs}

# --- SIDEBAR: MANAGEMENT ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genre Preset:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Scenario (.json)", type="json")
    if up_json:
        try:
            d_load = json.load(up_json)
            st.session_state.progetti[d_load["nome"]] = d_load["dati"]
            st.session_state.progetto_attivo = d_load["nome"]
            st.rerun()
        except: st.error("File JSON non valido.")

    st.session_state.progetto_attivo = st.selectbox("Scenario Corrente", list(st.session_state.progetti.keys()))
    
    # Export
    curr_data = st.session_state.progetti[st.session_state.progetto_attivo]
    export_json = json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr_data}, indent=4)
    st.download_button("📥 Esporta Studio", export_json, file_name=f"{st.session_state.progetto_attivo}.json")

# --- UI PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison & EQ"])

# --- TAB 1: SONGWRITING ---
with t1:
    st.header("Emotional Songwriting")
    audio_recorder(text="Registra idea vocale")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    
    if p_s := st.chat_input("Consigli su melodia o testo..."):
        if not api_key: st.warning("Inserisci l'API Key!")
        else:
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un esperto songwriter."}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p_s}])
            ans = r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p_s})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":ans})
            st.rerun()

# --- TAB 2: MIXING DESK ---
with t2:
    st.header("Analisi Mix")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix_desk")
    if f_mix:
        d = get_pro_stats(f_mix)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.1f}%")
        c3.metric("Tension Score", f"{d['tension']:.1f}")
        c4.metric("Punch (Crest)", f"{d['crest']:.1f} dB")
        
        target = st.radio("Isolamento Frequenze:", ["Tutto", "Bassi", "Alti (Air)"], horizontal=True)
        f_play = d['y']
        if target == "Bassi": f_play = apply_filter(d['y'], 20, 250, d['sr'])
        elif target == "Alti (Air)": f_play = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_play, sample_rate=d['sr'])

        if st.button("🪄 Checklist Strategica AI"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt_mix = f"Dati: Air {d['air']:.2%}, Tension {d['tension']:.1f}, Crest {d['crest']:.1f}dB. Stile: {genere}."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un Mix Engineer."}, {"role":"user","content":prompt_mix}])
                st.success(r.choices[0].message.content)

# --- TAB 3: COMPARISON ---
with t3:
    st.header("Visual EQ Match & A/B Test")
    col_l, col_r = st.columns(2)
    f_my = col_l.file_uploader("Il tuo Mix", type=["wav", "mp3"], key="my_mix")
    f_ref = col_r.file_uploader("Reference Pro", type=["wav", "mp3"], key="ref_pro")
    
    if f_my and f_ref:
        d1, d2 = get_pro_stats(f_my), get_pro_stats(f_ref)
        
        # Grafico EQ Match
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio Mix", color="cyan", lw=2)
        ax.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Reference Pro", color="orange", alpha=0.6, lw=2)
        ax.set_title("EQ Curve Comparison (Log Scale)")
        ax.set_xlim(20, 20000); ax.legend(); ax.grid(True, which="both", ls="-", alpha=0.1)
        st.pyplot(fig)
        
        # Delta Metrics
        st.write("### 📊 Differenze Rispetto alla Reference")
        dm1, dm2 = st.columns(2)
        dm1.metric("Delta Air", f"{(d1['air']-d2['air'])*100:.1f}%")
        dm2.metric("Delta Loudness", f"{d1['lufs']-d2['lufs']:.1f} LUFS")

        if st.button("🚀 Ottieni Analisi Mastering"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx_comp = (f"MIO: {d1['lufs']:.1f} LUFS, Air {d1['air']:.2%}, Punch {d1['crest']:.1f}dB. "
                            f"REF: {d2['lufs']:.1f} LUFS, Air {d2['air']:.2%}, Punch {d2['crest']:.1f}dB.")
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un Mastering Engineer."}, {"role":"user","content":ctx_comp}])
                st.info(r.choices[0].message.content)

        # Chat Comparazione
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        if p_c := st.chat_input("Perché il mio drop suona meno potente della ref?"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un produttore senior."}]+st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]+[{"role":"user","content":p_c}])
                ans_c = r.choices[0].message.content
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"user","content":p_c})
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"assistant","content":ans_c})
                st.rerun()
