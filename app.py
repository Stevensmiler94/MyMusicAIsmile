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

# --- ENGINE AUDIO AVANZATO ---
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
    
    # 1. Analisi Spettrale (EQ) e Air
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_psd = np.mean(S, axis=1)
    idx_air = np.where(freqs >= 10000)
    air_val = np.mean(S[idx_air, :]) / (np.mean(S) + 1e-9) if len(idx_air) > 0 else 0.0
    
    # 2. Analisi KICK-BASS Relationship (Low End Masking)
    # Calcoliamo la stabilità dell'energia sotto i 100Hz
    low_idx = np.where(freqs <= 100)
    low_energy_ts = np.mean(S[low_idx, :], axis=0)
    # Se l'energia è troppo costante (std bassa), il basso copre il kick. 
    # Se è ritmica (std alta), il sidechain sta funzionando.
    kick_bass_score = np.std(low_energy_ts) * 100 
    
    # 3. Stereo Width & Phase
    correlation = np.corrcoef(y_stereo, y_stereo)[0, 1]
    stereo_width = (1 - correlation) * 100 
    
    # 4. Loudness & Punch
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y_stereo.T)
    rms = librosa.feature.rms(y=y_mono)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    tension = np.std(rms) * 100
    
    return {
        "y": y_mono, "sr": sr, "lufs": lufs, "crest": crest, "air": air_val, 
        "tension": tension, "psd": avg_psd, "freqs": freqs, "width": stereo_width,
        "kick_bass": kick_bass_score, "phase": correlation
    }

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genere Target:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    
    st.divider()
    if st.button("🔄 Reset Totale"):
        st.session_state.clear(); st.rerun()
        
    up_json = st.file_uploader("📂 Importa Scenario", type="json")
    if up_json:
        try:
            d_l = json.load(up_json)
            st.session_state.progetti[d_l["nome"]] = d_l["dati"]
            st.session_state.progetto_attivo = d_l["nome"]; st.rerun()
        except: st.error("Errore file")

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    
    if st.button("➕ Nuovo Scenario"):
        n = f"Scenario {len(st.session_state.progetti)+1}"
        st.session_state.progetti[n] = crea_struttura_progetto()
        st.session_state.progetto_attivo = n; st.rerun()

    exp = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]})
    st.download_button("📥 Esporta Studio", exp, file_name=f"{st.session_state.progetto_attivo}.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo} | Mode: {genere}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Comparison & EQ"])

sys_instruction = f"Rispondi SEMPRE in ITALIANO. Sei un produttore esperto di {genere}. Sii molto tecnico."

# --- TAB 1: SONGWRITING ---
with t1:
    st.header("Ideazione & Testi")
    audio_recorder(text="Registra idea vocale")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p_s := st.chat_input("Consigli melodia/testo..."):
        if api_key:
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p_s}])
            ans = r.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p_s})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":ans}); st.rerun()

# --- TAB 2: MIXING DESK ---
with t2:
    st.header("Analisi Platinum")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        d = get_platinum_stats(f_mix)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.2f}%")
        c3.metric("Stereo Width", f"{d['width']:.1f}%")
        c4.metric("Kick-Bass Space", f"{d['kick_bass']:.1f}", help="Più alto = più spazio per il kick")
        c5.metric("Punch", f"{d['crest']:.1f} dB")

        st.subheader("🎧 Isolamento Spettrale")
        target = st.radio("Ascolta:", ["Tutto", "Sub/Bassi (<200Hz)", "Alti (>8kHz)"], horizontal=True)
        f_p = d['y']
        if target == "Sub/Bassi (<200Hz)": f_p = apply_filter(d['y'], 20, 200, d['sr'])
        elif target == "Alti (>8kHz)": f_p = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_p, sample_rate=d['sr'])

        if st.button("🪄 Genera Analisi Kick-Bass & Mix"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt = f"Dati: Air {d['air']:.2%}, Width {d['width']:.1f}%, Kick-Bass Space {d['kick_bass']:.1f}, LUFS {d['lufs']:.1f}. Analizza in italiano."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}, {"role":"user","content":prompt}])
                st.success(r.choices[0].message.content)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_m := st.chat_input("Chiedi al fonico (ITA)..."):
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]+[{"role":"user","content":p_m}])
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"user","content":p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"assistant","content":r.choices[0].message.content}); st.rerun()

# --- TAB 3: COMPARISON ---
with t3:
    st.header("EQ Match & Comparison")
    cl, cr = st.columns(2)
    f1, f2 = cl.file_uploader("Mio Mix", key="c1"), cr.file_uploader("Ref", key="c2")
    if f1 and f2:
        d1, d2 = get_platinum_stats(f1), get_platinum_stats(f2)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio", color="cyan", lw=2)
        ax.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Ref", color="orange", alpha=0.6, lw=2)
        ax.set_xlim(20, 20000); ax.legend(); plt.grid(True, which="both", alpha=0.1)
        st.pyplot(fig)
        
        st.metric("Delta Kick-Bass Space", f"{d1['kick_bass']-d2['kick_bass']:.1f}", help="Se negativo, il tuo kick è meno pulito della ref")
        
        if st.button("🚀 Strategia Mastering Platinum (ITA)"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = f"MIO: LUFS {d1['lufs']:.1f}, Air {d1['air']:.2%}, KB-Space {d1['kick_bass']:.1f}. REF: LUFS {d2['lufs']:.1f}, Air {d2['air']:.2%}, KB-Space {d2['kick_bass']:.1f}."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction + " Rispondi in italiano."}, {"role":"user","content":ctx}])
                st.info(r.choices[0].message.content)
