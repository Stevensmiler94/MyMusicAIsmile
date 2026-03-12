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
st.set_page_config(page_title="AI Studio Pro: Producer Edition", layout="wide", page_icon="🎧")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- UTILS AUDIO ---
def apply_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

@st.cache_data
def get_pro_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=45, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    
    # 1. AIR & TENSION (Analisi Garrix/Avicii)
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    air_energy = np.mean(S[np.where(freqs >= 12000), :]) / (np.mean(S) + 1e-9)
    rms = librosa.feature.rms(y=y_mono)
    tension = np.std(rms) * 100
    
    # 2. BASE STATS
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-9))
    phase = np.corrcoef(y_stereo)[0,1]
    
    return {"y": y_mono, "sr": sr, "lufs": lufs, "phase": phase, "crest": crest,
            "air": air_energy, "tension": tension}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genre Preset:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    
    sys_prompts = {
        "Progressive House (Garrix/Avicii)": "Sei un produttore alla STMPD RCRDS. Focus su Euforia, Sidechain e Air (>12kHz).",
        "Techno": "Sei un produttore Techno. Focus su Rumble e Kick Drive.",
        "Pop/Urban": "Sei un produttore Pop. Focus su Vocals e 808."
    }

    st.divider()
    up_json = st.file_uploader("📂 Importa Progetto", type="json")
    if up_json:
        d_l = json.load(up_json)
        st.session_state.progetti[d_l["nome"]] = d_l["dati"]
        st.session_state.progetto_attivo = d_l["nome"]; st.rerun()

    st.session_state.progetto_attivo = st.selectbox("Scenario", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name=f"{st.session_state.progetto_attivo}.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo}")
t1, t2, t3 = st.tabs(["📝 Songwriting", "🎚️ Mixing Desk", "🏆 Mastering Comparison"])

with t1:
    st.header("Emotional Ideation")
    audio_recorder(text="Registra idea")
    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.write(m["content"])
    if p := st.chat_input("Scrivi qui..."):
        if api_key:
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un esperto songwriter."}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p}])
            ans = res.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user","content":p})
            st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant","content":ans}); st.rerun()

with t2:
    st.header("Analisi Mix")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        d = get_pro_stats(f_mix)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.1f}%")
        c3.metric("Tension Score", f"{d['tension']:.1f}")
        c4.metric("Punch/Crest", f"{d['crest']:.1f} dB")
        
        target = st.radio("Isolamento:", ["Tutto", "Bassi", "Medi", "Alti (Air)"], horizontal=True)
        f_a = d['y']
        if target == "Bassi": f_a = apply_filter(d['y'], 20, 200, d['sr'])
        elif target == "Alti (Air)": f_a = apply_filter(d['y'], 8000, 15000, d['sr'])
        st.audio(f_a, sample_rate=d['sr'])

        if st.button("🪄 Analisi Strategica AI"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt = f"Dati: Air {d['air']:.2f}, Tension {d['tension']:.1f}, LUFS {d['lufs']:.1f}. Analizza stile {genere}."
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompts[genere]}, {"role":"user","content":prompt}])
                st.success(res.choices[0].message.content)

with t3:
    st.header("🏆 Mastering Comparison (A/B Test)")
    st.info("Confronta il tuo mix con una traccia Reference di Garrix o Avicii.")
    col_a, col_b = st.columns(2)
    f_my = col_a.file_uploader("Mio Mix", type=["wav", "mp3"], key="c_my")
    f_ref = col_b.file_uploader("Reference Pro", type=["wav", "mp3"], key="c_ref")
    
    if f_my and f_ref:
        d_my, d_ref = get_pro_stats(f_my), get_pro_stats(f_ref)
        
        # Dashboard Comparativa
        st.subheader("📊 Analisi Differenziale")
        m1, m2, m3 = st.columns(3)
        m1.metric("Delta LUFS (Volume)", f"{d_my['lufs']-d_ref['lufs']:.1f} LUFS", help="Cerca di arrivare a +/- 1.0 rispetto alla ref")
        m2.metric("Delta Air (Euphoria)", f"{(d_my['air']-d_ref['air'])*100:.1f}%", help="Se negativo, aggiungi un High Shelf")
        m3.metric("Delta Punch", f"{d_my['crest']-d_ref['crest']:.1f} dB")

        if st.button("🚀 Genera Piano d'Azione Mastering"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = (f"MIO: {d_my['lufs']:.1f} LUFS, Air {d_my['air']:.2%}, Punch {d_my['crest']:.1f}dB. "
                       f"REF: {d_ref['lufs']:.1f} LUFS, Air {d_ref['ref']:.2%}, Punch {d_ref['crest']:.1f}dB.")
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Sei un Mastering Engineer Pro."},{"role":"user","content":ctx + " Suggerisci i plugin esatti da usare."}])
                st.write(res.choices[0].message.content)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_c := st.chat_input("Cosa manca al mio drop?"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ans = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_prompts[genere]}]+st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]+[{"role":"user","content":p_c}])
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"user","content":p_c})
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"assistant","content":ans.choices[0].message.content}); st.rerun()
