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
st.set_page_config(page_title="AI Studio Pro: Quantum Edition", layout="wide", page_icon="🎹")

def crea_struttura_progetto():
    return {"songwriting": [], "mixing": [], "comparison": []}

if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": crea_struttura_progetto()}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- ENGINE AUDIO CORRETTO ---
def apply_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if low >= high: low = high * 0.5
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

@st.cache_data
def get_platinum_stats(file_bytes):
    y_stereo, sr = librosa.load(file_bytes, duration=30, mono=False)
    if y_stereo.ndim == 1: y_stereo = np.vstack([y_stereo, y_stereo])
    y_mono = librosa.to_mono(y_stereo)
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # EQ & Air
    def get_e(f1, f2):
        idx = np.where((freqs >= f1) & (freqs <= f2))
        return np.mean(S[idx, :]) if len(idx) > 0 else 1e-9
    l, m, h = get_e(20, 250), get_e(250, 4500), get_e(4500, 20000)
    bands = {"Bassi": l, "Medi": m, "Alti": h}
    air_val = h / (l + m + h + 1e-9)
    
    # SUB MONO (Protezione NaN)
    y_low_L = apply_filter(y_stereo[0], 20, 100, sr)
    y_low_R = apply_filter(y_stereo[1], 20, 100, sr)
    sub_corr = np.corrcoef(y_low_L, y_low_R)[0, 1]
    if np.isnan(sub_corr): sub_corr = 1.0
    
    # Stereo & Stats
    corr_val = np.corrcoef(y_stereo[0], y_stereo[1])[0, 1]
    if np.isnan(corr_val): corr_val = 1.0
    width_val = float((1 - corr_val) * 100)
    
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    rms = np.sqrt(np.mean(y_mono**2)) + 1e-9
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / rms)
    
    return {"y": y_mono, "y_s": y_stereo, "sr": sr, "lufs": float(lufs), "crest": float(crest), "air": float(air_val), 
            "psd": np.mean(S, axis=1), "freqs": freqs, "width": width_val, "sub_mono": float(sub_corr), "bands": bands}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genere:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    sys_inst = f"Rispondi SEMPRE in ITALIANO. Sei un produttore esperto di {genere}."
    
    st.divider()
    up_json = st.file_uploader("📂 Importa Scenario", type="json")
    if up_json:
        try:
            d_l = json.load(up_json); st.session_state.progetti[d_l["nome"]] = d_l["dati"]
            st.session_state.progetto_attivo = d_l["nome"]; st.rerun()
        except: st.error("Errore JSON")
    
    st.session_state.progetto_attivo = st.selectbox("Scenario Attivo", list(st.session_state.progetti.keys()))
    curr = st.session_state.progetti[st.session_state.progetto_attivo]
    st.download_button("📥 Esporta Studio", json.dumps({"nome": st.session_state.progetto_attivo, "dati": curr}), file_name="studio.json")

# --- UI TABS ---
st.title(f"🚀 {st.session_state.progetto_attivo} | {genere}")
t1, t2, t3 = st.tabs(["📝 Songwriting & Lyrics", "🎚️ Mixing Desk", "🏆 Comparison & EQ"])

# --- TAB 1: SONGWRITING ---
with t1:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🖋️ Lyrics Generator")
        mood = st.text_input("Mood", "hope and euphoria")
        if st.button("✨ Genera Testo"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un paroliere EDM."}, {"role": "user", "content": f"Testo {genere} tema {mood} in inglese."}])
                st.text_area("Lyrics", r.choices.message.content, height=300)
    with col_r:
        st.subheader("💬 Chat")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_s := st.chat_input("Consigli songwriting..."):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p_s}])
                st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user", "content":p_s})
                st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant", "content":r.choices.message.content}); st.rerun()

# --- TAB 2: MIXING ---
with t2:
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        d = get_platinum_stats(f_mix)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS"); c2.metric("Air", f"{d['air']*100:.1f}%")
        c3.metric("Stereo Width", f"{d['width']:.1f}%"); c4.metric("Sub Mono", f"{d['sub_mono']:.2f}")
        c5.metric("EQ Focus", max(d['bands'], key=d['bands'].get))

        v1, v2, v3 = st.columns(3)
        with v1: 
            fig1, ax1 = plt.subplots(); librosa.display.waveshow(d['y'], sr=d['sr'], ax=ax1, color="#00d1ff"); st.pyplot(fig1)
        with v2:
            fig2, ax2 = plt.subplots(); ax2.scatter(d['y_s'][0, ::150], d['y_s'][1, ::150], s=1, color="#00ff41", alpha=0.4)
            ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.8, 0.8); ax2.set_aspect('equal'); st.pyplot(fig2)
        with v3:
            fig3, ax3 = plt.subplots(); S_db = librosa.power_to_db(np.abs(librosa.stft(d['y'])), ref=np.max)
            librosa.display.specshow(S_db, sr=d['sr'], ax=ax3, y_axis='mel'); st.pyplot(fig3)

        st.subheader("🎧 Monitor & Isolamento")
        target = st.radio("Isola Banda:", ["Tutto", "Bassi (<250Hz)", "Medi (250-4500Hz)", "Alti (>8kHz)"], horizontal=True)
        f_p = d['y']
        if target == "Bassi (<250Hz)": f_p = apply_filter(d['y'], 20, 250, d['sr'])
        elif target == "Medi (250-4500Hz)": f_p = apply_filter(d['y'], 250, 4000, d['sr'])
        elif target == "Alti (>8kHz)": f_p = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_p, sample_rate=d['sr'])

        # Pulsanti Analisi
        cb1, cb2 = st.columns(2)
        if cb1.button("🪄 Checklist Tecnica"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}, {"role":"user","content":f"Dati: Air {d['air']:.2%}, Width {d['width']:.1f}%, Sub-Mono {d['sub_mono']:.2f}. Dammi 5 consigli pratici."}])
                st.success(r.choices.message.content)
        if cb2.button("🧠 DEEP AI REVIEW"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}, {"role":"user","content":f"Analizza l'equilibrio artistico di questo mix {genere}. Parametri: Air {d['air']:.2%}, Stereo {d['width']:.1f}%. Ponimi 3 domande filosofico-tecniche."}])
                st.info(r.choices.message.content)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_m := st.chat_input("Approfondisci un dettaglio..."):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}]+st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]+[{"role":"user","content":p_m}])
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"user", "content":p_m})
                st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role":"assistant", "content":r.choices.message.content}); st.rerun()

# --- TAB 3: COMPARISON ---
with t3:
    st.header("EQ Comparison & Deep Benchmarking")
    cl, cr = st.columns(2); f1, f2 = cl.file_uploader("Mio", key="c1"), cr.file_uploader("Ref", key="c2")
    if f1 and f2:
        d1, d2 = get_platinum_stats(f1), get_platinum_stats(f2)
        fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
        ax_eq.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio", color="#00d1ff")
        ax_eq.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Ref", color="#ff8700", alpha=0.6)
        ax_eq.set_xlim(20, 20000); ax_eq.legend(); plt.grid(True, which="both", alpha=0.1); st.pyplot(fig_eq)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Delta LUFS", f"{d1['lufs']-d2['lufs']:.1f}"); m2.metric("Delta Sub Mono", f"{d1['sub_mono']-d2['sub_mono']:.2f}"); m3.metric("Delta Air", f"{(d1['air']-d2['air'])*100:.1f}%")

        if st.button("🚀 Strategia & Deep Comparison"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx = f"MIO: {d1['lufs']:.1f}LUFS, Air {d1['air']:.1%}. REF: {d2['lufs']:.1f}LUFS, Air {d2['air']:.1%}. Confronta l'energia emotiva tra i due brani."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}, {"role":"user","content":ctx}])
                st.success(r.choices.message.content)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_c := st.chat_input("Perché il drop professionale suona più profondo?"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_inst}]+st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]+[{"role":"user","content":p_c}])
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"user", "content":p_c})
                st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role":"assistant", "content":r.choices.message.content}); st.rerun()
