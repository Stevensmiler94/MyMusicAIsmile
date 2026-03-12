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
st.set_page_config(page_title="AI Studio Pro: Final Edition", layout="wide", page_icon="🎹")

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
    
    # 1. EQ & Air
    def get_e(f1, f2):
        idx = np.where((freqs >= f1) & (freqs <= f2))
        return np.mean(S[idx, :]) if len(idx) > 0 else 0
    l, m, h = get_e(20, 250), get_e(250, 4500), get_e(4500, 20000)
    bands = {"Bassi": l, "Medi": m, "Alti": h}
    air_val = h / (l + m + h + 1e-9)
    
    # 2. Sub Mono Check (<100Hz)
    y_low_L = apply_filter(y_stereo, 20, 100, sr)
    y_low_R = apply_filter(y_stereo, 20, 100, sr)
    sub_mono_corr = float(np.corrcoef(y_low_L, y_low_R))
    
    # 3. Stereo Correlation & Width
    corr_matrix = np.corrcoef(y_stereo, y_stereo)
    phase_compat = float(corr_matrix)
    width_val = float((1 - phase_compat) * 100)
    
    # 4. Stats
    lufs = pdn.Meter(sr).integrated_loudness(y_stereo.T)
    rms = librosa.feature.rms(y=y_mono)
    crest = 20 * np.log10(np.max(np.abs(y_mono)) / (np.sqrt(np.mean(y_mono**2)) + 1e-9))
    
    return {"y": y_mono, "y_s": y_stereo, "sr": sr, "lufs": float(lufs), "crest": float(crest), "air": air_val, 
            "psd": np.mean(S, axis=1), "freqs": freqs, "width": width_val, "sub_mono": sub_mono_corr, 
            "phase": phase_compat, "bands": bands}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Studio Manager")
    api_key = st.text_input("OpenAI API Key", type="password")
    genere = st.selectbox("Genere:", ["Progressive House (Garrix/Avicii)", "Techno", "Pop/Urban"])
    sys_instruction = f"Rispondi SEMPRE in ITALIANO. Sei un produttore esperto di {genere}."
    
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

# --- TAB 1: SONGWRITING & LYRICS ---
with t1:
    st.header("Creatività & Testi AI")
    col_lyrics, col_chat = st.columns([1, 2])
    
    with col_lyrics:
        st.subheader("🖋️ Lyrics Generator")
        mood = st.text_input("Mood (es. nostalgia, alba, libertà)", "euphoria")
        if st.button("✨ Genera Testo"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt_ly = f"Scrivi il testo per una canzone {genere} sul tema {mood}. Includi Strofa, Pre-Chorus e Chorus. Lingua: Inglese (stile Avicii/Garrix)."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system", "content":"Sei un paroliere di hit mondiali EDM."}, {"role":"user", "content":prompt_ly}])
                st.text_area("Testo Generato", r.choices[0].message.content, height=400)
    
    with col_chat:
        st.subheader("💬 Songwriting Chat")
        audio_recorder(text="Registra idea melodica")
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        if p_s := st.chat_input("Consigli melodia o armonia..."):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}]+st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]+[{"role":"user","content":p_s}])
                st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"user", "content":p_s})
                st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role":"assistant", "content":r.choices[0].message.content}); st.rerun()

# --- TAB 2: MIXING DESK ---
with t2:
    st.header("Analisi Platinum Elite + Visual Stereo")
    f_mix = st.file_uploader("Carica Mix", type=["wav", "mp3"], key="u_mix")
    if f_mix:
        d = get_platinum_stats(f_mix)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Loudness", f"{d['lufs']:.1f} LUFS")
        c2.metric("Euphoric Air", f"{d['air']*100:.2f}%")
        c3.metric("Stereo Width", f"{d['width']:.1f}%")
        c4.metric("Sub Mono", f"{d['sub_mono']:.2f}", delta="OK" if d['sub_mono']>0.9 else "Check Phase")
        c5.metric("EQ Focus", max(d['bands'], key=d['bands'].get))

        st.subheader("📊 Analisi Tecnica Visiva")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.write("**Waveform**")
            fig1, ax1 = plt.subplots(figsize=(6, 4)); librosa.display.waveshow(d['y'], sr=d['sr'], ax=ax1, color="skyblue"); st.pyplot(fig1)
        with v2:
            st.write("**Goniometro (Stereo Spread)**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(d['y_s'][0, ::120], d['y_s'][1, ::120], s=1, color="lime", alpha=0.3)
            ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.8, 0.8); ax2.set_aspect('equal'); plt.grid(alpha=0.1); st.pyplot(fig2)
        with v3:
            st.write("**Spettrogramma**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            S_db = librosa.power_to_db(np.abs(librosa.stft(d['y'])), ref=np.max)
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=d['sr'], ax=ax3); st.pyplot(fig3)

        st.subheader("🎧 Isolamento Spettrale")
        target = st.radio("Filtro:", ["Tutto", "Bassi (<250Hz)", "Medi (250-4500Hz)", "Alti (>8kHz)"], horizontal=True)
        f_p = d['y']
        if target == "Bassi (<250Hz)": f_p = apply_filter(d['y'], 20, 250, d['sr'])
        elif target == "Medi (250-4500Hz)": f_p = apply_filter(d['y'], 250, 4500, d['sr'])
        elif target == "Alti (>8kHz)": f_p = apply_filter(d['y'], 8000, 16000, d['sr'])
        st.audio(f_p, sample_rate=d['sr'])

        if st.button("🪄 Genera Checklist Strategica (ITA)"):
            if api_key:
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}, {"role":"user","content":f"Analizza: Air {d['air']:.2%}, Width {d['width']:.1f}%, Sub-Mono {d['sub_mono']:.2f}."}])
                st.success(r.choices[0].message.content)

# --- TAB 3: COMPARISON ---
with t3:
    st.header("EQ Comparison & A/B Test")
    cl, cr = st.columns(2); f1, f2 = cl.file_uploader("Mio Mix", key="c1"), cr.file_uploader("Reference", key="c2")
    if f1 and f2:
        d1, d2 = get_platinum_stats(f1), get_platinum_stats(f2)
        fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
        ax_eq.semilogx(d1['freqs'], librosa.amplitude_to_db(d1['psd']), label="Mio", color="cyan", lw=2)
        ax_eq.semilogx(d2['freqs'], librosa.amplitude_to_db(d2['psd']), label="Ref", color="orange", alpha=0.6, lw=2)
        ax_eq.set_xlim(20, 20000); ax_eq.legend(); plt.grid(True, which="both", alpha=0.1); st.pyplot(fig_eq)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Delta LUFS", f"{d1['lufs']-d2['lufs']:.1f}")
        m2.metric("Delta Sub Mono", f"{d1['sub_mono']-d2['sub_mono']:.2f}")
        m3.metric("Delta Air", f"{(d1['air']-d2['air'])*100:.1f}%")

        if st.button("🚀 Ottieni Strategia Mastering"):
            if api_key:
                client = OpenAI(api_key=api_key)
                ctx_m = f"MIO: {d1['lufs']:.1f}LUFS. REF: {d2['lufs']:.1f}LUFS."
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_instruction}, {"role":"user","content":ctx_m}])
                st.info(r.choices[0].message.content)
