import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

st.set_page_config(page_title="AI Music Producer Assistant", layout="wide")

st.title("🎧 AI Mix Assistant: Progressive House Edition")
st.write("Analisi professionale ispirata allo stile di **Martin Garrix & Avicii**.")

# --- SIDEBAR ---
st.sidebar.header("Impostazioni & Upload")
user_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")

mode = st.sidebar.selectbox("Cosa vuoi analizzare?", ["Mix Completo vs Reference", "Traccia Singola (Lead, Bass, Kick)"])

if mode == "Mix Completo vs Reference":
    uploaded_mix = st.sidebar.file_uploader("Il tuo Mix (.wav)", type=["wav", "mp3"], key="mix")
    uploaded_ref = st.sidebar.file_uploader("Traccia Reference (.wav)", type=["wav", "mp3"], key="ref")
else:
    uploaded_stem = st.sidebar.file_uploader("Carica la tua Traccia Singola (es. solo Lead)", type=["wav", "mp3"], key="stem")
    uploaded_ref = None

# --- LOGICA DI ANALISI ---
if (mode == "Mix Completo vs Reference" and uploaded_mix and uploaded_ref) or (mode == "Traccia Singola (Lead, Bass, Kick)" and uploaded_stem):
    
    with st.spinner("Analizzando l'audio..."):
        # Caricamento Audio
        audio_to_analyze = uploaded_mix if mode == "Mix Completo vs Reference" else uploaded_stem
        y_mix, sr = librosa.load(audio_to_analyze, duration=30)
        
        if uploaded_ref:
            y_ref, _ = librosa.load(uploaded_ref, duration=30)
        
        # 1. LOUDNESS (LUFS)
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            meter = pdn.Meter(rate)
            return meter.integrated_loudness(data)

        lufs_m = get_lufs(y_mix, sr)
        
        # 2. EQ & SPETTRO
        spec_m = np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1)

        # --- VISUALIZZAZIONE ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Loudness Traccia", f"{lufs_m:.1f} LUFS")
        
        # GRAFICO EQ
        st.subheader("📊 Bilanciamento Spettrale (EQ)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(spec_m, label="Tua Traccia", color="cyan")
        if uploaded_ref:
            spec_r = np.mean(librosa.feature.melspectrogram(y=y_ref, sr=sr), axis=1)
            ax.plot(spec_r, label="Reference", color="orange", alpha=0.5)
        ax.set_yscale('log')
        ax.legend()
        st.pyplot(fig)

        # WAVEFORM & ZOOM
        st.subheader("🌊 Forma d'Onda & Dinamica")
        fig_w, ax_w = plt.subplots(2, 1, figsize=(10, 5))
        librosa.display.waveshow(y_mix, sr=sr, ax=ax_w[0], color='cyan')
        ax_w[0].set_title("Intero (30s)")
        
        # Zoom primi 5 secondi (per il punch)
        librosa.display.waveshow(y_mix[:int(5*sr)], sr=sr, ax=ax_w[1], color='magenta')
        ax_w[1].set_title("Zoom Attacco (5s)")
        plt.tight_layout()
        st.pyplot(fig_w)

        # 3. FEEDBACK IA PERSONALIZZATO
        st.divider()
        if st.button("Genera Analisi Tecnica Avanzata"):
            if user_key:
                openai.api_key = user_key
                tipo = "Mix finale" if mode == "Mix Completo vs Reference" else "Traccia singola (Stem)"
                prompt = f"""
                Analizza questa traccia {tipo}. 
                Dati tecnici: Loudness {lufs_m:.1f} LUFS. 
                Obiettivo: Suono Progressive House stile Garrix/Avicii.
                Se è un Mix: commenta equilibrio e sidechain.
                Se è un Lead: commenta brillantezza, riverbero e layering.
                Sii molto tecnico e diretto.
                """
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
                st.success(resp.choices.message.content)
            else:
                st.error("Inserisci la API Key nella sidebar!")
