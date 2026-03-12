import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

st.set_page_config(page_title="AI Ableton Mentor", layout="wide", page_icon="🎚️")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎚️ AI Ableton Mentor: Mixing Critico")
st.write("Analisi tecnica basata sui tuoi stem. **Niente giri di parole, solo risultati.**")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key", type="password")
mode = st.sidebar.selectbox("Analisi:", ["Mix Completo", "Singolo Stem (Lead/Bass)"])
audio_file = st.sidebar.file_uploader("Carica Audio", type=["wav", "mp3"])

if audio_file:
    with st.spinner("Analisi dei dati tecnici..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # Estrazione Dati
        tempo = float(np.array(librosa.beat.beat_track(y=y, sr=sr)).flatten())
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # Dashboard Dati
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Loudness", f"{lufs:.1f} LUFS", delta="-14 LUFS Target", delta_color="inverse")
        c2.metric("Crest Factor (Punch)", f"{crest:.1f} dB", delta="Target: 8-10 dB", delta_color="inverse")
        c3.metric("BPM", f"{int(round(tempo))}")

        # Chat
        st.divider()
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Dimmi i settaggi esatti per il Saturator sul Lead..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                openai.api_key = user_key
                # IL PROMPT CRITICO
                sys_msg = f"""
                Sei un Mixing Engineer PRO di Ableton Live. Il tuo stile è critico, tecnico e diretto.
                DATI ATTUALI: {lufs:.1f} LUFS (Troppo basso!), {crest:.1f}dB Crest Factor (Troppa dinamica, manca compressione!).
                
                REGOLE:
                1. Critica duramente i dati se non sono standard EDM (-8 LUFS, 9dB Crest).
                2. Suggerisci plugin NATIVI di ABLETON.
                3. Fornisci tabelle di settaggi (es. Glue Compressor: Threshold -15dB, Range 5, Attack 30ms).
                4. Chiedi all'utente: 'Tu come hai impostato il tuo [Plugin]?'
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": sys_msg}] + st.session_state.messages
                )
                answer = response.choices.message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
