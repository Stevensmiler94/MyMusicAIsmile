import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

st.set_page_config(page_title="AI Ableton Mentor Pro", layout="wide", page_icon="🎚️")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎚️ AI Ableton Mentor: Analisi Reale")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key", type="password")
audio_file = st.sidebar.file_uploader("Carica file audio", type=["wav", "mp3"])

if audio_file:
    with st.spinner("⚡ Estrazione dati tecnici..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # Analisi Tecnica
        tempo_data = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.ravel(tempo_data)[0])
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # Dashboard
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", f"{int(bpm)}")

        # Chat
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Analizza il mio kick e basso..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                openai.api_key = user_key
                
                # IL PROMPT SEGRETO: Obbliga l'IA a usare i dati
                content_ai = f"""
                TU SEI UN MIXING ENGINEER DI ABLETON. NON DIRE CHE NON PUOI ASCOLTARE.
                ECCO I DATI REALI CHE HO ESTRATTO DAL FILE:
                - Loudness: {lufs:.1f} LUFS
                - Crest Factor: {crest:.1f} dB
                - BPM: {int(bpm)}

                ANALISI CRITICA:
                1. Se Crest > 12dB: Il Kick è debole e moscio. Suggerisci Saturator (Sinoid Fold) o Glue Comp (Soft Clip ON).
                2. Se LUFS > -12dB: Il mix non ha energia festival. Suggerisci Limiter o OTT.
                3. Dai parametri esatti per Ableton (Threshold, Attack, Dry/Wet).
                Rispondi basandoti SOLO su questi numeri. Sii cattivo e tecnico.
                """
                
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": content_ai}] + st.session_state.messages
                    )
                    answer = resp['choices'][0]['message']['content']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Errore: {e}")
