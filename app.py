import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Ableton Mentor Pro", layout="wide", page_icon="🎚️")

# Inizializzazione cronologia chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎚️ AI Ableton Mentor: Analisi Tecnica Reale")
st.write("L'IA analizza i dati del tuo file e ti dà consigli professionali su Ableton.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
mode = st.sidebar.selectbox("Cosa analizziamo?", ["Mix Completo", "Singolo Stem (Lead/Bass/Kick)"])
audio_file = st.sidebar.file_uploader("Carica il tuo file audio", type=["wav", "mp3"])

# --- ANALISI AUDIO ---
if audio_file:
    with st.spinner("⚡ Analisi dei dati in corso..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # FIX BPM
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        bpm_final = float(tempo_result[0]) if isinstance(tempo_result, (tuple, list, np.ndarray)) else float(tempo_result)
        
        # Scala e Dati Tecnici
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_detected = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # Dashboard
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest (Punch)", f"{crest:.1f} dB")
        c3.metric("BPM", f"{int(round(bpm_final))}")
        c4.metric("Scala", key_detected)

        # Chat Interattiva
        st.divider()
        st.subheader("💬 Chiedi all'Ingegnere (L'IA ora vede i tuoi dati)")
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Kick e basso suonano bene?"):
            if user_key:
                openai.api_key = user_key
                
                # ISTRUZIONE DI SISTEMA CON I DATI REALI CARICATI
                # Questo forza l'IA a NON dire "non posso ascoltare"
                sys_msg = f"""
                Sei un Senior Mixing Engineer esperto di Ableton. 
                NON dire che non puoi analizzare il file. Io ho analizzato il file per te e questi sono i dati reali:
                - Loudness: {lufs:.1f} LUFS (Se > -10 il mix è moscio per EDM).
                - Crest Factor: {crest:.1f} dB (Se > 12 il kick non ha punch).
                - BPM: {int(bpm_final)}, Scala: {key_detected}.

                ANALISI KICK/BASSO: 
                - Se il Crest Factor è alto (>11), di' che il Kick è debole e suggerisci 'Glue Compressor' (Soft Clip ON).
                - Suggerisci 'EQ Eight' per tagliare il basso sotto i 30Hz e il Kick a 20Hz.
                - Parla di 'Sidechain' usando il Compressor nativo di Ableton.
                - Dai settaggi numerici precisi. Sii critico e tecnico.
                """

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": sys_msg}] + st.session_state.messages
                        )
                        answer = resp.choices.message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore: {e}")
            else:
                st.warning("Inserisci la API Key!")

# Glossario sempre visibile
st.divider()
with st.expander("📖 Glossario"):
    st.write("LUFS: Volume reale. Crest Factor: Potenza del Kick. OTT: Brillantezza.")
