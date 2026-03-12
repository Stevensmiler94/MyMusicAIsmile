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

# --- LOGICA DI ANALISI AUDIO ---
if audio_file:
    with st.spinner("⚡ Analisi dei dati in corso..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # --- FIX DEFINITIVO BPM ---
        tempo_data = librosa.beat.beat_track(y=y, sr=sr)
        # Estraiamo il valore numerico indipendentemente dal formato (array o scalare)
        bpm_final = float(np.ravel(tempo_data)[0])
        
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

        # Grafici Visuali
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_eq, ax_eq = plt.subplots(figsize=(10, 4))
            ax_eq.plot(np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1), color="#00f2ff")
            ax_eq.set_title("Spectral Balance (EQ)")
            ax_eq.set_yscale('log')
            st.pyplot(fig_eq)
        with col_g2:
            fig_w, ax_w = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y[:int(5*sr)], sr=sr, ax=ax_w, color='#ff00ff')
            ax_w.set_title("Zoom Transienti (5s)")
            st.pyplot(fig_w)

        # Chat Interattiva
        st.divider()
        st.subheader("💬 Chiedi all'Ingegnere AI")
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Kick e basso suonano bene?"):
            if user_key:
                openai.api_key = user_key
                
                # ISTRUZIONE DI SISTEMA CON I DATI REALI (FORZA L'IA AD ANALIZZARE)
                sys_msg = f"""
                Agisci come un Senior Mixing Engineer di Ableton Live. 
                I dati della traccia caricata dall'utente sono questi (USALI PER RISPONDERE):
                - Loudness attuale: {lufs:.1f} LUFS.
                - Crest Factor attuale: {crest:.1f} dB.
                - BPM: {int(bpm_final)}, Scala: {key_detected}.

                REGOLE CRITICHE:
                1. Se l'utente chiede 'come suonano kick e basso', guarda il CREST FACTOR. Se è sopra 11dB, di' che il kick è MOLLACCIO e suggerisci Saturator (Sinoid Fold) o Glue Compressor (Soft Clip).
                2. Se chiede del mix in generale, guarda i LUFS. Se sono sopra -12dB, di' che il mix non ha ENERGIA festival.
                3. Suggerisci plugin NATIVI ABLETON con parametri precisi (Threshold, Attack, Dry/Wet).
                4. NON dire mai 'non posso sentire l'audio'. Usa i dati tecnici forniti qui sopra.
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
                        st.error(f"Errore API: {e}")
            else:
                st.warning("Inserisci la API Key!")

# Glossario
st.divider()
with st.expander("📖 Glossario Tecnico"):
    st.write("- **LUFS**: Potenza reale. -8 è lo standard Garrix. - **Crest**: Punch. 8-10 è EDM, >13 è moscio.")
