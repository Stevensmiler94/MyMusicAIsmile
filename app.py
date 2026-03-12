import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Music Producer Assistant", layout="wide", page_icon="🎹")

# Inizializzazione della cronologia chat se non esiste
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎹 AI Music Assistant: Interattivo")
st.write("Analisi Mix e Songwriting con **Chat di approfondimento**.")

# --- SIDEBAR: IMPOSTAZIONI ---
st.sidebar.header("⚙️ Configurazione")
user_key = st.sidebar.text_input("Chiave API OpenAI (sk-...)", type="password")

mode = st.sidebar.selectbox("Scegli Modalità:", ["Mix Completo vs Reference", "Traccia Singola (Stem)"])

if mode == "Mix Completo vs Reference":
    uploaded_mix = st.sidebar.file_uploader("Il tuo Mix", type=["wav", "mp3"], key="mix")
    uploaded_ref = st.sidebar.file_uploader("Reference", type=["wav", "mp3"], key="ref")
    audio_to_analyze = uploaded_mix
else:
    uploaded_stem = st.sidebar.file_uploader("Carica la tua Traccia Singola", type=["wav", "mp3"], key="stem")
    audio_to_analyze = uploaded_stem
    uploaded_ref = None

# --- LOGICA DI ANALISI AUDIO ---
if audio_to_analyze:
    with st.spinner("🚀 Analizzando l'audio..."):
        y_mix, sr = librosa.load(audio_to_analyze, duration=30)
        
        # Rilevamento BPM e Scala
        tempo_data = librosa.beat.beat_track(y=y_mix, sr=sr)
        bpm_val = tempo_data if isinstance(tempo_data, (list, np.ndarray, tuple)) else tempo_data
        bpm_final = float(np.atleast_1d(bpm_val))
        
        chroma = librosa.feature.chroma_stft(y=y_mix, sr=sr)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_detected = notes[np.argmax(np.mean(chroma, axis=1))]

        # Loudness e Dinamica
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            return pdn.Meter(rate).integrated_loudness(data)
        lufs_m = get_lufs(y_mix, sr)
        crest_factor = 20 * np.log10(np.max(np.abs(y_mix)) / (np.sqrt(np.mean(y_mix**2)) + 1e-9))

        # --- LAYOUT TECNICO ---
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BPM", f"{int(round(bpm_final))}")
        c2.metric("Scala", f"{key_detected}")
        c3.metric("Loudness", f"{lufs_m:.1f} LUFS")
        c4.metric("Crest Factor", f"{crest_factor:.1f} dB")

        # --- GRAFICI (Compatti) ---
        st.subheader("📊 Analisi Visiva")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1), color="#00f2ff")
        ax1.set_title("Spettro EQ")
        ax1.set_yscale('log')
        librosa.display.waveshow(y_mix[:int(5*sr)], sr=sr, ax=ax2, color='#ff00ff')
        ax2.set_title("Zoom Transienti (5s)")
        st.pyplot(fig)

        # --- CHAT INTERATTIVA ---
        st.divider()
        st.subheader("💬 Chiedi approfondimenti all'Ingegnere AI")

        # Mostra messaggi precedenti
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input utente
        if prompt := st.chat_input("Esempio: Come imposto il compressore per migliorare il Crest Factor?"):
            if user_key:
                openai.api_key = user_key
                # Aggiungi contesto tecnico al primo messaggio se la chat è vuota
                context = ""
                if len(st.session_state.messages) == 0:
                    context = f"DATI TECNICI: {lufs_m:.1f} LUFS, {crest_factor:.1f} dB Crest Factor, BPM {int(bpm_final)}, Scala {key_detected}. Genere: Progressive House (Garrix style). "
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        full_prompt = context + prompt
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                        )
                        answer = response['choices'][0]['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key nella sidebar!")

# --- INFO ---
if not (uploaded_mix if mode == "Mix Completo vs Reference" else uploaded_stem):
    st.info("👋 Carica i file per avviare l'analisi e iniziare la chat.")
