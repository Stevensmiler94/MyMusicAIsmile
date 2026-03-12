import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Music Master Assistant", layout="wide", page_icon="🎧")

# Inizializzazione cronologia chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎧 AI Music Master: Pro Studio Edition")
st.write("Analisi avanzata: **Saturazione, Riverbero, Immagine Stereo e Vocal FX**.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Settings")
user_key = st.sidebar.text_input("Chiave API OpenAI (sk-...)", type="password")
mode = st.sidebar.selectbox("Modalità Analisi:", ["Mix Completo vs Reference", "Traccia Singola (Stem)"])

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
    with st.spinner("🚀 Eseguendo scansione multiparametrica..."):
        y_mix, sr = librosa.load(audio_to_analyze, duration=30)
        
        # 1. RILEVAMENTO BPM (FIX DEFINITIVO)
        tempo_result = librosa.beat.beat_track(y=y_mix, sr=sr)
        
        # Gestione output librosa (può essere float o tuple)
        if isinstance(tempo_result, (tuple, list, np.ndarray)):
            # Se è una collezione, prendiamo il primo elemento (il tempo)
            bpm_val = tempo_result[0]
        else:
            # Se è già un numero singolo
            bpm_val = tempo_result
            
        # Forza la conversione a float standard di Python
        bpm_final = float(np.array(bpm_val).item())

        # Rilevamento Scala
        chroma = librosa.feature.chroma_stft(y=y_mix, sr=sr)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_detected = notes[np.argmax(np.mean(chroma, axis=1))]

        # 2. LOUDNESS E DINAMICA
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            return pdn.Meter(rate).integrated_loudness(data)
        
        lufs_m = get_lufs(y_mix, sr)
        peak = np.max(np.abs(y_mix))
        rms = np.sqrt(np.mean(y_mix**2))
        crest_factor = 20 * np.log10(peak / (rms + 1e-9))

        # --- LAYOUT TECNICO ---
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BPM", f"{int(round(bpm_final))}")
        c2.metric("Scala", f"{key_detected}")
        c3.metric("Loudness", f"{lufs_m:.1f} LUFS")
        c4.metric("Crest Factor", f"{crest_factor:.1f} dB")

        # --- GRAFICI ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        spec = np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1)
        ax1.plot(spec, color="#00f2ff")
        ax1.set_title("Spectral Balance (EQ)")
        ax1.set_yscale('log')
        librosa.display.waveshow(y_mix[:int(5*sr)], sr=sr, ax=ax2, color='#ff00ff')
        ax2.set_title("Transient Punch (5s)")
        st.pyplot(fig)

        # --- CHAT INTERATTIVA ---
        st.divider()
        st.subheader("💬 Parla con il tuo Mixing Engineer AI")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Chiedi info su saturazione, riverberi o vocal chain..."):
            if user_key:
                openai.api_key = user_key
                
                system_instruction = f"""
                Agisci come un Master Producer di Progressive House (stile Martin Garrix/Avicii). 
                DATI TECNICI: {lufs_m:.1f} LUFS, {crest_factor:.1f}dB Crest Factor, BPM {int(round(bpm_final))}, Scala {key_detected}.
                Focus: SATURAZIONE (OTT, Soft Clip), REVERB (Sidechain Hall), STEREO (Imager), COMPRESSIONE (Fast Attack), VOCALS (FX Layering).
                """

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        api_messages = [{"role": "system", "content": system_instruction}]
                        for m in st.session_state.messages:
                            api_messages.append({"role": m["role"], "content": m["content"]})

                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=api_messages
                        )
                        answer = response['choices']['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key nella sidebar!")

# --- FOOTER ---
if not (uploaded_mix if mode == "Mix Completo vs Reference" else uploaded_stem):
    st.info("👋 Carica i tuoi file per iniziare la sessione di studio.")
