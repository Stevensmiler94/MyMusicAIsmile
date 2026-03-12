import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Music Producer & Songwriter", layout="wide", page_icon="🎹")

st.title("🎹 AI Music Assistant: Progressive House Creator")
st.write("Analisi Mix e Songwriting emozionale stile **Martin Garrix & Avicii**.")

# --- SIDEBAR: IMPOSTAZIONI ---
st.sidebar.header("⚙️ Configurazione")
user_key = st.sidebar.text_input("Chiave API OpenAI (sk-...)", type="password")

mode = st.sidebar.selectbox("Scegli Modalità:", 
                            ["Mix Completo vs Reference", "Traccia Singola (Lead, Bass, Kick)"])

if mode == "Mix Completo vs Reference":
    uploaded_mix = st.sidebar.file_uploader("Il tuo Mix (.wav/.mp3)", type=["wav", "mp3"], key="mix")
    uploaded_ref = st.sidebar.file_uploader("Traccia Reference (.wav/.mp3)", type=["wav", "mp3"], key="ref")
    audio_to_analyze = uploaded_mix
else:
    uploaded_stem = st.sidebar.file_uploader("Carica la tua Traccia Singola (Stem)", type=["wav", "mp3"], key="stem")
    uploaded_ref = None
    audio_to_analyze = uploaded_stem

# --- LOGICA DI ANALISI AUDIO ---
if audio_to_analyze:
    with st.spinner("🚀 Analizzando l'audio e rilevando BPM/Scala..."):
        # Caricamento Audio
        y_mix, sr = librosa.load(audio_to_analyze, duration=30)
        
        # 1. RILEVAMENTO BPM (Correzione definitiva per versioni diverse di librosa)
        tempo_data = librosa.beat.beat_track(y=y_mix, sr=sr)
        # Estraiamo il valore numerico indipendentemente dal formato di output
        if isinstance(tempo_data, tuple):
            bpm_val = tempo_data[0]
        else:
            bpm_val = tempo_data
        
        # Convertiamo in float standard di Python per evitare errori con int()
        bpm_final = float(np.atleast_1d(bpm_val)[0])
        
        # Rilevamento Scala
        chroma = librosa.feature.chroma_stft(y=y_mix, sr=sr)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(np.mean(chroma, axis=1))
        key_detected = notes[key_idx]

        # 2. LOUDNESS (LUFS)
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            meter = pdn.Meter(rate)
            return meter.integrated_loudness(data)

        lufs_m = get_lufs(y_mix, sr)
        
        # 3. CREST FACTOR (DINAMICA)
        peak = np.max(np.abs(y_mix))
        rms = np.sqrt(np.mean(y_mix**2))
        crest_factor = 20 * np.log10(peak / (rms + 1e-9))

        # --- LAYOUT TECNICO ---
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BPM Stimati", f"{int(round(bpm_final))}")
        c2.metric("Scala Rilevata", f"{key_detected}")
        c3.metric("Loudness", f"{lufs_m:.1f} LUFS")
        c4.metric("Crest Factor", f"{crest_factor:.1f} dB")

        # --- GRAFICI ---
        st.subheader("📊 Visualizzazione Tecnica")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            spec_m = np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1)
            fig_eq, ax_eq = plt.subplots(figsize=(10, 5))
            ax_eq.plot(spec_m, color="#00f2ff")
            ax_eq.set_title("Spettro di Frequenza")
            ax_eq.set_yscale('log')
            st.pyplot(fig_eq)

        with col_g2:
            fig_w, ax_w = plt.subplots(figsize=(10, 5))
            librosa.display.waveshow(y_mix[:int(5*sr)], sr=sr, ax=ax_w, color='#ff00ff')
            ax_w.set_title("Zoom Transienti (Primi 5s)")
            st.pyplot(fig_w)

        # --- SEZIONE SONGWRITING ---
        st.divider()
        st.subheader("✍️ AI Songwriter & Vocal Guide")
        guida_testo = st.text_area("Di cosa deve parlare il testo?", "Nostalgia, summer nights, feeling alive")
        
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("✨ Genera Testo e Melodia"):
            if user_key:
                openai.api_key = user_key
                prompt_song = f"Songwriter EDM. BPM {int(round(bpm_final))}, Scala {key_detected}. Tema: {guida_testo}. Scrivi Titoli, Testo inglese e Guida Melodia."
                try:
                    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_song}])
                    st.success("🎤 Proposta Creativa:")
                    st.write(resp.choices.message.content)
                except Exception as e:
                    st.error(f"Errore API: {e}")
            else:
                st.error("Inserisci la API Key!")

        if col_btn2.button("🛠️ Genera Consigli Mixaggio"):
            if user_key:
                openai.api_key = user_key
                prompt_mix = f"Analizza Mix EDM: {lufs_m:.1f} LUFS, Crest Factor {crest_factor:.1f}. Suggerisci 3 correzioni stile Garrix."
                try:
                    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_mix}])
                    st.info("🎧 Suggerimenti Mix:")
                    st.write(resp.choices.message.content)
                except Exception as e:
                    st.error(f"Errore API: {e}")

# --- GLOSSARIO ---
st.divider()
with st.expander("📖 Glossario Tecnico Rapido"):
    st.write("- **LUFS**: Volume percepito. -7/-9 per i drop EDM.\n- **Crest Factor**: Differenza picchi/media. 8-10dB è ottimo per i lead.\n- **Scala Musicale**: La chiave in cui devi cantare o suonare.")

if not (uploaded_mix if mode == "Mix Completo vs Reference" else uploaded_stem):
    st.info("👋 Carica la tua base nella sidebar per iniziare!")
