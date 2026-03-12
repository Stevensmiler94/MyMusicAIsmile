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

st.title("🎚️ AI Ableton Mentor: Mixing Critico & Songwriting")
st.write("Analisi spietata dei tuoi stem. **Niente giri di parole, solo risultati professionali.**")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
mode = st.sidebar.selectbox("Cosa analizziamo?", ["Mix Completo", "Singolo Stem (Lead/Bass/Kick)"])
audio_file = st.sidebar.file_uploader("Carica il tuo file audio", type=["wav", "mp3"])

# --- ANALISI AUDIO ---
if audio_file:
    with st.spinner("⚡ Scansione tecnica in corso..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # FIX DEFINITIVO BPM
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        # Se tempo_result è una tupla (BPM, frames), prendiamo il primo elemento
        if isinstance(tempo_result, tuple):
            bpm_val = tempo_result[0]
        else:
            bpm_val = tempo_result
        # Convertiamo in float standard
        bpm_final = float(np.array(bpm_val).item())
        
        # Scala Musicale
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_detected = notes[np.argmax(np.mean(chroma, axis=1))]
        
        # Loudness e Dinamica
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # DASHBOARD TECNICA
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS", delta="-8.0 Target", delta_color="inverse")
        c2.metric("Crest (Punch)", f"{crest:.1f} dB", delta="Target: 9.0", delta_color="inverse")
        c3.metric("BPM", f"{int(round(bpm_final))}")
        c4.metric("Scala", key_detected)

        # GRAFICI
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
            ax_w.set_title("Zoom Transienti (Primi 5s)")
            st.pyplot(fig_w)

        # --- CHAT INTERATTIVA & SONGWRITING ---
        st.divider()
        st.subheader("💬 Parla con il Senior Engineer (Mix & Lyrics)")
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Chiedi i settaggi di Ableton o scrivi: 'Generami un testo emozionale'"):
            if user_key:
                openai.api_key = user_key
                
                sys_msg = f"""
                Sei un Senior Mixing Engineer di Ableton Live e paroliere Progressive House (Garrix style).
                DATI TECNICI ATTUALI: {lufs:.1f} LUFS, {crest:.1f}dB Crest Factor, BPM {int(bpm_final)}, Scala {key_detected}.
                
                MISSIONE:
                1. CRITICA SPIETATA: Se i dati sono scarichi (LUFS > -10), sii molto diretto sui problemi.
                2. ABLETON NATIVE: Suggerisci settaggi precisi solo per plugin nativi di Ableton.
                3. SONGWRITING: Se richiesto, scrivi testi in inglese emozionale e suggerisci la melodia su {key_detected}.
                4. TABELLE: Fornisci tabelle chiare per i parametri (Threshold, Attack, Release, Dry/Wet).
                5. CHIEDI SEMPRE: 'Come hai impostato il tuo [Plugin]?' per forzare il miglioramento.
                """

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": sys_msg}] + st.session_state.messages
                        )
                        answer = resp.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key nella sidebar!")

# --- GLOSSARIO ---
st.divider()
with st.expander("📖 Glossario per Produttori Pro"):
    st.markdown("""
    *   **LUFS:** Il volume percepito. Garrix punta a -7 LUFS. Se sei a -17, il tuo pezzo sparisce.
    *   **Crest Factor:** Differenza picco/media. Sotto 8dB il mix è EDM solido, sopra 12dB è moscio.
    *   **OTT:** Compressione multibanda estrema nativa in Ableton. Fondamentale per i Lead.
    *   **Glue Compressor:** Ottimo sui bus per 'incollare' Kick e Basso.
    """)

if not audio_file:
    st.info("👋 Carica uno stem o il mix per iniziare. L'Ingegnere ti sta aspettando.")
