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

st.title("🎚️ AI Ableton Mentor: Analisi Reale")
st.write("Analisi tecnica spietata basata sui tuoi dati reali.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
audio_file = st.sidebar.file_uploader("Carica il tuo file audio", type=["wav", "mp3"])

if audio_file:
    with st.spinner("⚡ Estrazione dati tecnici in corso..."):
        # Caricamento Audio
        y, sr = librosa.load(audio_file, duration=30)
        
        # --- FIX BPM COMPATIBILE PYTHON 3.14 ---
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        # Estraiamo il valore scalare del tempo in modo sicuro
        bpm_val = np.array(tempo_result).flatten()[0]
        bpm = float(bpm_val)
        
        # Analisi Loudness e Crest Factor
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # Dashboard Dati
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest Factor (Punch)", f"{crest:.1f} dB")
        c3.metric("BPM", f"{int(round(bpm))}")

        # Visualizzazione Chat
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Chiedi all'ingegnere: 'Analizza il mio kick e basso'"):
            if user_key:
                openai.api_key = user_key
                
                # ISTRUZIONE DI SISTEMA: Obbliga l'IA a usare i dati ed essere critica
                content_ai = f"""
                TU SEI UN MIXING ENGINEER DI ABLETON LIVE CATTIVO E TECNICO.
                NON DIRE CHE NON PUOI ASCOLTARE. USA QUESTI DATI CHE HO ESTRATTO PER TE:
                - Loudness rilevata: {lufs:.1f} LUFS
                - Crest Factor rilevato: {crest:.1f} dB
                - BPM rilevati: {int(bpm)}

                Tuo compito:
                1. Critica duramente se il Crest Factor è sopra 11dB (Kick debole) o LUFS sopra -10 (Mix moscio).
                2. Suggerisci plugin NATIVI ABLETON (Saturator, Glue Compressor, EQ Eight, OTT).
                3. Fornisci valori precisi (Threshold, Attack, Dry/Wet).
                4. Chiedi all'utente: 'Tu come hai impostato il tuo [Plugin]?' per forzare il miglioramento.
                """
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": content_ai}] + st.session_state.messages
                        )
                        answer = resp['choices']['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key nella sidebar!")

# Info iniziale
if not audio_file:
    st.info("👋 Carica un file audio per sbloccare l'analisi tecnica e la chat.")
