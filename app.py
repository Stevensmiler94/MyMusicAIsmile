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
st.write("L'IA analizza i dati del tuo file e ti dà consigli professionali su Ableton.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Config")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
audio_file = st.sidebar.file_uploader("Carica il tuo file audio", type=["wav", "mp3"])

# --- LOGICA DI ANALISI AUDIO ---
if audio_file:
    with st.spinner("⚡ Analisi tecnica in corso..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # FIX BPM UNIVERSALE
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        bpm_val = tempo_result[0] if isinstance(tempo_result, (tuple, list, np.ndarray)) else tempo_result
        bpm_final = float(bpm_val)
        
        # Scala e Dati Tecnici
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_detected = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        
        # Dashboard Dati
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest Factor (Punch)", f"{crest:.1f} dB")
        c3.metric("BPM", f"{int(round(bpm_final))}")

        # Chat Interattiva
        st.divider()
        st.subheader("💬 Parla con il tuo Mixing Engineer AI")
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Kick e basso suonano bene?"):
            if user_key:
                openai.api_key = user_key
                
                # ISTRUZIONE DI SISTEMA: Obbliga l'IA a usare i dati ed essere critica
                # Inseriamo i dati DIRETTAMENTE nel prompt utente per non farli ignorare
                content_ai = f"""
                TU SEI UN MIXING ENGINEER DI ABLETON LIVE CATTIVO E TECNICO.
                NON DIRE CHE NON PUOI ASCOLTARE. USA QUESTI DATI CHE HO ESTRATTO PER TE:
                - Loudness rilevata: {lufs:.1f} LUFS
                - Crest Factor rilevato: {crest:.1f} dB
                - BPM rilevati: {int(bpm_final)}

                Tuo compito:
                1. Se il Crest Factor è sopra 11dB, il Kick è debole (manca di punch). Suggerisci Saturator o Glue Comp.
                2. Se i LUFS sono sopra -10dB, il mix è moscio per la Progressive House.
                3. Suggerisci plugin NATIVI ABLETON con settaggi precisi.
                4. Se l'utente chiede 'come suonano', usa i numeri sopra per dare un verdetto spietato.
                """
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        # Uniamo il sistema e la cronologia
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": content_ai}] + st.session_state.messages
                        )
                        answer = resp['choices'][0]['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key nella sidebar!")

if not audio_file:
    st.info("👋 Carica un file audio per sbloccare l'analisi tecnica e la chat.")
