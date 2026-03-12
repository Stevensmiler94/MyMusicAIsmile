import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

st.set_page_config(page_title="AI Music Master Pro", layout="wide", page_icon="🎧")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎧 AI Ableton Mentor: Real-Time Analysis")
st.write("L'IA analizza i TUOI dati tecnici per darti consigli spietati.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Settings")
user_key = st.sidebar.text_input("OpenAI API Key", type="password")
audio_file = st.sidebar.file_uploader("Carica file audio", type=["wav", "mp3"])

if audio_file:
    with st.spinner("⚡ Estrazione dati in corso..."):
        y, sr = librosa.load(audio_file, duration=30)
        
        # FIX BPM DEFINITIVO (INDISTRUTTIBILE)
        tempo_res = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo_res)[0])
        
        # Analisi Tecnica
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
        
        # Dashboard
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS")
        c2.metric("Crest Factor", f"{crest:.1f} dB")
        c3.metric("BPM", f"{int(bpm)}")
        c4.metric("Scala", key)

        # Chat
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Esempio: Analizza kick e basso"):
            if user_key:
                openai.api_key = user_key
                
                # INIEZIONE DATI NEL MESSAGGIO UTENTE (L'IA NON PUO' IGNORARLI)
                system_data = f"[Dati Tecnici Rilevati: {lufs:.1f} LUFS, {crest:.1f}dB Crest, {int(bpm)} BPM, Scala {key}]. "
                full_user_prompt = system_data + prompt

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        # Istruzioni di comportamento "Cattivo/Tecnico"
                        instruction = "Sei un Mixing Engineer di Ableton. USA I DATI TECNICI TRA PARENTESI QUADRE per rispondere. Se il Crest > 12 o LUFS > -12, sii critico. Suggerisci plugin Ableton e parametri precisi."
                        
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": instruction}] + 
                                     [{"role": "user" if i==len(st.session_state.messages)-1 else m["role"], 
                                       "content": full_user_prompt if i==len(st.session_state.messages)-1 else m["content"]} 
                                      for i, m in enumerate(st.session_state.messages)]
                        )
                        answer = resp['choices'][0]['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key!")
