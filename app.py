import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Progressive House Mentor Pro", layout="wide", page_icon="🎹")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎹 AI Emotional Progressive House Mentor")
st.write("Analisi Mix, Songwriting e **Piano Roll Coach** (Garrix & Avicii Style).")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Setup")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
uploaded_mix = st.sidebar.file_uploader("CARICA MIX / STEM (Opzionale)", type=["wav", "mp3"])
uploaded_ref = st.sidebar.file_uploader("REFERENCE (Opzionale)", type=["wav", "mp3"])

# --- LOGICA DI ANALISI AUDIO (Se presente) ---
tech_info = "Nessun file caricato (Modalità Creativa Libera)."
if uploaded_mix:
    with st.spinner("⚡ Scansione tecnica in corso..."):
        y_m, sr = librosa.load(uploaded_mix, duration=30)
        tempo_res = librosa.beat.beat_track(y=y_m, sr=sr)
        bpm_m = float(np.atleast_1d(tempo_res)[0])
        lufs_m = pdn.Meter(sr).integrated_loudness(y_m.reshape(-1, 1) if y_m.ndim == 1 else y_m.T)
        crest_m = 20 * np.log10(np.max(np.abs(y_m)) / (np.sqrt(np.mean(y_m**2)) + 1e-9))
        chroma = librosa.feature.chroma_stft(y=y_m, sr=sr)
        key_m = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
        
        tech_info = f"[Dati Mix: {lufs_m:.1f} LUFS, {crest_m:.1f}dB Crest, BPM {int(bpm_m)}, Scala {key_m}]"
        
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Loudness", f"{lufs_m:.1f} LUFS")
        col2.metric("Crest Factor", f"{crest_m:.1f} dB")
        col3.metric("BPM", f"{int(bpm_m)}")
        col4.metric("Scala", key_m)

# --- CHAT INTERATTIVA (Analisi + Composizione Libera) ---
st.divider()
st.subheader("💬 Parla con il tuo Mentor (Mix, Testi o Idee Piano Roll)")
if not uploaded_mix:
    st.info("💡 Non hai caricato file: descrivi i tuoi accordi, bpm e scala qui sotto per ricevere consigli!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Esempio: Ho questi 4 accordi (Am, F, C, G) a 128 bpm, fammi un giro di lead"):
    if user_key:
        openai.api_key = user_key
        
        system_instruction = f"""
        Sei un Senior Producer Progressive House (stile Garrix/Avicii).
        CONTESTO TECNICO: {tech_info}.
        
        MISSIONE:
        1. PIANO ROLL: Se l'utente dà accordi/scala, disegna uno schema ritmico (es: | C3 | - | X | - |) e suggerisci note per il Lead.
        2. CONFRONTO: Se c'è una reference, sii spietato sui dati tecnici del mix.
        3. SONGWRITING: Scrivi testi emozionali in inglese e suggerisci melodie.
        4. ABLETON: Dai parametri esatti per Saturator, OTT e Glue Comp.
        Sii tecnico, critico e usa un linguaggio da studio pro.
        """

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_instruction}] + st.session_state.messages
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
with st.expander("📖 Glossario Tecnico EDM"):
    st.write("**LUFS**: Volume. **Crest**: Punch. **OTT**: Brillantezza. **Piano Roll**: Griglia melodica.")
