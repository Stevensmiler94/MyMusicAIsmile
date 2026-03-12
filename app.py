import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai
from audio_recorder_streamlit import audio_recorder
import json

# Configurazione Pagina
st.set_page_config(page_title="AI Studio Vault", layout="wide", page_icon="💾")

# --- INIZIALIZZAZIONE SESSIONE ---
if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": []}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- FUNZIONI DI SALVATAGGIO/CARICAMENTO ---
def export_progetto():
    data = {
        "nome": st.session_state.progetto_attivo,
        "chat": st.session_state.progetti[st.session_state.progetto_attivo]
    }
    return json.dumps(data, indent=4)

def import_progetto(uploaded_json):
    if uploaded_json:
        try:
            data = json.load(uploaded_json)
            nome = data["nome"]
            st.session_state.progetti[nome] = data["chat"]
            st.session_state.progetto_attivo = nome
            st.toast(f"Progetto '{nome}' caricato!")
        except:
            st.error("Errore nel caricamento del file JSON.")

# --- SIDEBAR: STUDIO MANAGER ---
st.sidebar.title("🏢 Studio Vault")
user_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Gestione Progetti
st.sidebar.subheader("📁 Scenari")
st.session_state.progetto_attivo = st.sidebar.selectbox("Seleziona Progetto", list(st.session_state.progetti.keys()))

if st.sidebar.button("🗑️ Reset Chat"):
    st.session_state.progetti[st.session_state.progetto_attivo] = []
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("💾 Backup su PC")
json_string = export_progetto()
st.sidebar.download_button("📥 Salva Progetto (.json)", data=json_string, file_name=f"{st.session_state.progetto_attivo}.json")

file_caricato = st.sidebar.file_uploader("📤 Carica Progetto (.json)", type=["json"])
if file_caricato:
    import_progetto(file_caricato)

st.sidebar.divider()
nuovo_p = st.sidebar.text_input("Nuovo Scenario")
if st.sidebar.button("Crea"):
    if nuovo_p and nuovo_p not in st.session_state.progetti:
        st.session_state.progetti[nuovo_p] = []
        st.session_state.progetto_attivo = nuovo_p
        st.rerun()

# --- INTERFACCIA PRINCIPALE ---
st.title(f"🎙️ Scenario: {st.session_state.progetto_attivo}")

# Registratore
audio_bytes = audio_recorder(text="Registra idea", icon_size="2x")
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

# Analisi Tecnica
st.divider()
uploaded_file = st.sidebar.file_uploader("Analizza Audio", type=["wav", "mp3"])

tech_context = ""
if uploaded_file:
    with st.spinner("⚡ Analisi tecnica..."):
        y, sr = librosa.load(uploaded_file, duration=30)
        
        # --- FIX BPM UNIVERSALE (PYTHON 3.14 SAFE) ---
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        # Estraiamo il valore scalare in modo sicuro
        if isinstance(tempo_result, (tuple, list, np.ndarray)):
            bpm_val = tempo_result[0] if isinstance(tempo_result, tuple) else tempo_result
            bpm = float(np.ravel(bpm_val)[0])
        else:
            bpm = float(tempo_result)
        
        lufs = pdn.Meter(sr).integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
        crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
        st.info(f"📊 {lufs:.1f} LUFS | {crest:.1f}dB Crest | {int(bpm)} BPM")
        tech_context = f"[ANALISI: {lufs:.1f} LUFS, {crest:.1f}dB Crest, {int(bpm)} BPM]. "

# --- CHAT ---
st.divider()
for msg in st.session_state.progetti[st.session_state.progetto_attivo]:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Chiedi all'Ingegnere AI..."):
    if user_key:
        openai.api_key = user_key
        sys_instr = f"Sei un Senior Producer Progressive House (Ableton Expert). USA DATI: {tech_context}. Sii critico, suggerisci parametri precisi e disegna Piano Roll | X | - | se serve."
        
        st.session_state.progetti[st.session_state.progetto_attivo].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": sys_instr}] + st.session_state.progetti[st.session_state.progetto_attivo]
                )
                answer = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else resp.choices[0].message.content
                st.markdown(answer)
                st.session_state.progetti[st.session_state.progetto_attivo].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Errore: {e}")
    else:
        st.warning("Inserisci la API Key!")
