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
st.set_page_config(page_title="AI Music Command Center", layout="wide", page_icon="🎛️")

# --- INIZIALIZZAZIONE SESSION STATE ---
if "progetti" not in st.session_state:
    st.session_state.progetti = {"Default": {"songwriting": [], "mixing": [], "comparison": []}}
if "progetto_attivo" not in st.session_state:
    st.session_state.progetto_attivo = "Default"

# --- SIDEBAR: STUDIO MANAGER ---
st.sidebar.title("🏢 Studio Manager")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

# Gestione Scenari
st.sidebar.subheader("📁 Scenari Progetto")
st.session_state.progetto_attivo = st.sidebar.selectbox("Seleziona Scenario", list(st.session_state.progetti.keys()))

if st.sidebar.button("➕ Nuovo Scenario"):
    nuovo_nome = f"Progetto {len(st.session_state.progetti) + 1}"
    st.session_state.progetti[nuovo_nome] = {"songwriting": [], "mixing": [], "comparison": []}
    st.session_state.progetto_attivo = nuovo_nome
    st.rerun()

# Backup su PC
st.sidebar.divider()
st.sidebar.subheader("💾 Backup Locale")
progetto_data = json.dumps({"nome": st.session_state.progetto_attivo, "dati": st.session_state.progetti[st.session_state.progetto_attivo]}, indent=4)
st.sidebar.download_button("📥 Salva Scenario su PC", progetto_data, file_name=f"{st.session_state.progetto_attivo}.json")

carica_json = st.sidebar.file_uploader("📤 Carica Scenario (.json)", type=["json"])
if carica_json:
    data_caricata = json.load(carica_json)
    st.session_state.progetti[data_caricata["nome"]] = data_caricata["dati"]
    st.session_state.progetto_attivo = data_caricata["nome"]
    st.toast("Scenario Caricato!")

# --- FUNZIONE ANALISI TECNICA ---
def get_audio_stats(file):
    y, sr = librosa.load(file, duration=30)
    # Fix BPM
    tempo_data = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.ravel(tempo_data)[0]) if isinstance(tempo_data, tuple) else float(tempo_data)
    # Loudness
    meter = pdn.Meter(sr)
    lufs = meter.integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
    # Crest Factor
    crest = 20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))
    # Scala
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]
    return y, sr, bpm, lufs, crest, key

# --- INTERFACCIA PRINCIPALE ---
st.title(f"🚀 {st.session_state.progetto_attivo}: Studio Command Center")

tab1, tab2, tab3 = st.tabs(["📝 Songwriting & Melodia", "🎚️ Mixing Desk (Spietato)", "🏆 Festival Comparison"])

# --- TAB 1: SONGWRITING & MELODIA ---
with tab1:
    st.header("Creatività Emozionale")
    st.info("Genera testi in inglese e schemi Piano Roll per i tuoi Lead.")
    
    # Audio Recorder per idee rapide
    vocal_memo = audio_recorder(text="Registra un'idea melodica", icon_size="2x", key="vocal_rec")
    if vocal_memo: st.audio(vocal_memo)

    for m in st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    p_s = st.chat_input("Scrivi: 'Ho questi 4 accordi Am-F-C-G, fammi un testo e un piano roll'", key="in_s")
    if p_s and user_key:
        openai.api_key = user_key
        sys_s = "Sei un paroliere Progressive House (Garrix style). Scrivi testi emozionali e disegna il Piano Roll con schemi tipo | X | - | X |."
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_s}] + st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"] + [{"role": "user", "content": p_s}])
        ans = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else resp.choices[0].message.content
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "user", "content": p_s})
        st.session_state.progetti[st.session_state.progetto_attivo]["songwriting"].append({"role": "assistant", "content": ans})
        st.rerun()

# --- TAB 2: MIXING DESK (SPIETATO) ---
with tab2:
    st.header("Analisi Tecnica & Ableton Tips")
    mix_file = st.file_uploader("Carica il tuo Mix/Stem", type=["wav", "mp3"], key="u_mix")
    
    if mix_file:
        y, sr, bpm, lufs, crest, key = get_audio_stats(mix_file)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loudness", f"{lufs:.1f} LUFS", "-8.0 Target")
        c2.metric("Crest Factor", f"{crest:.1f} dB", "9.0 Target")
        c3.metric("BPM", int(bpm))
        c4.metric("Scala", key)

        for m in st.session_state.progetti[st.session_state.progetto_attivo]["mixing"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        p_m = st.chat_input("Chiedi: 'Cosa ne pensi del mio kick e basso?'", key="in_mix")
        if p_m and user_key:
            openai.api_key = user_key
            # Iniezione dati spietata
            full_p = f"[DATI REALI TRACCIA: {lufs:.1f} LUFS, {crest:.1f}dB Crest, {int(bpm)} BPM]. {p_m}"
            sys_m = "Sei un Mixing Engineer CATTIVO. NON dire che non puoi sentire. Usa i dati LUFS e Crest forniti per insultare o lodare il mix. Dai parametri Ableton precisi (Saturator, Glue Comp, EQ Eight)."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_m}] + st.session_state.progetti[st.session_state.progetto_attivo]["mixing"] + [{"role": "user", "content": full_p}])
            ans = resp.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "user", "content": p_m})
            st.session_state.progetti[st.session_state.progetto_attivo]["mixing"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- TAB 3: FESTIVAL COMPARISON ---
with tab3:
    st.header("Tu vs The Festival Pro")
    col1, col2 = st.columns(2)
    m1 = col1.file_uploader("Tuo Mix", type=["wav", "mp3"], key="comp_m")
    m2 = col2.file_uploader("Reference Pro", type=["wav", "mp3"], key="comp_r")

    if m1 and m2:
        y1, _, bpm1, lufs1, crest1, _ = get_audio_stats(m1)
        y2, _, bpm2, lufs2, crest2, _ = get_audio_stats(m2)
        
        st.write(f"📊 **Differenza Loudness:** {lufs1 - lufs2:.1f} LUFS | **Differenza Punch:** {crest1 - crest2:.1f} dB")
        
        for m in st.session_state.progetti[st.session_state.progetto_attivo]["comparison"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        p_c = st.chat_input("Confronta la coesione del kick tra i due", key="in_comp")
        if p_c and user_key:
            openai.api_key = user_key
            full_pc = f"[MIO MIX: {lufs1:.1f} LUFS, {crest1:.1f}dB Crest] vs [REFERENCE: {lufs2:.1f} LUFS, {crest2:.1f}dB Crest]. {p_c}"
            sys_c = "Confronta tecnicamente le due tracce. Spiega perché la PRO suona meglio e cosa deve copiare l'utente in Ableton (Saturazione, Sidechain, OTT)."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_c}] + st.session_state.progetti[st.session_state.progetto_attivo]["comparison"] + [{"role": "user", "content": full_pc}])
            ans = resp.choices[0].message.content
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "user", "content": p_c})
            st.session_state.progetti[st.session_state.progetto_attivo]["comparison"].append({"role": "assistant", "content": ans})
            st.rerun()

# --- GLOSSARIO ---
st.divider()
with st.expander("📖 Glossario Tecnico EDM"):
    st.write("**LUFS**: Volume reale. **Crest Factor**: Punch del Kick. **OTT**: Brillantezza lead. **Piano Roll**: Griglia melodica.")
