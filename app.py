import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Progressive House Mentor", layout="wide", page_icon="🎹")

# Inizializzazione Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎹 AI Emotional Progressive House Mentor")
st.write("Analisi spietata, Songwriting e Piano Roll Coaching (Garrix & Avicii Style).")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Setup")
user_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
uploaded_mix = st.sidebar.file_uploader("IL TUO MIX / STEM", type=["wav", "mp3"], key="mix")
uploaded_ref = st.sidebar.file_uploader("REFERENCE (Garrix/Avicii)", type=["wav", "mp3"], key="ref")

# --- LOGICA DI ANALISI AUDIO ---
if uploaded_mix:
    with st.spinner("⚡ Scansione tecnica in corso..."):
        # Caricamento Audio
        y_m, sr = librosa.load(uploaded_mix, duration=30)
        
        # Estrazione Dati Tecnici Mix
        tempo_m = float(np.atleast_1d(librosa.beat.beat_track(y=y_m, sr=sr)[0])[0])
        lufs_m = pdn.Meter(sr).integrated_loudness(y_m.reshape(-1, 1) if y_m.ndim == 1 else y_m.T)
        crest_m = 20 * np.log10(np.max(np.abs(y_m)) / (np.sqrt(np.mean(y_m**2)) + 1e-9))
        chroma = librosa.feature.chroma_stft(y=y_m, sr=sr)
        key_m = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.mean(chroma, axis=1))]

        # Analisi Reference (se presente)
        lufs_r, crest_r = None, None
        if uploaded_ref:
            y_r, _ = librosa.load(uploaded_ref, duration=30)
            lufs_r = pdn.Meter(sr).integrated_loudness(y_r.reshape(-1, 1) if y_r.ndim == 1 else y_r.T)
            crest_r = 20 * np.log10(np.max(np.abs(y_r)) / (np.sqrt(np.mean(y_r**2)) + 1e-9))

        # Dashboard Risultati
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Loudness", f"{lufs_m:.1f} LUFS", f"{lufs_m - (lufs_r if lufs_r else -8.0):.1f} diff")
        col2.metric("Crest Factor", f"{crest_m:.1f} dB", f"{crest_m - (crest_r if crest_r else 9.0):.1f} diff")
        col3.metric("BPM", f"{int(tempo_m)}")
        col4.metric("Scala", key_m)

        # Grafici
        st.subheader("📊 Visual Compare")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(np.mean(librosa.feature.melspectrogram(y=y_m, sr=sr), axis=1), color="cyan", label="Mix")
        if uploaded_ref:
            ax1.plot(np.mean(librosa.feature.melspectrogram(y=y_r, sr=sr), axis=1), color="orange", alpha=0.5, label="Ref")
        ax1.set_yscale('log')
        ax1.set_title("EQ Balance")
        librosa.display.waveshow(y_m[:int(5*sr)], sr=sr, ax=ax2, color='magenta')
        ax2.set_title("Transient Zoom (5s)")
        st.pyplot(fig)

        # --- CHAT INTERATTIVA ---
        st.divider()
        st.subheader("💬 Parla con il tuo Mentor (Ableton, Lyrics, Melodia)")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Esempio: Confronta il mio kick con la ref / Generami un testo / Fammi un piano roll"):
            if user_key:
                openai.api_key = user_key
                
                # INIEZIONE DATI TECNICI (OBBLIGA L'IA A USARLI)
                comparison_data = f"[Dati Mix: {lufs_m:.1f} LUFS, {crest_m:.1f}dB Crest. "
                if lufs_r: comparison_data += f"Dati Reference: {lufs_r:.1f} LUFS, {crest_r:.1f}dB Crest. "
                comparison_data += f"Scala: {key_m}, BPM: {int(tempo_m)}]"

                system_instruction = f"""
                Sei un Senior Mixing Engineer di Ableton e paroliere Progressive House.
                USA QUESTI DATI: {comparison_data}.
                
                MISSIONE:
                1. CONFRONTO: Se la reference ha LUFS o Crest migliori, spiega PERCHÉ (es: 'La ref è coesa perché il sidechain è perfetto, il tuo kick è smorzato').
                2. PIANO ROLL: Se chiedono di scrivere un lead, usa una rappresentazione visiva testuale (es: |---X---|---X---|) o griglie per spiegare la ritmica delle note.
                3. SONGWRITING: Scrivi testi in inglese emozionale e suggerisci la melodia su {key_m}.
                4. ABLETON: Dai parametri precisi per i plugin nativi (Saturator, Glue Comp, EQ Eight).
                Sii critico, professionale e focalizzato sul genere Progressive House.
                """

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        # Inseriamo i dati nel primo messaggio per dare contesto
                        api_msgs = [{"role": "system", "content": system_instruction}]
                        for i, m in enumerate(st.session_state.messages):
                            content = comparison_data + m["content"] if i == 0 else m["content"]
                            api_msgs.append({"role": m["role"], "content": content})

                        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=api_msgs)
                        answer = resp['choices']['message']['content']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Errore API: {e}")
            else:
                st.warning("⚠️ Inserisci la API Key!")

# --- GLOSSARIO ---
st.divider()
with st.expander("📖 Glossario Tecnico EDM"):
    st.write("**LUFS**: Volume percepito. **Crest Factor**: Punch del Kick. **OTT**: Brillantezza lead.")

if not uploaded_mix:
    st.info("👋 Carica il tuo brano e una reference per iniziare la sfida!")
