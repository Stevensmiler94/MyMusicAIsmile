import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

# Configurazione Pagina
st.set_page_config(page_title="AI Music Producer Assistant", layout="wide", page_icon="🎧")

st.title("🎧 AI Mix Assistant: Progressive House Edition")
st.write("Analisi professionale ispirata allo stile di **Martin Garrix & Avicii**.")

# --- SIDEBAR: IMPOSTAZIONI ---
st.sidebar.header("⚙️ Impostazioni & Upload")
user_key = st.sidebar.text_input("Chiave API OpenAI (sk-...)", type="password")

mode = st.sidebar.selectbox("Cosa vuoi analizzare?", 
                            ["Mix Completo vs Reference", "Traccia Singola (Lead, Bass, Kick)"])

if mode == "Mix Completo vs Reference":
    uploaded_mix = st.sidebar.file_uploader("Il tuo Mix (.wav/.mp3)", type=["wav", "mp3"], key="mix")
    uploaded_ref = st.sidebar.file_uploader("Traccia Reference (.wav/.mp3)", type=["wav", "mp3"], key="ref")
else:
    uploaded_stem = st.sidebar.file_uploader("Carica la tua Traccia Singola (es. solo Lead)", type=["wav", "mp3"], key="stem")
    uploaded_ref = None

# --- LOGICA DI ANALISI ---
if (mode == "Mix Completo vs Reference" and uploaded_mix and uploaded_ref) or (mode == "Traccia Singola (Lead, Bass, Kick)" and uploaded_stem):
    
    with st.spinner("🚀 Analizzando l'audio in profondità..."):
        # Caricamento Audio (primi 30 secondi)
        audio_to_analyze = uploaded_mix if mode == "Mix Completo vs Reference" else uploaded_stem
        y_mix, sr = librosa.load(audio_to_analyze, duration=30)
        
        if uploaded_ref:
            y_ref, _ = librosa.load(uploaded_ref, duration=30)
        
        # 1. LOUDNESS (LUFS)
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            meter = pdn.Meter(rate)
            return meter.integrated_loudness(data)

        lufs_m = get_lufs(y_mix, sr)
        
        # 2. CREST FACTOR (DINAMICA)
        peak = np.max(np.abs(y_mix))
        rms = np.sqrt(np.mean(y_mix**2))
        crest_factor = 20 * np.log10(peak / (rms + 1e-9))

        # 3. SPETTRO EQ
        spec_m = np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1)

        # --- LAYOUT RISULTATI ---
        st.divider()
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Loudness (Volume)", f"{lufs_m:.1f} LUFS")
            st.caption("Target EDM: -7/-9 LUFS")
        
        with col_m2:
            st.metric("Crest Factor (Punch)", f"{crest_factor:.1f} dB")
            st.caption("Dinamica dei transienti")

        if uploaded_ref:
            lufs_r = get_lufs(y_ref, sr)
            with col_m3:
                st.metric("Loudness Reference", f"{lufs_r:.1f} LUFS")

        # --- GRAFICI ---
        st.subheader("📊 Analisi Spettrale ed EQ")
        fig_eq, ax_eq = plt.subplots(figsize=(10, 3))
        ax_eq.plot(spec_m, label="Tua Traccia", color="#00f2ff", linewidth=1.5)
        if uploaded_ref:
            spec_r = np.mean(librosa.feature.melspectrogram(y=y_ref, sr=sr), axis=1)
            ax_eq.plot(spec_r, label="Reference (Garrix Style)", color="#ff9100", alpha=0.5)
        ax_eq.set_yscale('log')
        ax_eq.set_facecolor('#1e1e1e')
        ax_eq.legend()
        st.pyplot(fig_eq)

        st.subheader("🌊 Visualizzazione Waveform e Zoom")
        fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Waveform intera
        librosa.display.waveshow(y_mix, sr=sr, ax=ax1, color='#00f2ff')
        ax1.set_title("Intero (30 sec)")
        
        # Zoom primi 5 secondi per il punch
        librosa.display.waveshow(y_mix[:int(5*sr)], sr=sr, ax=ax2, color='#ff00ff')
        ax2.set_title("Zoom Iniziale (Dettaglio Transienti - 5 sec)")
        plt.tight_layout()
        st.pyplot(fig_wave)

        # --- FEEDBACK IA ---
        st.divider()
        st.subheader("🤖 Il Parere del Produttore AI")
        
        if st.button("✨ Genera Analisi Professionale e Correzioni"):
            if user_key:
                openai.api_key = user_key
                tipo_traccia = "Mix finale" if mode == "Mix Completo vs Reference" else "Traccia singola (Stem)"
                
                prompt = f"""
                Sei un esperto produttore di Progressive House (stile Martin Garrix, Avicii, Alesso).
                Analizza questa {tipo_traccia}.
                DATI TECNICI:
                - Loudness: {lufs_m:.1f} LUFS
                - Crest Factor: {crest_factor:.1f} dB
                
                COMPITI:
                1. Commenta il volume e la dinamica.
                2. Indica quali frequenze tagliare o enfatizzare.
                3. Suggerisci plugin specifici (es. OTT, Limiter, EQ) e settaggi.
                4. Se è un Lead: parla di brillantezza e layering. Se è un mix: parla di sidechain e impatto del drop.
                Sii tecnico, diretto e usa un tono professionale.
                """
                
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini", 
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.success("✅ Analisi Completata:")
                    st.write(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"Errore API: {e}")
            else:
                st.error("⚠️ Inserisci la tua OpenAI API Key nella barra laterale per sbloccare l'analisi!")

# --- GLOSSARIO ---
st.divider()
with st.expander("📖 Glossario Tecnico: Impara a mixare come i Pro"):
    st.markdown("""
    *   **LUFS:** Il volume percepito. In Progressive House il drop deve essere "loud" (-7/-8 LUFS).
    *   **Crest Factor:** La differenza tra picchi e media. Se è basso, il suono è compresso e solido.
    *   **Sidechain:** L'effetto che abbassa il basso quando colpisce il kick. Vitale per il groove.
    *   **High-Pass Filter:** Taglio delle basse frequenze inutili (su lead e voci) per evitare il "fango" nel mix.
    *   **Transienti:** L'impatto iniziale di un suono. Non schiacciarli troppo o perderai energia nel drop.
    """)

else:
    st.info("👋 Benvenuto! Carica i file nella sidebar a sinistra per iniziare l'analisi.")
