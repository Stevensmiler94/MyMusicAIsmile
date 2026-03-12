import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pdn
import openai

st.set_page_config(page_title="AI Mix Assistant", layout="wide")

st.title("🎧 Assistant Mix Progressive House (Garrix Style)")
st.write("Analizza il tuo brano e ricevi consigli tecnici dall'IA.")

# --- SIDEBAR ---
st.sidebar.header("Impostazioni")
user_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")
uploaded_mix = st.sidebar.file_uploader("Il tuo Mix (.wav)", type=["wav", "mp3"])
uploaded_ref = st.sidebar.file_uploader("Traccia Reference (.wav)", type=["wav", "mp3"])

if uploaded_mix and uploaded_ref:
    with st.spinner("Analizzando l'audio..."):
        # Caricamento
        y_mix, sr = librosa.load(uploaded_mix, duration=30)
        y_ref, _ = librosa.load(uploaded_ref, duration=30)

        # 1. LOUDNESS (LUFS)
        def get_lufs(y, rate):
            data = y.reshape(-1, 1) if y.ndim == 1 else y.T
            meter = pdn.Meter(rate)
            return meter.integrated_loudness(data)

        lufs_m = get_lufs(y_mix, sr)
        lufs_r = get_lufs(y_ref, sr)

        # 2. EQ & STEREO (Calcoli semplificati)
        spec_m = np.mean(librosa.feature.melspectrogram(y=y_mix, sr=sr), axis=1)
        spec_r = np.mean(librosa.feature.melspectrogram(y=y_ref, sr=sr), axis=1)

        # Visualizzazione Risultati
        col1, col2, col3 = st.columns(3)
        col1.metric("Tua Loudness", f"{lufs_m:.1f} LUFS")
        col2.metric("Ref Loudness", f"{lufs_r:.1f} LUFS")
        col3.metric("Gap", f"{lufs_m - lufs_r:.1f}")

        # GRAFICO EQ
        st.subheader("📊 Confronto Frequenze (EQ)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(spec_m, label="Tuo Mix", color="cyan")
        ax.plot(spec_r, label="Garrix Ref", color="orange", alpha=0.5)
        ax.set_yscale('log')
        st.pyplot(fig)

        # 3. FEEDBACK IA
        if st.button("Genera Critica Tecnica dell'Ingegnere AI"):
            if user_key:
                openai.api_key = user_key
                prompt = f"Analizza: Mix a {lufs_m} LUFS vs Ref a {lufs_r}. EQ Mix ha valore medio {np.mean(spec_m)}. Dammi 3 consigli stile Martin Garrix."
                resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
                st.success(resp.choices[0].message.content)
            else:
                st.error("Inserisci la API Key nella sidebar!")
