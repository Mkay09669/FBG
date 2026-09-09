import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("FBG_Fault_Detector.pkl")

# ----------------------------
# Extract peaks
# ----------------------------
def extract_peaks(df):
    peaks = []
    for col in df.columns[1:]:
        idx = df[col].idxmax()
        peaks.append({
            "Condition": col,
            "Peak Power (a.u.)": df[col].iloc[idx],
            "Peak Wavelength (nm)": df['wavelength'].iloc[idx]
        })
    return pd.DataFrame(peaks)

# ----------------------------
# Extract strain values from column names
# ----------------------------
def extract_strain_values(columns):
    strains = []
    for col in columns:
        try:
            strain = int(col.split()[-1])
            strains.append(strain)
        except:
            strains.append(0)
    return strains

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("FBG-Based Seismic Fault Detection System")

st.markdown("""
Upload the processed dataset: **strain_spectra_trunc.csv**

This dataset already includes:
- Strain isolation (Sensor1 – Sensor2)
- Temperature compensation
- Wavelength filtering (1547–1558 nm)
""")

uploaded_file = st.file_uploader("Upload strain_spectra_trunc.csv", type="csv")

# Optional scaling toggle
scale_option = st.checkbox("Apply scaling (×100 for visualization)", value=True)

if uploaded_file:
    strain = pd.read_csv(uploaded_file)

    st.success("Data loaded successfully")

    if st.button("Run Analysis"):

        # Create a copy for plotting only
        strain_plot = strain.copy()

        if scale_option:
            strain_plot.iloc[:, 1:] = strain_plot.iloc[:, 1:] * 100

        # ----------------------------
        # Plot spectra (use strain_plot)
        # ----------------------------
        st.subheader("Isolated Strain Spectra")

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in strain_plot.columns[1:]:
            ax.plot(strain_plot['wavelength'], strain_plot[col], linewidth=1.2)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflected Power (a.u.)")
        ax.set_title("Truncated Strain Spectra (1547–1558 nm)")
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

        # ----------------------------
        # Peak extraction
        # ----------------------------
        peaks = extract_peaks(strain)

        st.subheader("Extracted Peak Table")
        st.dataframe(peaks)

        # ----------------------------
        # Peak vs Strain plot
        # ----------------------------
        st.subheader("Peak Power vs Strain")

        strains = extract_strain_values(peaks["Condition"])

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(strains, peaks["Peak Power (a.u.)"], marker='o', linewidth=2)

        # Add labels (wavelength)
        for s, p, wl in zip(strains, peaks["Peak Power (a.u.)"], peaks["Peak Wavelength (nm)"]):
            ax2.text(s, p, f"{wl:.2f} nm", fontsize=8, ha='center')

        ax2.set_xlabel("Strain (µε)")
        ax2.set_ylabel("Peak Power (a.u.)")
        ax2.set_title("FBG Sensor Response: Peak Power vs Strain")
        ax2.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig2)

        # ----------------------------
        # Machine Learning
        # ----------------------------
        X = np.array([strain[col].values for col in strain.columns[1:]])

        preds = model.predict(X)
        probs = model.predict_proba(X)

        st.subheader("Fault Classification Results")

        classes = ["Normal", "Warning", "Fault"]

        for i, (p, prob) in enumerate(zip(preds, probs)):
            label = classes[p]
            conf = np.max(prob)

            if label == "Fault" and conf > 0.8:
                st.error(f"Sample {i+1}: FAULT | Confidence = {conf:.2f}")
            elif label == "Warning" and conf > 0.7:
                st.warning(f"Sample {i+1}: WARNING | Confidence = {conf:.2f}")
            else:
                st.success(f"Sample {i+1}: NORMAL | Confidence = {conf:.2f}")