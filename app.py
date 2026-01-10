import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Petrophysical Calculator")

# --- Sidebar with Author Info --- 
with st.sidebar:
	st.image("GM_LOGO.JPG", width=120) # optional logo/photo
	st.markdown("### 👨‍💻 About the Author")
	st.write("Developed by **Gerardo Monsegui**")
	st.write("📍 Merida, Venezuela")
	st.write("📚 Focus: Archie equation, porosity, salinity–resistivity")
	st.write("🛠️ Tools: Streamlit, NumPy, Pandas, Matplotlib")
	st.write("🔗 [LinkedIn](https://www.linkedin.com/in/gmonsegui)")
	st.write("📧 gmonsegui@gmail.com")

# Define all tabs together 
tab1, tab2, tab3, tab4 = st.tabs([ "Archie Equation", "Porosity", "Salinity–Resistivity", "Credits" ])

# --- Archie Equation Calculator ---
with tab1:
    st.header("Archie Equation Calculator")
    Rt = st.number_input("True Resistivity Rt (ohm·m)", min_value=0.0)
    Rw = st.number_input("Water Resistivity Rw (ohm·m)", min_value=0.0)
    phi = st.slider("Porosity ? (fraction)", 0.0, 1.0, 0.25)
    a = st.slider("Constant a", 0.5, 1.5, 1.0)
    m = st.slider("Cementation exponent m", 1.5, 3.0, 2.0)
    n = st.slider("Saturation exponent n", 1.5, 3.0, 2.0)

    if Rt > 0 and Rw > 0 and phi > 0:
        F = a / (phi ** m)
        Sw = ((a * Rw) / (phi ** m * Rt)) ** (1 / n)
        Sw = np.clip(Sw, 0, 1)
        Sh = 1 - Sw
        st.write(f"Formation Factor F = {F:.2f}")
        st.write(f"Water Saturation Sw = {Sw:.2f}")
        st.write(f"Hydrocarbon Saturation Sh = {Sh:.2f}")
    else:
        st.error("Inputs must be > 0 for Rt, Rw, and ?.")

# --- Porosity Calculator ---
with tab2:
    st.header("Porosity Calculator")

    st.subheader("Density Log Method")
    rho_ma = st.number_input("Matrix density ?ma (g/cm³)", value=2.65)
    rho_b = st.number_input("Bulk density ?b (g/cm³)", value=2.35)
    rho_f = st.number_input("Fluid density ?f (g/cm³)", value=1.0)

    if rho_ma != rho_f:
        phi_density = (rho_ma - rho_b) / (rho_ma - rho_f)
        st.write(f"Porosity (Density) = {phi_density:.2f}")
    else:
        st.error("Matrix and fluid densities must differ.")

    st.subheader("Sonic Log Method")
    dt_b = st.number_input("?t bulk (µs/ft)", value=80.0)
    dt_ma = st.number_input("?t matrix (µs/ft)", value=55.5)
    dt_f = st.number_input("?t fluid (µs/ft)", value=189.0)

    if dt_f != dt_ma:
        phi_sonic = (dt_b - dt_ma) / (dt_f - dt_ma)
        st.write(f"Porosity (Sonic) = {phi_sonic:.2f}")
    else:
        st.error("?t fluid and ?t matrix must differ.")

    st.subheader("Neutron–Density Cross-over")
    phi_N = st.slider("Neutron porosity ?N", 0.0, 1.0, 0.25)
    phi_D = st.slider("Density porosity ?D", 0.0, 1.0, 0.25)
    if phi_N + 0.05 < phi_D:
        st.warning("Gas effect detected (?N << ?D).")

# --- Salinity–Resistivity Calculator ---
with tab3:
    st.header("Salinity–Resistivity Calculator")

    st.subheader("Temperature Correction (Arps)")
    Rw_T1 = st.number_input("Rw at lab temp (ohm·m)", value=0.1)
    T1 = st.number_input("Lab temperature T1 (°C)", value=25.0)
    T2 = st.number_input("Formation temperature T2 (°C)", value=75.0)

    if T1 + 6.77 != 0:
        Rw_T2 = Rw_T1 * (T2 + 6.77) / (T1 + 6.77)
        st.write(f"Rw at {T2}°C = {Rw_T2:.3f} ohm·m")

    st.subheader("Chlorinity to Salinity")
    Cl = st.number_input("Chlorinity (g/L)", value=10.0)
    salinity = 1.80655 * Cl
    st.write(f"Salinity ? {salinity:.2f} g/L")

    st.subheader("Empirical Salinity–Rw Fit")
    uploaded_file = st.file_uploader("Upload CSV with columns: Salinity,Rw", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Salinity" in df.columns and "Rw" in df.columns:
            logS = np.log(df["Salinity"])
            logRw = np.log(df["Rw"])
            coeffs = np.polyfit(logS, logRw, 1)
            b = -coeffs[0]
            k = np.exp(coeffs[1])
            st.write(f"Fit: Rw ? {k:.3f} * Salinity^(-{b:.3f})")

            fig, ax = plt.subplots()
            ax.loglog(df["Salinity"], df["Rw"], 'o', label="Data")
            sal_range = np.linspace(df["Salinity"].min(), df["Salinity"].max(), 100)
            ax.loglog(sal_range, k * sal_range ** (-b), '-', label="Fit")
            ax.set_xlabel("Salinity (g/L)")
            ax.set_ylabel("Rw (ohm·m)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("CSV must have columns: Salinity, Rw")
# Add a Credits tab

with tab4:
    st.header("👨‍💻 Credits")
    st.markdown("""
    This **Petrophysical Calculator App** was created by **Gerardo Monsegui**.

    - 📚 Focus areas: Archie equation, porosity computation, salinity–resistivity relationships  
    - 🛠️ Built with: Streamlit, NumPy, Pandas, Matplotlib  
    - 🌍 Location: Merida, Venezuela  
    - 🔗 Connect: [LinkedIn](https://www.linkedin.com/in/gmonsegui) | [Email](mailto:gmonsegui@gmail.com)  

    ---
    © 2026 Gerardo Monsegui — All rights reserved
    """)
