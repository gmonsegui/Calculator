import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from calculator.core import (
    archie,
    porosity_density,
    porosity_sonic,
    temperature_correct_rw,
    chlorinity_to_salinity,
    empirical_salinity_rw_fit,
    CalcError,
)

# Page config
st.set_page_config(page_title="Petrophysical Calculator", layout="wide")

# --- Page background: light blue (improved selectors) ---
st.markdown(
    """
    <style>
      :root { --bg: #e6f7ff; --sidebar-bg: #e6f7ff; --header-color: #03396c; }

      /* Main app containers */
      html, body, .stApp, .block-container, [data-testid="stAppViewContainer"], [data-testid="stMainContent"] {
        background-color: var(--bg) !important;
      }

      /* Sidebar */
      [data-testid="stSidebar"] > div:first-child {
        background-color: var(--sidebar-bg) !important;
      }

      /* Keep panels/cards readable */
      .css-1d391kg, .css-1v3fvcr, .css-18e3th9 {
        background-color: transparent !important;
      }

      /* Header color tweak for contrast */
      h1, .css-1d391kg h1, .css-1v3fvcr h1 {
        color: var(--header-color) !important;
      }

      /* Ensure markdown cards remain readable */
      .stMarkdown, .stText {
        background-color: transparent !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Petrophysical Calculator")

# --- Sidebar with Author Info and usage instructions ---
with st.sidebar:
    st.image("GM_LOGO.JPG", width=120, caption="GM Logo")
    st.markdown("### 👨‍💻 About the Author")
    st.write("Developed by **Gerardo Monsegui**")
    st.write("📍 Merida, Venezuela")
    st.write("📚 Focus: Archie equation, porosity computation, salinity–resistivity")
    st.write("🛠️ Tools: Streamlit, NumPy, Pandas, Altair")
    st.write("🔗 [LinkedIn](https://www.linkedin.com/in/gmonsegui)")
    st.write("📧 gmonsegui@gmail.com")
    st.markdown("---")

    # User-provided README usage text shown in the sidebar
    with st.expander("How to use the calculator"):
        st.markdown(
            """Petrophysical Calculator was designed to make quick Water Saturation using Archie equation as well as Porosity computation using Density and/or Sonic compressional data. Lastly the Water salinity estimation.

Select the calculation by clicking on the tab, input the values and parameters or use the sliding bar to obtain the results."""
        )

    # Sample CSV and download
    sample_csv = "Salinity,Rw\n10,0.1\n20,0.06\n30,0.04\n"
    st.markdown("Sample CSV format for Salinity,Rw:")
    st.code(sample_csv, language="csv")
    st.download_button("Download sample CSV", data=sample_csv, file_name="sample_salinity_rw.csv", mime="text/csv")

tab1, tab2, tab3, tab4 = st.tabs(["Archie Equation", "Porosity", "Salinity–Resistivity", "Credits"])

# Archie
with tab1:
    st.header("Archie Equation Calculator")
    with st.form("archie_form"):
        Rt = st.number_input(
            "True Resistivity Rt (ohm·m)",
            min_value=0.0,
            max_value=20000.0,
            value=1.0,
            format="%.6f",
            help="Maximum allowed Rt = 20,000 ohm·m",
        )
        Rw = st.number_input(
            "Water Resistivity Rw (ohm·m)",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            format="%.6f",
            help="Maximum allowed Rw = 100 ohm·m",
        )
        phi = st.slider("Porosity φ (fraction)", 0.0, 1.0, 0.25)
        a = st.slider("Constant a", 0.1, 2.0, 1.0)
        m = st.slider("Cementation exponent m", 1.0, 4.0, 2.0)
        n = st.slider("Saturation exponent n", 1.0, 4.0, 2.0)
        submitted = st.form_submit_button("Compute")

    if submitted:
        if Rt > 20000.0 or Rw > 100.0:
            st.error("Rt must be ≤ 20,000 and Rw must be ≤ 100. Please adjust inputs.")
        else:
            try:
                F, Sw, Sh = archie(Rt, Rw, phi, a=a, m=m, n=n)
                st.success(f"Formation Factor F = {F:.3f}")
                st.info(f"Water Saturation Sw = {Sw:.3f} — Hydrocarbon Saturation Sh = {Sh:.3f}")
            except CalcError as e:
                st.error(f"Input error: {e}")

# Porosity
with tab2:
    st.header("Porosity Calculator")
    st.subheader("Density Log Method")
    with st.form("density_form"):
        rho_ma = st.number_input("Matrix density ρma (g/cm³)", value=2.65)
        rho_b = st.number_input("Bulk density ρb (g/cm³)", value=2.35)
        rho_f = st.number_input("Fluid density ρf (g/cm³)", value=1.0)
        submit_density = st.form_submit_button("Compute Density Porosity")
    if submit_density:
        try:
            phi_density = porosity_density(rho_ma, rho_b, rho_f)
            st.success(f"Porosity (Density) = {phi_density:.3f}")
        except CalcError as e:
            st.error(e)

    st.subheader("Sonic Log Method")
    with st.form("sonic_form"):
        dt_b = st.number_input("Δt bulk (µs/ft)", value=80.0)
        dt_ma = st.number_input("Δt matrix (µs/ft)", value=55.5)
        dt_f = st.number_input("Δt fluid (µs/ft)", value=189.0)
        submit_sonic = st.form_submit_button("Compute Sonic Porosity")
    if submit_sonic:
        try:
            phi_sonic = porosity_sonic(dt_b, dt_ma, dt_f)
            st.success(f"Porosity (Sonic) = {phi_sonic:.3f}")
        except CalcError as e:
            st.error(e)

    st.subheader("Neutron–Density Cross-over")
    phi_N = st.slider("Neutron porosity φN", 0.0, 1.0, 0.25, key="phiN")
    phi_D = st.slider("Density porosity φD", 0.0, 1.0, 0.25, key="phiD")
    if phi_N + 0.05 < phi_D:
        st.warning("Gas effect detected (φN << φD).")

# Salinity–Resistivity with Altair interactive plot
@st.cache_data(show_spinner=False)
def load_csv_and_fit(buf: bytes):
    df = pd.read_csv(pd.io.common.BytesIO(buf))
    if "Salinity" not in df.columns or "Rw" not in df.columns:
        raise CalcError("CSV must have columns: Salinity, Rw")
    df = df.dropna(subset=["Salinity", "Rw"])
    df["Salinity"] = pd.to_numeric(df["Salinity"], errors="coerce")
    df["Rw"] = pd.to_numeric(df["Rw"], errors="coerce")
    df = df.dropna(subset=["Salinity", "Rw"])
    if df.empty:
        raise CalcError("CSV contained no valid numeric rows.")
    k, b = empirical_salinity_rw_fit(df["Salinity"].to_numpy(), df["Rw"].to_numpy())
    return df, k, b

with tab3:
    st.header("Salinity–Resistivity Calculator")

    st.subheader("Temperature Correction (Arps)")
    with st.form("temp_form"):
        Rw_T1 = st.number_input("Rw at lab temp (ohm·m)", value=0.1)
        T1 = st.number_input("Lab temperature T1 (°C)", value=25.0)
        T2 = st.number_input("Formation temperature T2 (°C)", value=75.0)
        submit_temp = st.form_submit_button("Correct Temperature")
    if submit_temp:
        try:
            Rw_T2 = temperature_correct_rw(Rw_T1, T1, T2)
            st.success(f"Rw at {T2:.1f} °C = {Rw_T2:.6f} ohm·m")
        except CalcError as e:
            st.error(e)

    st.subheader("Chlorinity to Salinity")
    Cl = st.number_input("Chlorinity (g/L)", value=10.0)
    salinity = chlorinity_to_salinity(Cl)
    st.write(f"Salinity ≈ {salinity:.3f} g/L")

    st.subheader("Empirical Salinity–Rw Fit (interactive)")
    uploaded_file = st.file_uploader("Upload CSV with columns: Salinity,Rw", type="csv")
    if uploaded_file is not None:
        try:
            buf = uploaded_file.getvalue()
            df, k, b = load_csv_and_fit(buf)
            st.write(f"Fit: Rw ≈ {k:.3f} * Salinity^(-{b:.3f})")

            # Prediction curve
            sal_range = np.logspace(np.log10(df["Salinity"].min()), np.log10(df["Salinity"].max()), 200)
            pred_df = pd.DataFrame({"Salinity": sal_range, "Rw": k * sal_range ** (-b)})

            base = alt.Chart(df).encode(
                x=alt.X("Salinity:Q", scale=alt.Scale(type="log"), title="Salinity (g/L)"),
                y=alt.Y("Rw:Q", scale=alt.Scale(type="log"), title="Rw (ohm·m)"),
                tooltip=["Salinity", "Rw"],
            )

            points = base.mark_point(filled=True, size=60).encode(color=alt.value("#1f77b4"))
            line = alt.Chart(pred_df).mark_line(color="#ff7f0e").encode(
                x=alt.X("Salinity:Q", scale=alt.Scale(type="log")),
                y=alt.Y("Rw:Q", scale=alt.Scale(type="log")),
                tooltip=[alt.Tooltip("Salinity", format=".3f"), alt.Tooltip("Rw", format=".6f")],
            )

            chart = (points + line).interactive().properties(width=700, height=450)
            st.altair_chart(chart, use_container_width=True)
        except CalcError as e:
            st.error(f"CSV error: {e}")
        except Exception as e:
            st.error(f"Unexpected error reading CSV: {e}")

with tab4:
    st.header("👨‍💻 Credits")
    st.markdown(
        """
    This **Petrophysical Calculator App** was created by **Gerardo Monsegui**.

    - 📚 Focus areas: Archie equation, porosity computation, salinity–resistivity relationships  
    - 🛠️ Built with: Streamlit, NumPy, Pandas, Altair  
    - 🌍 Location: Merida, Venezuela  
    - 🔗 Connect: [LinkedIn](https://www.linkedin.com/in/gmonsegui) | [Email](mailto:gmonsegui@gmail.com)  

    ---
    © 2026 Gerardo Monsegui — All rights reserved
    """
    )