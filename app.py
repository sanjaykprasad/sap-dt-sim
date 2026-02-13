import streamlit as st
import plotly.graph_objects as go
from plant_model import burner, converter, steam_generation

st.title("Sulphuric Acid Plant – Reduced Order Physics Twin")

# -------------------------
# Controls
# -------------------------
sulphur = st.slider("Sulphur Feed (TPD)", 80, 140, 110)
air_ratio = st.slider("Air / Sulphur Ratio", 8.0, 14.0, 11.0)
activity = st.slider("Catalyst Activity", 0.6, 1.0, 0.9)
inlet_temp_adjust = st.slider("Converter Inlet Temp Bias (°C)", -40, 40, 0)

# -------------------------
# Model Execution
# -------------------------
gas_flow, so2, o2, T_burner = burner(sulphur, air_ratio)

T_inlet = 420 + inlet_temp_adjust

so2_out, o2_out, temps, conv, T_last = converter(
    gas_flow, so2, o2, T_inlet, activity
)

steam = steam_generation(gas_flow, T_last)
conversion_total = 1 - so2_out/so2
stack_ppm = so2_out * 1e6

# -------------------------
# KPIs
# -------------------------
st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Overall Conversion", f"{conversion_total*100:.2f}%")
col2.metric("Steam Generation (t/h)", f"{steam/1000:.1f}")
col3.metric("Stack SO2 (ppm)", f"{stack_ppm:.0f}")

# -------------------------
# Temperature Profile Plot
# -------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=temps,
    x=["Bed1","Bed2","Bed3","Bed4"],
    mode="lines+markers"
))
fig.update_layout(title="Converter Temperature Profile", yaxis_title="°C")
st.plotly_chart(fig)

# -------------------------
# Conversion Plot
# -------------------------
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=["Bed1","Bed2","Bed3","Bed4"],
    y=[c*100 for c in conv]
))
fig2.update_layout(title="Per-Bed Conversion (%)")
st.plotly_chart(fig2)

st.caption("Behavioural physics model – for operational reasoning demonstration")
