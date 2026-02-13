"""
Sulphuric Acid Plant Digital Twin - User Interface
==================================================
Streamlit application for operator training and decision support.
Includes scenario selection, diagnosis, recommendations, and what-if simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from plant_model import (
    burner, converter, steam_generation,
    economic_metrics, reactor_temperature_profile,
    diagnose_process, recommend_actions, simulate_action,
    calculate_conversion, catalyst_aging
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sulphuric Acid Plant Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIOS = {
    "Normal Operation": {
        "sulphur": 100,
        "air_ratio": 1.10,
        "inlet_temp": 420,
        "catalyst_activity": 1.0,
        "fault": "None"
    },
    "Scenario 1 — Conversion drop overnight": {
        "sulphur": 100,
        "air_ratio": 1.10,
        "inlet_temp": 470,  # too hot
        "catalyst_activity": 1.0,
        "fault": "Equilibrium limitation"
    },
    "Scenario 2 — Gradual performance loss": {
        "sulphur": 100,
        "air_ratio": 1.10,
        "inlet_temp": 420,
        "catalyst_activity": 0.65,  # degraded catalyst
        "fault": "Catalyst ageing"
    },
    "Scenario 3 — Steam spike": {
        "sulphur": 135,  # feed increased
        "air_ratio": 1.10,
        "inlet_temp": 420,
        "catalyst_activity": 1.0,
        "fault": "Throughput increase"
    }
}

# ============================================================================
# SIDEBAR - INPUT CONTROLS
# ============================================================================

with st.sidebar:
    st.title("🏭 Plant Controls")
    st.markdown("Adjust operating parameters to explore plant behaviour")

    # Scenario selection
    scenario_name = st.selectbox(
        "Choose operating scenario",
        list(SCENARIOS.keys())
    )
    scenario = SCENARIOS[scenario_name]

    st.header("⚙️ Operating Parameters")

    sulphur = st.slider(
        "Sulphur Feed (TPD)",
        min_value=80, max_value=150, value=scenario["sulphur"], step=5,
        help="Tons of sulphur fed to the burner per day"
    )

    air_ratio = st.slider(
        "Air Ratio (kg air / kg sulphur)",
        min_value=1.0, max_value=1.3, value=scenario["air_ratio"], step=0.01,
        help="Mass ratio of combustion air to sulphur feed"
    )

    inlet_temp = st.slider(
        "Converter Inlet Temperature (°C)",
        min_value=380, max_value=500, value=scenario["inlet_temp"], step=5,
        help="Temperature entering first catalyst bed"
    )

    # Catalyst activity input
    activity_method = st.radio(
        "Catalyst activity input",
        ["Direct value", "Months online"],
        horizontal=True
    )

    if activity_method == "Direct value":
        catalyst_activity = st.slider(
            "Catalyst Activity Factor",
            min_value=0.5, max_value=1.1, value=scenario["catalyst_activity"], step=0.05,
            help="1.0 = fresh catalyst, lower = degraded"
        )
    else:
        months = st.slider("Months since last replacement", 0, 48, 12, step=3)
        catalyst_activity = catalyst_aging(months)
        st.metric("Calculated Activity", f"{catalyst_activity:.2f}")

    st.header("📊 Current Settings")
    st.metric("Sulphur", f"{sulphur} TPD")
    st.metric("Air Ratio", f"{air_ratio:.2f}")
    st.metric("Inlet Temp", f"{inlet_temp}°C")
    st.metric("Activity", f"{catalyst_activity:.2f}")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

st.title("🏭 Sulphuric Acid Plant — Training Digital Twin")
st.markdown("""
    This digital twin simulates the core process units of a sulphuric acid plant.
    Your task: diagnose the current condition and decide the best operator action.
""")

# ============================================================================
# RUN THE MODEL
# ============================================================================

try:
    gas_flow, so2, o2, T_burner = burner(sulphur, air_ratio)
    conv_results = converter(gas_flow, so2, o2, inlet_temp, catalyst_activity)

    so2_out = conv_results['so2_out']
    temps = conv_results['bed_temps_out']
    bed_conversions = conv_results['bed_conversions']
    cum_conv = conv_results['cumulative_conversion']
    conversion_total = conv_results['overall_conversion'] * 100.0   # percent
    stack_ppm = so2_out * 1e6

    steam = steam_generation(gas_flow, conv_results['final_temp']) / 1000.0  # t/h

    economics = economic_metrics(
        steam_tph=steam,
        conversion=conversion_total/100.0,
        sulphur_tpd=sulphur,
        so2_slip_ppm=stack_ppm
    )

    # Also generate a simplified temperature profile for what-if (we'll keep both)
    bed_temps_simple = reactor_temperature_profile(sulphur, air_ratio, inlet_temp, catalyst_activity)

except Exception as e:
    st.error(f"Model error: {str(e)}")
    st.stop()

# ============================================================================
# KEY PERFORMANCE INDICATORS
# ============================================================================

st.header("📈 Key Performance Indicators")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Overall Conversion", f"{conversion_total:.1f}%",
              delta=f"{conversion_total-99.5:.1f}%" if conversion_total else None,
              help="Percentage of SO2 converted to SO3")

with kpi_cols[1]:
    st.metric("Steam Generation", f"{steam:.1f} t/h",
              delta=f"{steam-45:.1f} t/h" if steam else None,
              help="High-pressure steam production")

with kpi_cols[2]:
    st.metric("Stack SO2", f"{stack_ppm:.0f} ppm",
              delta=f"{-stack_ppm+200:.0f} ppm" if stack_ppm else None,
              delta_color="inverse",
              help="SO2 concentration in stack gas (lower is better)")

with kpi_cols[3]:
    st.metric("Daily Profit", f"${economics['daily_profit_usd']:,.0f}",
              delta=f"{economics['profit_margin_percent']:.1f}% margin")

# ============================================================================
# TEMPERATURE PROFILE PLOT
# ============================================================================

st.subheader("🔥 Converter Temperature Profile")
df_temps = pd.DataFrame({
    "Bed": [1, 2, 3, 4],
    "Temperature (°C)": temps
})
st.line_chart(df_temps.set_index("Bed"), height=300)

# Optional hint based on shape
if temps[2] > temps[1]:
    st.info("⚠️ Heat release occurring later in beds — possible catalyst degradation")

# ============================================================================
# DIAGNOSIS & RECOMMENDATIONS
# ============================================================================

st.divider()

if st.button("🔍 Ask Digital Twin for Diagnosis"):
    diagnosis, confidence = diagnose_process(
        sulphur, air_ratio, inlet_temp, catalyst_activity,
        temps, conversion_total
    )

    st.subheader("🧠 AI Process Interpretation")
    for d in diagnosis:
        st.write("• " + d)
    st.caption(f"Confidence: {confidence}")

    # Get and display recommendations
    actions = recommend_actions(
        diagnosis, sulphur, air_ratio, inlet_temp, catalyst_activity, conversion_total
    )

    st.subheader("📋 Recommended Operator Actions")
    for act, impact in actions:
        st.write(f"• **{act}**")
        st.caption(f"Impact: {impact}")

    # Store in session state for what-if
    st.session_state.last_diagnosis = diagnosis
    st.session_state.last_actions = actions

# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================

st.divider()
st.subheader("🔮 What-If Simulator")

if "last_actions" in st.session_state and st.session_state.last_actions:
    action_labels = [a[0] for a in st.session_state.last_actions]
    selected_action = st.selectbox("Select an action to simulate", action_labels)

    if st.button("Predict Plant Response"):
        # Simulate the action
        new_s, new_a, new_t, new_act = simulate_action(
            selected_action, sulphur, air_ratio, inlet_temp, catalyst_activity
        )

        # Predict new temperature profile and conversion
        new_beds = reactor_temperature_profile(new_s, new_a, new_t, new_act)
        new_conv = calculate_conversion(new_s, new_a, new_t, new_act)

        st.markdown("### 📈 Predicted Outcome")
        st.write(f"**Conversion:** {new_conv:.1f}% (current: {conversion_total:.1f}%)")
        st.write(f"**Inlet temperature:** {new_t:.0f}°C")
        st.write("**Temperature profile after action:**")
        df_new = pd.DataFrame({"Bed": [1,2,3,4], "Temperature": new_beds})
        st.line_chart(df_new.set_index("Bed"), height=250)

        st.info("You can now adjust the sliders manually to match this scenario.")
else:
    st.info("Run a diagnosis first to get recommended actions.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("""
    Sulphuric Acid Plant Digital Twin — First Principles Model.
    For operator training and decision support. Not for actual plant control.
""")