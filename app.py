"""
Sulphuric Acid Plant Digital Twin - User Interface
==================================================
This Streamlit application provides an interactive interface to the
first-principles sulphuric acid plant model.

Users can adjust operating parameters and immediately see the impact on:
- Conversion efficiency
- Steam production
- Economic performance
- Environmental emissions

The interface is designed for operators and engineers to explore
"what-if" scenarios and understand process behavior.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import numpy as np

# Import the plant model
from plant_model import (
    burner, converter, steam_generation,
    economic_metrics, suggest_optimization,
    ScenarioManager, catalyst_aging
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
# INITIALIZE SESSION STATE
# ============================================================================

# Session state persists across reruns, allowing us to store data
if 'scenario_manager' not in st.session_state:
    st.session_state.scenario_manager = ScenarioManager()

if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None

if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

# ============================================================================
# SIDEBAR - INPUT CONTROLS
# ============================================================================

with st.sidebar:
    # --- Header ---
    st.title("🏭 Plant Controls")
    st.markdown("Adjust operating parameters to explore plant behavior")
    
    # --- Operating Parameters Section ---
    st.header("⚙️ Operating Parameters")
    
    # Sulphur feed rate (typical range: 80-140 TPD)
    sulphur = st.slider(
        "Sulphur Feed Rate (TPD)",
        min_value=80,
        max_value=140,
        value=110,
        step=5,
        help="Tons of sulphur fed to the burner per day"
    )
    
    # Air to sulphur ratio (typical range: 8-14)
    air_ratio = st.slider(
        "Air / Sulphur Ratio",
        min_value=8.0,
        max_value=14.0,
        value=11.0,
        step=0.1,
        help="Mass ratio of combustion air to sulphur feed"
    )
    
    # Converter inlet temperature (typical: 420°C ± 40°C)
    inlet_temp_adjust = st.slider(
        "Converter Inlet Temp Bias (°C)",
        min_value=-40,
        max_value=40,
        value=0,
        step=5,
        help="Adjust temperature entering first catalyst bed"
    )
    
    # --- Catalyst Condition Section ---
    st.header("🧪 Catalyst Condition")
    
    # Two ways to specify catalyst activity
    activity_input_method = st.radio(
        "Activity Input Method",
        ["Direct Activity", "Months Online"],
        horizontal=True
    )
    
    if activity_input_method == "Direct Activity":
        activity = st.slider(
            "Catalyst Activity Factor",
            min_value=0.6,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="1.0 = fresh catalyst, lower values indicate deactivation"
        )
        months_online = None
    else:
        months_online = st.slider(
            "Months Since Last Replacement",
            min_value=0,
            max_value=48,
            value=12,
            step=3,
            help="Time online affects catalyst activity"
        )
        # Calculate activity from time online
        activity = catalyst_aging(months_online)
        st.metric("Calculated Activity", f"{activity:.2f}")
    
    # --- Display Current Settings ---
    st.header("📊 Current Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sulphur", f"{sulphur} TPD")
        st.metric("Air Ratio", f"{air_ratio:.1f}")
    with col2:
        st.metric("Inlet Temp", f"{420 + inlet_temp_adjust}°C")
        st.metric("Activity", f"{activity:.2f}")
    
    # --- Scenario Management ---
    st.header("💾 Scenario Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Current", use_container_width=True):
            st.session_state.current_scenario = "current"
            st.success("Scenario saved!")
    
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.rerun()

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# --- Title and Description ---
st.title("🏭 Sulphuric Acid Plant Digital Twin")
st.markdown("""
    This reduced-order physics model simulates the core process units of a sulphuric acid plant.
    Adjust parameters in the sidebar to explore how operating decisions impact performance.
""")

# ============================================================================
# RUN THE MODEL
# ============================================================================

try:
    # --- Step 1: Run Burner Model ---
    with st.spinner("Running burner model..."):
        gas_flow, so2, o2, T_burner = burner(sulphur, air_ratio)
    
    # --- Step 2: Run Converter Model ---
    T_inlet = 420 + inlet_temp_adjust
    converter_results = converter(gas_flow, so2, o2, T_inlet, activity)
    
    # Extract converter results for easy access
    so2_out = converter_results['so2_out']
    o2_out = converter_results['o2_out']
    temps = converter_results['bed_temps_out']
    conv = converter_results['bed_conversions']
    cum_conv = converter_results['cumulative_conversion']
    
    # --- Step 3: Run Steam System Model ---
    steam = steam_generation(gas_flow, converter_results['final_temp'])
    
    # --- Step 4: Calculate KPIs ---
    conversion_total = converter_results['overall_conversion']
    stack_ppm = so2_out * 1e6  # Convert mass fraction to ppm
    
    # --- Step 5: Calculate Economics ---
    economics = economic_metrics(
        steam_tph=steam/1000,  # Convert kg/hr to tons/hr
        conversion=conversion_total,
        sulphur_tpd=sulphur,
        so2_slip_ppm=stack_ppm
    )
    
    # --- Step 6: Get Operational Suggestions ---
    suggestions = suggest_optimization(
        so2_slip_ppm=stack_ppm,
        temps=temps,
        conversion=conversion_total,
        steam_tph=steam/1000,
        sulphur_tpd=sulphur,
        air_ratio=air_ratio
    )

except Exception as e:
    st.error(f"Error running model: {str(e)}")
    st.stop()

# ============================================================================
# KEY PERFORMANCE INDICATORS (Top Row)
# ============================================================================

st.header("📈 Key Performance Indicators")

# Create three columns for KPIs
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric(
        label="Overall Conversion",
        value=f"{conversion_total*100:.2f}%",
        delta=f"{(conversion_total*100 - 99.5):.1f}% vs target",
        help="Percentage of SO2 converted to SO3"
    )

with kpi_cols[1]:
    st.metric(
        label="Steam Generation",
        value=f"{steam/1000:.1f} t/h",
        delta=f"{(steam/1000 - 45):.1f} t/h vs design",
        help="High-pressure steam production"
    )

with kpi_cols[2]:
    st.metric(
        label="Stack SO2",
        value=f"{stack_ppm:.0f} ppm",
        delta=f"{-stack_ppm + 200:.0f} ppm vs limit",
        delta_color="inverse",  # Lower is better for emissions
        help="SO2 concentration in stack gas (lower is better)"
    )

with kpi_cols[3]:
    st.metric(
        label="Daily Profit",
        value=f"${economics['daily_profit_usd']:,.0f}",
        delta=f"{economics['profit_margin_percent']:.1f}% margin",
        help="Estimated daily profit from acid and steam sales"
    )

# ============================================================================
# OPERATIONAL SUGGESTIONS
# ============================================================================

if suggestions:
    st.header("💡 Operational Suggestions")
    
    # Create columns for suggestions
    for i, suggestion in enumerate(suggestions):
        # Choose color based on priority
        if suggestion['priority'] == 'HIGH':
            color = "🔴"
        elif suggestion['priority'] == 'MEDIUM':
            color = "🟡"
        else:
            color = "🟢"
        
        # Display suggestion in an expander
        with st.expander(f"{color} {suggestion['priority']} Priority: {suggestion['issue']}"):
            st.markdown(f"**Suggestion:** {suggestion['suggestion']}")
            st.markdown(f"**Recommended Action:** {suggestion['action']}")

# ============================================================================
# PROCESS VISUALIZATIONS
# ============================================================================

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Temperature Profile",
    "📈 Conversion Analysis",
    "💰 Economic Performance",
    "📉 Sensitivity Analysis"
])

with tab1:
    st.header("Converter Temperature Profile")
    
    # Create temperature profile plot
    fig_temp = go.Figure()
    
    # Add temperature trace
    fig_temp.add_trace(go.Scatter(
        y=temps,
        x=[f"Bed {i+1}" for i in range(len(temps))],
        mode="lines+markers+text",
        name="Bed Outlet Temperature",
        line=dict(color="red", width=3),
        marker=dict(size=10),
        text=[f"{t:.0f}°C" for t in temps],
        textposition="top center"
    ))
    
    # Add safe operating limit
    fig_temp.add_hline(
        y=620,
        line_dash="dash",
        line_color="orange",
        annotation_text="Maximum Safe Temperature (620°C)",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig_temp.update_layout(
        title="Temperature Profile Across Catalyst Beds",
        yaxis_title="Temperature (°C)",
        xaxis_title="Catalyst Bed",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Add temperature data table
    temp_data = pd.DataFrame({
        'Bed': [f"Bed {i+1}" for i in range(len(temps))],
        'Inlet Temp (°C)': converter_results['bed_temps_in'],
        'Outlet Temp (°C)': temps,
        'Temperature Rise (°C)': [temps[i] - converter_results['bed_temps_in'][i] for i in range(len(temps))]
    })
    
    st.subheader("Temperature Details")
    st.dataframe(temp_data, use_container_width=True)

with tab2:
    # Create two columns for conversion visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Per-Bed Conversion")
        
        # Bar chart of bed conversions
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Bar(
            x=[f"Bed {i+1}" for i in range(len(conv))],
            y=conv,
            text=[f"{c:.1f}%" for c in conv],
            textposition="outside",
            marker_color="lightblue",
            name="Bed Conversion"
        ))
        
        fig_conv.update_layout(
            yaxis_title="Conversion (%)",
            xaxis_title="Catalyst Bed",
            showlegend=False,
            height=400,
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig_conv, use_container_width=True)
    
    with col2:
        st.subheader("Cumulative Conversion")
        
        # Line chart of cumulative conversion
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=[f"After Bed {i+1}" for i in range(len(cum_conv))],
            y=cum_conv,
            mode="lines+markers+text",
            line=dict(color="green", width=3),
            marker=dict(size=8),
            text=[f"{c:.1f}%" for c in cum_conv],
            textposition="top center"
        ))
        
        # Add target line
        fig_cum.add_hline(
            y=99.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Target (99.5%)"
        )
        
        fig_cum.update_layout(
            yaxis_title="Cumulative Conversion (%)",
            xaxis_title="Stage",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_cum, use_container_width=True)
    
    # Show final composition
    st.subheader("Gas Composition")
    comp_cols = st.columns(3)
    
    with comp_cols[0]:
        st.metric("Inlet SO₂", f"{so2:.3f} mass frac")
        st.metric("Outlet SO₂", f"{so2_out:.6f} mass frac")
    
    with comp_cols[1]:
        st.metric("Inlet O₂", f"{o2:.3f} mass frac")
        st.metric("Outlet O₂", f"{o2_out:.6f} mass frac")
    
    with comp_cols[2]:
        st.metric("SO₂ Reduction", f"{(1 - so2_out/so2)*100:.1f}%")

with tab3:
    st.header("Economic Analysis")
    
    # Create metrics in columns
    econ_cols = st.columns(3)
    
    with econ_cols[0]:
        st.metric(
            "Daily Revenue",
            f"${economics['daily_revenue_usd']:,.0f}",
            help="Total revenue from acid and steam sales"
        )
        
        # Revenue breakdown
        revenue_data = pd.DataFrame({
            'Source': ['Acid Sales', 'Steam Sales'],
            'Revenue': [
                economics['daily_revenue_usd'] - economics.get('steam_revenue', 0),
                economics.get('steam_revenue', steam/1000 * 24 * 25)
            ]
        })
        
        fig_revenue = go.Figure(data=[
            go.Pie(labels=revenue_data['Source'], values=revenue_data['Revenue'])
        ])
        fig_revenue.update_layout(title="Revenue Breakdown", height=300)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with econ_cols[1]:
        st.metric(
            "Daily Costs",
            f"${economics['daily_cost_usd']:,.0f}",
            help="Operating costs including sulphur and maintenance"
        )
        
        # Production metrics
        st.metric("Acid Production", f"{economics['acid_production_tpd']:.1f} TPD")
        st.metric("Steam Production", f"{economics['steam_production_tpd']:.1f} TPD")
    
    with econ_cols[2]:
        st.metric(
            "Daily Profit",
            f"${economics['daily_profit_usd']:,.0f}",
            delta=f"{economics['profit_margin_percent']:.1f}% margin"
        )
        
        # Environmental metrics
        st.metric("SO₂ Emissions", f"{economics['so2_emissions_kg_day']:.1f} kg/day")
        
        # Calculate emissions cost if applicable
        if economics['so2_emissions_kg_day'] > 100:
            st.warning("⚠️ Emissions exceed typical limits")

with tab4:
    st.header("Sensitivity Analysis")
    st.markdown("""
        This section shows how key parameters affect plant performance.
        Use the controls below to explore parameter sensitivity.
    """)
    
    # Parameter selection for sensitivity
    sensitivity_param = st.selectbox(
        "Select parameter to analyze",
        ["Air Ratio", "Inlet Temperature", "Catalyst Activity"]
    )
    
    # Generate sensitivity data based on selection
    if sensitivity_param == "Air Ratio":
        param_range = np.linspace(8, 14, 20)
        conversion_sens = []
        steam_sens = []
        
        for ar in param_range:
            # Run model with this air ratio
            g, s, o, t = burner(sulphur, ar)
            conv_res = converter(g, s, o, T_inlet, activity)
            conv_sens = conv_res['overall_conversion']
            steam_sens_val = steam_generation(g, conv_res['final_temp'])/1000
            
            conversion_sens.append(conv_sens * 100)
            steam_sens.append(steam_sens_val)
        
        # Create sensitivity plot
        fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_sens.add_trace(
            go.Scatter(x=param_range, y=conversion_sens, name="Conversion %", line=dict(color="blue")),
            secondary_y=False
        )
        
        fig_sens.add_trace(
            go.Scatter(x=param_range, y=steam_sens, name="Steam (t/h)", line=dict(color="red")),
            secondary_y=True
        )
        
        fig_sens.update_layout(
            title="Sensitivity to Air Ratio",
            xaxis_title="Air Ratio"
        )
        
        fig_sens.update_yaxes(title_text="Conversion (%)", secondary_y=False)
        fig_sens.update_yaxes(title_text="Steam Production (t/h)", secondary_y=True)
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
        # Show optimal range
        st.info("💡 Optimal air ratio typically between 10.5-12.0 for maximum conversion")
    
    elif sensitivity_param == "Inlet Temperature":
        param_range = np.linspace(380, 460, 20)
        conversion_sens = []
        
        for temp in param_range:
            conv_res = converter(gas_flow, so2, o2, temp, activity)
            conversion_sens.append(conv_res['overall_conversion'] * 100)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=param_range, y=conversion_sens,
            mode="lines", line=dict(color="green", width=3)
        ))
        
        fig_sens.update_layout(
            title="Sensitivity to Inlet Temperature",
            xaxis_title="Inlet Temperature (°C)",
            yaxis_title="Conversion (%)",
            yaxis_range=[90, 100]
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
    
    else:  # Catalyst Activity
        param_range = np.linspace(0.6, 1.0, 20)
        conversion_sens = []
        
        for act in param_range:
            conv_res = converter(gas_flow, so2, o2, T_inlet, act)
            conversion_sens.append(conv_res['overall_conversion'] * 100)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=param_range, y=conversion_sens,
            mode="lines", line=dict(color="purple", width=3)
        ))
        
        fig_sens.update_layout(
            title="Sensitivity to Catalyst Activity",
            xaxis_title="Catalyst Activity Factor",
            yaxis_title="Conversion (%)"
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)

# ============================================================================
# SCENARIO COMPARISON (Bottom Section)
# ============================================================================

st.header("🔄 Scenario Comparison")

# Create two scenarios for comparison
col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Scenario")
    
    # Base scenario parameters
    base_sulphur = st.number_input("Base Sulphur (TPD)", value=110, key="base_s")
    base_air = st.number_input("Base Air Ratio", value=11.0, key="base_air")
    base_temp = st.number_input("Base Temp Bias", value=0, key="base_temp")
    base_activity = st.number_input("Base Activity", value=0.9, key="base_act")
    
    # Run base scenario
    if st.button("Run Base Scenario", key="run_base"):
        b_gas, b_so2, b_o2, b_temp = burner(base_sulphur, base_air)
        b_conv = converter(b_gas, b_so2, b_o2, 420 + base_temp, base_activity)
        b_steam = steam_generation(b_gas, b_conv['final_temp'])
        
        st.session_state.base_results = {
            'conversion': b_conv['overall_conversion'] * 100,
            'steam': b_steam/1000,
            'so2_ppm': b_conv['so2_out'] * 1e6
        }
    
    if 'base_results' in st.session_state:
        st.metric("Conversion", f"{st.session_state.base_results['conversion']:.1f}%")
        st.metric("Steam", f"{st.session_state.base_results['steam']:.1f} t/h")
        st.metric("SO2", f"{st.session_state.base_results['so2_ppm']:.0f} ppm")

with col2:
    st.subheader("Comparison Scenario")
    
    # Comparison scenario parameters
    comp_sulphur = st.number_input("Comp Sulphur (TPD)", value=110, key="comp_s")
    comp_air = st.number_input("Comp Air Ratio", value=12.0, key="comp_air")
    comp_temp = st.number_input("Comp Temp Bias", value=10, key="comp_temp")
    comp_activity = st.number_input("Comp Activity", value=0.85, key="comp_act")
    
    # Run comparison scenario
    if st.button("Run Comparison Scenario", key="run_comp"):
        c_gas, c_so2, c_o2, c_temp = burner(comp_sulphur, comp_air)
        c_conv = converter(c_gas, c_so2, c_o2, 420 + comp_temp, comp_activity)
        c_steam = steam_generation(c_gas, c_conv['final_temp'])
        
        st.session_state.comp_results = {
            'conversion': c_conv['overall_conversion'] * 100,
            'steam': c_steam/1000,
            'so2_ppm': c_conv['so2_out'] * 1e6
        }
    
    if 'comp_results' in st.session_state:
        st.metric("Conversion", f"{st.session_state.comp_results['conversion']:.1f}%")
        st.metric("Steam", f"{st.session_state.comp_results['steam']:.1f} t/h")
        st.metric("SO2", f"{st.session_state.comp_results['so2_ppm']:.0f} ppm")

# Show comparison if both scenarios are run
if 'base_results' in st.session_state and 'comp_results' in st.session_state:
    st.subheader("Comparison Results")
    
    comp_cols = st.columns(3)
    
    with comp_cols[0]:
        conv_delta = st.session_state.comp_results['conversion'] - st.session_state.base_results['conversion']
        st.metric(
            "Conversion Difference",
            f"{conv_delta:+.1f}%",
            delta_color="normal"
        )
    
    with comp_cols[1]:
        steam_delta = st.session_state.comp_results['steam'] - st.session_state.base_results['steam']
        st.metric(
            "Steam Difference",
            f"{steam_delta:+.1f} t/h",
            delta_color="normal"
        )
    
    with comp_cols[2]:
        so2_delta = st.session_state.comp_results['so2_ppm'] - st.session_state.base_results['so2_ppm']
        st.metric(
            "SO2 Difference",
            f"{so2_delta:+.0f} ppm",
            delta_color="inverse"  # Lower is better for emissions
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 10px;'>
        Sulphuric Acid Plant Digital Twin - First Principles Model<br>
        For demonstration and operational reasoning purposes only
    </div>
""", unsafe_allow_html=True)