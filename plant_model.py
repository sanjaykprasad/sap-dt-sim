"""
Sulphuric Acid Plant - First Principles Model
============================================
This module implements a reduced-order physics-based model of a sulphuric acid plant,
focusing on three core areas: sulphur burner, converter beds, and waste heat boiler.

Extended with diagnosis, recommendation and what-if simulation functions for
operator training and decision support.
"""

import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# --- Physical Constants ---
CP_GAS = 1.1          # Specific heat of process gas [kJ/kg-K]
DELTAH_SO2_SO3 = 990  # Heat released when SO2 oxidizes to SO3 [kJ/kg SO2]
DELTAH_S_BURN = 9300  # Heat released when sulphur burns [kJ/kg sulphur]
LATENT_STEAM = 2100   # Latent heat of vaporization for steam [kJ/kg steam]
AIR_O2 = 0.21         # Mass fraction of oxygen in air

# --- Molecular Weights ---
MW_S = 32.0           # Sulphur [kg/kmol]
MW_O2 = 32.0          # Oxygen [kg/kmol]
MW_SO2 = 64.0         # Sulphur dioxide [kg/kmol]
MW_SO3 = 80.0         # Sulphur trioxide [kg/kmol]
MW_H2SO4 = 98.0       # Sulphuric acid [kg/kmol]

# --- Economic Assumptions ---
ECONOMIC_ASSUMPTIONS = {
    'steam_value': 25,        # Value of exported steam [$/ton]
    'sulphur_cost': 150,      # Cost of sulphur feed [$/ton]
    'acid_price': 200,        # Selling price of sulphuric acid [$/ton H2SO4]
    'catalyst_replace_cost': 50000,  # Cost to replace catalyst in one bed [$]
    'maintenance_factor': 0.02 # Annual maintenance as fraction of revenue
}


# ============================================================================
# BURNER MODEL
# ============================================================================

def burner(sulphur_tpd: float, air_ratio: float) -> Tuple[float, float, float, float]:
    """
    Simulate the sulphur burner where elemental sulphur is combusted to SO2.
    
    Args:
        sulphur_tpd: Sulphur feed rate [tons per day]
        air_ratio: Air to sulphur mass flow ratio [kg air / kg sulphur]
    
    Returns:
        Tuple containing:
            gas_flow: Total process gas flow [kg/hr]
            so2: Mass fraction of SO2 in exit gas [-]
            o2: Mass fraction of O2 in exit gas [-]
            T_out: Burner exit temperature [°C]
    """
    if sulphur_tpd <= 0 or air_ratio <= 0:
        raise ValueError("Sulphur feed and air ratio must be positive")
    
    m_s = sulphur_tpd * 1000 / 24  # kg/hr
    m_air = m_s * air_ratio
    gas_flow = m_air + m_s

    o2_in = m_air * AIR_O2
    o2_req = m_s * 1.0  # Simplified: 1 kg O2 per kg S
    excess_o2 = max(o2_in - o2_req, 0)

    so2 = m_s 
    o2 = excess_o2 

    Q = m_s * DELTAH_S_BURN
    dT = Q / (gas_flow * CP_GAS)
    T_out = 400 + dT   # assume air preheated ~400°C

    return gas_flow, so2, o2, T_out


# ============================================================================
# CONVERTER MODEL
# ============================================================================

def equilibrium_conversion(T: float) -> float:
    """Equilibrium conversion of SO2 to SO3 at temperature T (°C)."""
    T_k = T + 273.15
    Kp = np.exp(11.305 - 11300 / T_k)
    return 1 / (1 + 1/Kp**0.5)


def catalyst_aging(months_online: float, 
                   initial_activity: float = 1.0, 
                   decay_rate: float = 0.015) -> float:
    """Exponential catalyst deactivation model."""
    if months_online < 0:
        months_online = 0
    return initial_activity * np.exp(-decay_rate * months_online)


def converter(gas_flow: float, 
              so2_in: float, 
              o2_in: float, 
              T_in: float, 
              activity: float, 
              bed_count: int = 4) -> Dict[str, Union[float, List[float]]]:
    """
    Multi-bed catalytic converter with interpass cooling.
    
    Returns a dictionary containing bed temperatures, conversions, and outlet conditions.
    """
    so2 = so2_in
    o2 = o2_in
    current_T = T_in

    results = {
        'bed_temps_in': [],
        'bed_temps_out': [],
        'bed_conversions': [],      # % per bed
        'cumulative_conversion': [], # % cumulative
        'so2_out': so2_in,
        'o2_out': o2_in,
        'overall_conversion': 0.0
    }

    cum_conversion = 0.0

    for bed in range(bed_count):
        results['bed_temps_in'].append(current_T)

        # --- iterative bed reaction (adiabatic bed behaviour)
    T_guess = current_T

    for _ in range(6):   # small convergence loop
        Xeq = equilibrium_conversion(T_guess)
        approach = 0.75 * activity
        remaining_possible = max(Xeq - (1 - so2/so2_in), 0)

        X_bed = approach * remaining_possible

        reacted = so2 * X_bed
        Q = reacted * DELTAH_SO2_SO3
        dT = Q / (gas_flow * CP_GAS)

        T_guess = current_T + dT

        T_out = T_guess

        results['bed_temps_out'].append(T_out)

        so2 = so2 * (1 - X_bed)

        cum_conversion = 1 - so2/so2_in
        results['bed_conversions'].append(X_bed * 100)
        results['cumulative_conversion'].append(cum_conversion * 100)

        if bed < bed_count - 1:
            current_T = T_out - 120

    results['so2_out'] = so2
    results['o2_out'] = o2
    results['final_temp'] = T_out
    results['overall_conversion'] = cum_conversion

    return results


# ============================================================================
# SIMPLIFIED TEMPERATURE PROFILE GENERATOR (for fast what-if)
# ============================================================================

def reactor_temperature_profile(sulphur: float, air_ratio: float, inlet_temp: float, 
                                catalyst_activity: float) -> List[float]:
    """
    Simplified physics-based temperature profile across the four catalyst beds.
    Used for rapid what-if predictions without running full converter model.
    """
    heat_factor = sulphur / 100.0
    oxygen_factor = (air_ratio - 1.0) * 80.0
    activity_shift = (1 - catalyst_activity) * 120.0

    bed1 = inlet_temp + 90 * catalyst_activity * heat_factor - oxygen_factor
    bed2 = bed1 + 70 * heat_factor - activity_shift * 0.3
    bed3 = bed2 + 50 * heat_factor - activity_shift * 0.6
    bed4 = bed3 + 30 * heat_factor - activity_shift

    return [bed1, bed2, bed3, bed4]


# ============================================================================
# CONVERSION CALCULATOR (for what-if)
# ============================================================================

def calculate_conversion(sulphur: float, air_ratio: float, inlet_temp: float, 
                         catalyst_activity: float) -> float:
    """
    Quick conversion estimator for what-if simulations.
    """
    gas_flow, so2, o2, _ = burner(sulphur, air_ratio)
    conv_results = converter(gas_flow, so2, o2, inlet_temp, catalyst_activity)
    return conv_results['overall_conversion'] * 100.0


# ============================================================================
# WASTE HEAT BOILER
# ============================================================================

def steam_generation(gas_flow: float, T_hot: float, T_stack: float = 180) -> float:
    """Calculate steam production from waste heat [kg/hr]."""
    if T_hot <= T_stack:
        return 0
    Q = gas_flow * CP_GAS * (T_hot - T_stack)
    steam = Q / LATENT_STEAM
    return steam


# ============================================================================
# ECONOMIC METRICS
# ============================================================================

def economic_metrics(steam_tph: float, conversion: float, sulphur_tpd: float,
                     so2_slip_ppm: float, assumptions: Optional[Dict] = None) -> Dict[str, float]:
    """Calculate daily profit and related economic indicators."""
    if assumptions is None:
        assumptions = ECONOMIC_ASSUMPTIONS

    acid_produced = sulphur_tpd * (MW_H2SO4 / MW_S) * conversion
    acid_revenue = acid_produced * assumptions['acid_price']
    steam_revenue = steam_tph * 24 * assumptions['steam_value']
    total_revenue = steam_revenue + acid_revenue

    sulphur_cost = sulphur_tpd * assumptions['sulphur_cost']
    maintenance_cost = total_revenue * assumptions['maintenance_factor']
    total_cost = sulphur_cost + maintenance_cost

    so2_emissions_kg_day = (so2_slip_ppm / 1e6) * (sulphur_tpd * 1000)

    return {
        'daily_revenue_usd': total_revenue,
        'daily_cost_usd': total_cost,
        'daily_profit_usd': total_revenue - total_cost,
        'profit_margin_percent': ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0,
        'acid_production_tpd': acid_produced,
        'steam_production_tpd': steam_tph * 24,
        'so2_emissions_kg_day': so2_emissions_kg_day
    }


# ============================================================================
# DIAGNOSIS ENGINE
# ============================================================================

def diagnose_process(sulphur: float, air_ratio: float, inlet_temp: float,
                     catalyst_activity: float, bed_temps: List[float], 
                     conversion: float) -> Tuple[List[str], str]:
    """
    Rule-based diagnosis of plant condition based on observed behaviour.
    Returns a list of diagnostic statements and a confidence level.
    """
    diagnosis = []
    confidence = "Medium"

    # Pattern 1: Late bed hotter than early beds → catalyst ageing
    # Correct: reaction shifting downstream
    delta1 = bed_temps[0] - inlet_temp
    delta4 = bed_temps[3] - (bed_temps[2] - 120)

    if delta4 > delta1:
        diagnosis.append("Catalyst deactivation shifting reaction downstream")
        confidence = "High"

    # Pattern 2: High inlet temp + low conversion → equilibrium limitation
    if inlet_temp > 450 and conversion < 96:
        diagnosis.append("Equilibrium limited conversion due to high temperature")
        confidence = "High"

    # Pattern 3: Low air ratio → oxygen limitation
    if air_ratio < 1.05 and conversion < 95:
        diagnosis.append("Oxygen limitation — insufficient excess air")
        confidence = "High"

    # Pattern 4: High sulphur but conversion good → throughput change
    if sulphur > 120 and conversion > 97:
        diagnosis.append("Higher throughput — plant operating normally")
        confidence = "High"

    if not diagnosis:
        diagnosis.append("No clear abnormality detected")

    return diagnosis, confidence


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def recommend_actions(diagnosis: List[str], sulphur: float, air_ratio: float,
                      inlet_temp: float, catalyst_activity: float,
                      conversion: float) -> List[Tuple[str, str]]:
    """
    Suggest operator actions based on diagnosis.
    Returns list of (action description, impact level).
    """
    actions = []
    text = " ".join(diagnosis)

    if "Equilibrium" in text:
        actions.append(("Reduce converter inlet temperature", "High impact"))
        actions.append(("Do NOT increase air ratio further", "Avoid ineffective action"))

    if "Oxygen limitation" in text:
        actions.append(("Increase air ratio by ~0.05", "High impact"))
        actions.append(("Check blower performance", "Possible root cause"))

    if "Catalyst deactivation" in text:
        actions.append(("Increase operating temperature slightly (short term)", "Temporary mitigation"))
        actions.append(("Plan catalyst screening/replacement", "Permanent solution"))

    if "throughput" in text or sulphur > 120:
        actions.append(("No corrective action required", "Plant healthy"))
        actions.append(("Verify upstream feed rate change", "Operational awareness"))

    if not actions:
        actions.append(("No action recommended", "Normal operation"))

    return actions


# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================

def simulate_action(action: str, sulphur: float, air_ratio: float,
                    inlet_temp: float, catalyst_activity: float) -> Tuple[float, float, float, float]:
    """
    Simulate the effect of following a recommended action.
    Returns new process parameters after applying the action.
    """
    new_sulphur = sulphur
    new_air = air_ratio
    new_temp = inlet_temp
    new_activity = catalyst_activity

    if "Reduce converter inlet temperature" in action:
        new_temp -= 20
    elif "Increase air ratio" in action:
        new_air += 0.05
    elif "Increase operating temperature slightly" in action:
        new_temp += 10

    return new_sulphur, new_air, new_temp, new_activity


# ============================================================================
# SCENARIO MANAGEMENT (optional, kept from earlier)
# ============================================================================

class ScenarioManager:
    """Manages saving, loading, and comparing operating scenarios."""
    def __init__(self):
        self.scenarios = {}

    def save_scenario(self, name: str, params: Dict, results: Dict) -> Dict:
        scenario = {
            'timestamp': datetime.now().isoformat(),
            'parameters': params.copy(),
            'results': results.copy() if results else {}
        }
        self.scenarios[name] = scenario
        return scenario

    def compare_scenarios(self, base_name: str, compare_name: str) -> Optional[Dict]:
        if base_name not in self.scenarios or compare_name not in self.scenarios:
            return None
        base = self.scenarios[base_name]
        comp = self.scenarios[compare_name]
        comparison = {
            'base_name': base_name,
            'compare_name': compare_name,
            'parameter_deltas': {},
            'kpi_deltas': {}
        }
        for key in base['parameters']:
            if key in comp['parameters']:
                delta = comp['parameters'][key] - base['parameters'][key]
                pct = (delta / base['parameters'][key] * 100) if base['parameters'][key] != 0 else 0
                comparison['parameter_deltas'][key] = {'absolute': delta, 'percent': pct}
        if base['results'] and comp['results']:
            kpi_keys = ['conversion_total', 'steam_tph', 'stack_ppm', 'daily_profit_usd']
            for key in kpi_keys:
                if key in base['results'] and key in comp['results']:
                    delta = comp['results'][key] - base['results'][key]
                    pct = (delta / base['results'][key] * 100) if base['results'][key] != 0 else 0
                    comparison['kpi_deltas'][key] = {'absolute': delta, 'percent': pct}
        return comparison

    def list_scenarios(self) -> List[str]:
        return list(self.scenarios.keys())

    def export_scenario(self, name: str, filename: str) -> bool:
        if name in self.scenarios:
            with open(filename, 'w') as f:
                json.dump(self.scenarios[name], f, indent=2)
            return True
        return False

    def import_scenario(self, filename: str) -> Optional[str]:
        try:
            with open(filename, 'r') as f:
                scenario = json.load(f)
            name = scenario.get('name', datetime.fromisoformat(scenario['timestamp']).strftime('%Y%m%d_%H%M%S'))
            self.scenarios[name] = scenario
            return name
        except Exception:
            return None