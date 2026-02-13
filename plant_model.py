"""
Sulphuric Acid Plant - First Principles Model
============================================
This module implements a reduced-order physics-based model of a sulphuric acid plant,
focusing on three core areas: sulphur burner, converter beds, and waste heat boiler.

The model uses mass and energy balances with simplified kinetics to provide
real-time predictions of plant performance.
"""

import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# --- Physical Constants ---
# These are typical values for industrial sulphuric acid plants

# Heat capacities
CP_GAS = 1.1          # Specific heat of process gas [kJ/kg-K]

# Reaction enthalpies
DELTAH_SO2_SO3 = 990  # Heat released when SO2 oxidizes to SO3 [kJ/kg SO2]
DELTAH_S_BURN = 9300  # Heat released when sulphur burns [kJ/kg sulphur]

# Steam system
LATENT_STEAM = 2100   # Latent heat of vaporization for steam [kJ/kg steam]

# Air composition
AIR_O2 = 0.21         # Mass fraction of oxygen in air

# --- Molecular Weights ---
# Used for stoichiometric calculations
MW_S = 32.0           # Sulphur [kg/kmol]
MW_O2 = 32.0          # Oxygen [kg/kmol]
MW_SO2 = 64.0         # Sulphur dioxide [kg/kmol]
MW_SO3 = 80.0         # Sulphur trioxide [kg/kmol]
MW_H2SO4 = 98.0       # Sulphuric acid [kg/kmol]

# --- Economic Assumptions ---
# These can be adjusted based on current market conditions
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
    
    The burner model calculates:
    - Total gas flow from sulphur and air inputs
    - SO2 and O2 concentrations after combustion
    - Temperature rise from exothermic combustion
    
    Args:
        sulphur_tpd: Sulphur feed rate [tons per day]
        air_ratio: Air to sulphur mass flow ratio [kg air / kg sulphur]
    
    Returns:
        Tuple containing:
            gas_flow: Total process gas flow [kg/hr]
            so2: Mass fraction of SO2 in exit gas [-]
            o2: Mass fraction of O2 in exit gas [-]
            T_out: Burner exit temperature [°C]
    
    Raises:
        ValueError: If input parameters are invalid
    
    Example:
        >>> gas, so2, o2, temp = burner(100, 11.0)
        >>> print(f"SO2 concentration: {so2:.3f}")
    """
    # --- Input Validation ---
    if sulphur_tpd <= 0:
        raise ValueError(f"Sulphur feed must be positive, got {sulphur_tpd}")
    if air_ratio <= 0:
        raise ValueError(f"Air ratio must be positive, got {air_ratio}")
    
    # --- Mass Flow Calculations ---
    # Convert sulphur from tons/day to kg/hr for consistent units
    m_s = sulphur_tpd * 1000 / 24  # Sulphur flow [kg/hr]
    m_air = m_s * air_ratio         # Air flow [kg/hr]
    
    # Total gas flow = sulphur + air (mass conservation)
    gas_flow = m_air + m_s          # Total process gas [kg/hr]
    
    # --- Oxygen Balance ---
    # Calculate oxygen available from air and required for combustion
    o2_in = m_air * AIR_O2          # Oxygen supplied by air [kg/hr]
    o2_req = m_s * 1.0              # Simplified: 1 kg O2 per kg S [kg/hr]
    
    # Excess oxygen remains after complete sulphur combustion
    excess_o2 = max(o2_in - o2_req, 0)  # Can't have negative oxygen [kg/hr]
    
    # --- Gas Composition ---
    # Mass fractions of SO2 and O2 in the process gas
    so2 = m_s / gas_flow             # SO2 mass fraction [-]
    o2 = excess_o2 / gas_flow        # O2 mass fraction [-]
    
    # --- Energy Balance ---
    # Calculate temperature rise from combustion heat release
    Q = m_s * DELTAH_S_BURN          # Total heat released [kJ/hr]
    dT = Q / (gas_flow * CP_GAS)     # Temperature rise [°C]
    
    # Final temperature (assuming air preheated to 400°C)
    T_out = 400 + dT                  # Burner exit temperature [°C]

    return gas_flow, so2, o2, T_out


# ============================================================================
# CONVERTER MODEL
# ============================================================================

def equilibrium_conversion(T: float) -> float:
    """
    Calculate the equilibrium conversion of SO2 to SO3 at a given temperature.
    
    This uses a simplified thermodynamic model based on the equilibrium constant.
    Higher temperatures favor the reverse reaction, so conversion decreases.
    
    Args:
        T: Temperature [°C]
    
    Returns:
        Equilibrium conversion fraction (0 to 1)
    
    Notes:
        The equilibrium constant Kp is temperature-dependent:
        ln(Kp) = 11.305 - 11300/(T+273.15)
    """
    # Convert to Kelvin for thermodynamic calculations
    T_k = T + 273.15
    
    # Equilibrium constant for SO2 + 1/2 O2 <-> SO3
    Kp = np.exp(11.305 - 11300 / T_k)
    
    # Simplified relationship between Kp and conversion
    # For a feed with excess oxygen, conversion ~ 1/(1 + 1/sqrt(Kp))
    return 1 / (1 + 1/Kp**0.5)


def catalyst_aging(months_online: float, 
                   initial_activity: float = 1.0, 
                   decay_rate: float = 0.015) -> float:
    """
    Model catalyst deactivation over time.
    
    Catalyst activity decays exponentially due to:
    - Thermal sintering
    - Poisoning by contaminants
    - Mechanical degradation
    
    Args:
        months_online: Time since last catalyst replacement [months]
        initial_activity: Fresh catalyst activity (1.0 = 100%)
        decay_rate: Monthly decay rate [fraction/month]
            Typical range: 0.01 to 0.03 (1-3% per month)
    
    Returns:
        Current catalyst activity factor (0 to 1)
    
    Example:
        >>> activity = catalyst_aging(12, decay_rate=0.015)
        >>> print(f"After 1 year, activity = {activity:.2f}")
    """
    # Ensure we don't have negative time
    if months_online < 0:
        months_online = 0
    
    # Exponential decay model: a(t) = a0 * exp(-k * t)
    return initial_activity * np.exp(-decay_rate * months_online)


def converter(gas_flow: float, 
              so2_in: float, 
              o2_in: float, 
              T_in: float, 
              activity: float, 
              bed_count: int = 4) -> Dict[str, Union[float, List[float]]]:
    """
    Simulate the multi-bed catalytic converter with interpass cooling.
    
    The converter oxidizes SO2 to SO3 over multiple catalyst beds with
    cooling between beds to drive the equilibrium toward completion.
    
    Args:
        gas_flow: Process gas flow rate [kg/hr]
        so2_in: Inlet SO2 mass fraction [-]
        o2_in: Inlet O2 mass fraction [-]
        T_in: Temperature entering first bed [°C]
        activity: Catalyst activity factor (0 to 1)
        bed_count: Number of catalyst beds (typically 4)
    
    Returns:
        Dictionary containing:
            - bed_temps_in: Temperature entering each bed [°C]
            - bed_temps_out: Temperature leaving each bed [°C]
            - bed_conversions: Conversion in each bed [%]
            - cumulative_conversion: Total conversion after each bed [%]
            - so2_out: Final SO2 mass fraction [-]
            - o2_out: Final O2 mass fraction [-]
            - final_temp: Temperature after last bed [°C]
            - overall_conversion: Overall conversion fraction [-]
    
    Example:
        >>> results = converter(100000, 0.11, 0.08, 420, 0.9)
        >>> print(f"Overall conversion: {results['overall_conversion']:.2%}")
    """
    
    # --- Initialize tracking variables ---
    # Current state (will be updated through each bed)
    so2 = so2_in
    o2 = o2_in
    current_T = T_in
    
    # Storage for bed-by-bed results
    results = {
        'bed_temps_in': [],      # Inlet temperature each bed [°C]
        'bed_temps_out': [],     # Outlet temperature each bed [°C]
        'bed_conversions': [],   # Conversion in each bed [%]
        'cumulative_conversion': [],  # Total conversion after each bed [%]
        'so2_out': so2_in,       # Final SO2 (will be updated)
        'o2_out': o2_in,         # Final O2 (will be updated)
        'overall_conversion': 0.0
    }
    
    cum_conversion = 0.0  # Running total conversion
    
    # --- Process each bed sequentially ---
    for bed in range(bed_count):
        # Record inlet conditions for this bed
        results['bed_temps_in'].append(current_T)
        
        # --- Step 1: Determine achievable conversion in this bed ---
        # Equilibrium-limited conversion at inlet temperature
        Xeq = equilibrium_conversion(current_T)
        
        # Apply catalyst activity (reduces approach to equilibrium)
        X_bed = min(activity * Xeq, 1.0)
        
        # --- Step 2: Check if oxygen is limiting ---
        # Calculate oxygen required for desired conversion
        # Stoichiometry: 2SO2 + O2 -> 2SO3, so 0.5 mol O2 per mol SO2
        o2_required = so2 * X_bed * 0.5 * (MW_O2 / MW_SO2)
        
        # If oxygen is insufficient, conversion is limited by O2 availability
        if o2_required > o2:
            X_bed = o2 / (so2 * 0.5 * (MW_O2 / MW_SO2))
            X_bed = min(X_bed, 1.0)
            o2_required = o2  # Use all available oxygen
        
        # Mass of SO2 reacted in this bed [kg/hr]
        reacted = so2 * X_bed
        
        # --- Step 3: Energy balance - temperature rise from reaction ---
        # Heat released = mass reacted * reaction enthalpy
        Q = reacted * gas_flow * DELTAH_SO2_SO3  # [kJ/hr]
        
        # Temperature rise = heat released / (mass flow * heat capacity)
        dT = Q / (gas_flow * CP_GAS)  # [°C]
        
        # Outlet temperature
        T_out = current_T + dT
        results['bed_temps_out'].append(T_out)
        
        # --- Step 4: Update gas composition ---
        # SO2 decreases by amount reacted
        so2 = so2 * (1 - X_bed)
        
        # O2 decreases according to stoichiometry
        o2 = max(o2 - o2_required, 0)
        
        # --- Step 5: Track conversion progress ---
        cum_conversion = 1 - so2/so2_in
        results['bed_conversions'].append(X_bed * 100)  # Convert to %
        results['cumulative_conversion'].append(cum_conversion * 100)
        
        # --- Step 6: Interpass cooling (except after last bed) ---
        if bed < bed_count - 1:
            # Simple model: cool gas by 120°C between beds
            # In reality, this happens in heat exchangers
            current_T = T_out - 120
    
    # --- Store final conditions ---
    results['so2_out'] = so2
    results['o2_out'] = o2
    results['final_temp'] = T_out
    results['overall_conversion'] = cum_conversion
    
    return results


# ============================================================================
# WASTE HEAT BOILER MODEL
# ============================================================================

def steam_generation(gas_flow: float, 
                     T_hot: float, 
                     T_stack: float = 180) -> float:
    """
    Calculate steam production from waste heat recovery.
    
    The waste heat boiler recovers heat from hot process gas to generate
    high-pressure steam for power generation or process use.
    
    Args:
        gas_flow: Process gas flow rate [kg/hr]
        T_hot: Hot gas inlet temperature [°C]
        T_stack: Stack (outlet) gas temperature [°C]
            Typical range: 160-200°C to avoid acid condensation
    
    Returns:
        Steam generation rate [kg/hr]
    
    Notes:
        Steam production is limited by:
        - Available heat (gas flow * Cp * temperature drop)
        - Minimum stack temperature (to avoid acid dew point)
    """
    # Validate inputs
    if T_hot <= T_stack:
        # No heat recovery possible
        return 0
    
    # --- Heat Available for Recovery ---
    # Q = m * Cp * ΔT
    Q = gas_flow * CP_GAS * (T_hot - T_stack)  # [kJ/hr]
    
    # --- Steam Generation ---
    # Steam mass = heat available / latent heat of vaporization
    steam = Q / LATENT_STEAM  # [kg/hr]
    
    return steam


# ============================================================================
# ECONOMIC CALCULATIONS
# ============================================================================

def economic_metrics(steam_tph: float,
                    conversion: float,
                    sulphur_tpd: float,
                    so2_slip_ppm: float,
                    assumptions: Optional[Dict] = None) -> Dict[str, float]:
    """
    Calculate economic performance indicators.
    
    Converts process metrics to business value, helping operators
    understand the financial impact of operating decisions.
    
    Args:
        steam_tph: Steam production [tons/hour]
        conversion: Overall SO2 conversion [fraction]
        sulphur_tpd: Sulphur feed rate [tons/day]
        so2_slip_ppm: SO2 concentration in stack gas [ppm]
        assumptions: Economic assumptions dictionary (uses defaults if None)
    
    Returns:
        Dictionary of economic metrics:
            - daily_revenue_usd: Total daily revenue
            - daily_cost_usd: Total daily costs
            - daily_profit_usd: Net daily profit
            - profit_margin_percent: Profit margin [%]
            - acid_production_tpd: Sulphuric acid produced [tons/day]
            - steam_production_tpd: Steam produced [tons/day]
            - so2_emissions_kg_day: Estimated SO2 emissions [kg/day]
    
    Example:
        >>> econ = economic_metrics(45, 0.995, 110, 150)
        >>> print(f"Daily profit: ${econ['daily_profit_usd']:,.0f}")
    """
    
    # Use default assumptions if none provided
    if assumptions is None:
        assumptions = ECONOMIC_ASSUMPTIONS
    
    # --- Product Calculations ---
    # Acid production: S + O2 + H2O -> H2SO4
    # 1 ton S produces (98/32) = 3.0625 tons H2SO4
    acid_produced = sulphur_tpd * (MW_H2SO4 / MW_S) * conversion
    
    # --- Revenue Streams ---
    # Revenue from acid sales
    acid_revenue = acid_produced * assumptions['acid_price']
    
    # Revenue from steam export (if applicable)
    steam_revenue = steam_tph * 24 * assumptions['steam_value']
    
    total_revenue = steam_revenue + acid_revenue
    
    # --- Operating Costs ---
    # Raw material cost (sulphur)
    sulphur_cost = sulphur_tpd * assumptions['sulphur_cost']
    
    # Maintenance (simplified as fraction of revenue)
    maintenance_cost = total_revenue * assumptions['maintenance_factor']
    
    total_cost = sulphur_cost + maintenance_cost
    
    # --- Environmental Impact ---
    # Rough estimate of SO2 emissions
    # This is simplified - actual emissions depend on gas flow and concentration
    so2_emissions_kg_day = (so2_slip_ppm / 1e6) * (sulphur_tpd * 1000)
    
    # --- Profitability Metrics ---
    metrics = {
        'daily_revenue_usd': total_revenue,
        'daily_cost_usd': total_cost,
        'daily_profit_usd': total_revenue - total_cost,
        'profit_margin_percent': ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0,
        'acid_production_tpd': acid_produced,
        'steam_production_tpd': steam_tph * 24,
        'so2_emissions_kg_day': so2_emissions_kg_day
    }
    
    return metrics


# ============================================================================
# OPERATIONAL SUGGESTIONS
# ============================================================================

def suggest_optimization(so2_slip_ppm: float,
                        temps: List[float],
                        conversion: float,
                        steam_tph: float,
                        sulphur_tpd: float,
                        air_ratio: float) -> List[Dict[str, str]]:
    """
    Generate rule-based operating suggestions.
    
    This function acts as a simple "advisory system" that alerts operators
    to potential issues and suggests corrective actions based on heuristics.
    
    Args:
        so2_slip_ppm: Stack SO2 concentration [ppm]
        temps: List of bed outlet temperatures [°C]
        conversion: Overall conversion [fraction]
        steam_tph: Steam production [tons/hour]
        sulphur_tpd: Sulphur feed rate [tons/day]
        air_ratio: Current air to sulphur ratio
    
    Returns:
        List of suggestion dictionaries, each containing:
            - priority: "HIGH", "MEDIUM", or "LOW"
            - issue: Description of the issue
            - suggestion: Recommended action
            - action: Specific operational change
    
    Example:
        >>> suggestions = suggest_optimization(350, [580, 600, 590, 570], 0.98, 45, 110, 11.0)
        >>> for s in suggestions:
        ...     print(f"[{s['priority']}] {s['issue']}")
    """
    
    suggestions = []
    
    # --- Check 1: SO2 Emissions (Environmental & Efficiency) ---
    if so2_slip_ppm > 500:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Excessive SO2 emissions - possible environmental violation',
            'suggestion': 'Check catalyst activity and increase air ratio',
            'action': f'Increase air ratio from {air_ratio:.1f} to {air_ratio*1.1:.1f}'
        })
    elif so2_slip_ppm > 200:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': 'Elevated SO2 emissions above target',
            'suggestion': 'Monitor catalyst performance',
            'action': 'Schedule catalyst activity check within next month'
        })
    
    # --- Check 2: Temperature Excursions (Catalyst Protection) ---
    if temps:  # Only if we have temperature data
        max_temp = max(temps)
        
        if max_temp > 630:
            suggestions.append({
                'priority': 'HIGH',
                'issue': 'Catalyst temperature above safe limit',
                'suggestion': 'Risk of thermal damage to catalyst',
                'action': f'Reduce inlet temperature or verify interpass cooling (current max: {max_temp:.0f}°C)'
            })
        elif max_temp < 400:
            suggestions.append({
                'priority': 'MEDIUM',
                'issue': 'Low bed temperatures limiting reaction rate',
                'suggestion': 'Increase inlet temperature to improve conversion',
                'action': f'Current max temperature {max_temp:.0f}°C below optimum range'
            })
    
    # --- Check 3: Conversion Efficiency (Production Target) ---
    if conversion < 0.98:
        priority = 'HIGH' if conversion < 0.95 else 'MEDIUM'
        suggestions.append({
            'priority': priority,
            'issue': 'Suboptimal conversion efficiency',
            'suggestion': 'Review operating parameters and catalyst condition',
            'action': f'Current conversion {conversion*100:.1f}% below target of 99.5%'
        })
    
    # --- Check 4: Steam Production (Energy Recovery) ---
    # Typical steam-to-sulphur ratio: 4-5 tons steam per ton sulphur
    steam_ratio = steam_tph / (sulphur_tpd/24) if sulphur_tpd > 0 else 0
    
    if steam_ratio < 3.5:
        suggestions.append({
            'priority': 'LOW',
            'issue': 'Lower than expected steam production',
            'suggestion': 'Check heat recovery system efficiency',
            'action': f'Steam/sulphur ratio {steam_ratio:.2f} below typical range (3.5-5.0)'
        })
    elif steam_ratio > 5.5:
        suggestions.append({
            'priority': 'LOW',
            'issue': 'Higher than expected steam production',
            'suggestion': 'Verify temperature measurements',
            'action': 'Unusually high heat recovery - check instruments'
        })
    
    return suggestions


# ============================================================================
# SCENARIO MANAGEMENT
# ============================================================================

class ScenarioManager:
    """
    Manages saving, loading, and comparing operating scenarios.
    
    This class helps operators:
    - Save interesting operating points for later reference
    - Compare different scenarios to evaluate changes
    - Export/import scenarios for sharing between sites
    
    Example:
        >>> manager = ScenarioManager()
        >>> manager.save_scenario("base_case", params, results)
        >>> manager.save_scenario("increased_air", new_params, new_results)
        >>> comparison = manager.compare_scenarios("base_case", "increased_air")
    """
    
    def __init__(self):
        """Initialize an empty scenario manager."""
        self.scenarios = {}  # Dictionary of saved scenarios
    
    def save_scenario(self, name: str, params: Dict, results: Dict) -> Dict:
        """
        Save current operating scenario for later comparison.
        
        Args:
            name: Unique identifier for this scenario
            params: Input parameters used
            results: Model results obtained
        
        Returns:
            The saved scenario dictionary with timestamp
        """
        scenario = {
            'timestamp': datetime.now().isoformat(),
            'parameters': params.copy(),
            'results': results.copy() if results else {}
        }
        self.scenarios[name] = scenario
        return scenario
    
    def compare_scenarios(self, base_name: str, compare_name: str) -> Optional[Dict]:
        """
        Compare two saved scenarios and calculate differences.
        
        Args:
            base_name: Name of base scenario for comparison
            compare_name: Name of scenario to compare against base
        
        Returns:
            Dictionary containing:
                - base_name: Name of base scenario
                - compare_name: Name of compared scenario
                - parameter_deltas: Differences in input parameters
                - kpi_deltas: Differences in key performance indicators
            Returns None if either scenario doesn't exist
        """
        # Check if both scenarios exist
        if base_name not in self.scenarios or compare_name not in self.scenarios:
            return None
        
        base = self.scenarios[base_name]
        comp = self.scenarios[compare_name]
        
        # Initialize comparison dictionary
        comparison = {
            'base_name': base_name,
            'compare_name': compare_name,
            'parameter_deltas': {},
            'kpi_deltas': {}
        }
        
        # --- Compare Input Parameters ---
        for key in base['parameters']:
            if key in comp['parameters']:
                # Calculate absolute and percent differences
                delta = comp['parameters'][key] - base['parameters'][key]
                pct_change = (delta / base['parameters'][key] * 100) if base['parameters'][key] != 0 else 0
                
                comparison['parameter_deltas'][key] = {
                    'absolute': delta,
                    'percent': pct_change
                }
        
        # --- Compare Key Performance Indicators ---
        if base['results'] and comp['results']:
            # Define which KPIs to compare
            kpi_keys = ['conversion_total', 'steam_tph', 'stack_ppm', 'daily_profit_usd']
            
            for key in kpi_keys:
                if key in base['results'] and key in comp['results']:
                    delta = comp['results'][key] - base['results'][key]
                    pct_change = (delta / base['results'][key] * 100) if base['results'][key] != 0 else 0
                    
                    comparison['kpi_deltas'][key] = {
                        'absolute': delta,
                        'percent': pct_change
                    }
        
        return comparison
    
    def list_scenarios(self) -> List[str]:
        """Return list of all saved scenario names."""
        return list(self.scenarios.keys())
    
    def export_scenario(self, name: str, filename: str) -> bool:
        """
        Export a scenario to JSON file for sharing or backup.
        
        Args:
            name: Name of scenario to export
            filename: Path to save JSON file
        
        Returns:
            True if successful, False otherwise
        """
        if name in self.scenarios:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.scenarios[name], f, indent=2)
                return True
            except Exception as e:
                print(f"Error exporting scenario: {e}")
                return False
        return False
    
    def import_scenario(self, filename: str) -> Optional[str]:
        """
        Import a scenario from JSON file.
        
        Args:
            filename: Path to JSON file containing scenario
        
        Returns:
            Name of imported scenario if successful, None otherwise
        """
        try:
            with open(filename, 'r') as f:
                scenario = json.load(f)
            
            # Generate name from timestamp if not provided
            name = scenario.get('name', 
                               datetime.fromisoformat(scenario['timestamp']).strftime('%Y%m%d_%H%M%S'))
            
            self.scenarios[name] = scenario
            return name
            
        except Exception as e:
            print(f"Error importing scenario: {e}")
            return None


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis(base_params: Dict,
                        param_ranges: Dict[str, Tuple[float, float]],
                        model_func) -> Dict[str, List[Dict]]:
    """
    Perform sensitivity analysis on key parameters.
    
    This helps identify which parameters have the biggest impact on
    plant performance, guiding optimization efforts.
    
    Args:
        base_params: Dictionary of base case parameters
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        model_func: Function that takes parameters and returns a KPI value
    
    Returns:
        Dictionary of sensitivity results for each parameter
    
    Example:
        >>> def get_conversion(params):
        ...     results = run_model(params)
        ...     return results['conversion']
        >>> sens = sensitivity_analysis(base_params, 
        ...                           {'air_ratio': (9, 13)}, 
        ...                           get_conversion)
    """
    results = {}
    
    # Analyze each parameter in the specified ranges
    for param, (min_val, max_val) in param_ranges.items():
        if param not in base_params:
            continue
            
        base_val = base_params[param]
        sensitivities = []
        
        # Test at minimum, base, and maximum values
        test_vals = [min_val, base_val, max_val]
        
        for val in test_vals:
            # Create test case with this parameter changed
            test_params = base_params.copy()
            test_params[param] = val
            
            # Get KPI value for this test case
            kpi = model_func(test_params)
            
            sensitivities.append({
                'param_value': val,
                'kpi_value': kpi
            })
        
        # Calculate sensitivity (optional - can be added if needed)
        results[param] = sensitivities
    
    return results