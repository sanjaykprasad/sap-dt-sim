import numpy as np

# -------------------------
# Constants (rough but believable)
# -------------------------
CP_GAS = 1.1          # kJ/kg-K
DELTAH_SO2_SO3 = 990  # kJ/kg SO2 reacted (exothermic)
DELTAH_S_BURN = 9300  # kJ/kg sulphur combustion
LATENT_STEAM = 2100   # kJ/kg steam
AIR_O2 = 0.21

# -------------------------
# Burner Model
# -------------------------
def burner(sulphur_tpd, air_kg_s):
    """
    Returns: gas_flow, SO2_fraction, O2_fraction, temperature
    """

    m_s = sulphur_tpd * 1000 / 24  # kg/hr
    m_air = m_s * air_kg_s

    gas_flow = m_air + m_s

    # composition
    o2_in = m_air * AIR_O2
    o2_req = m_s * 1.0
    excess_o2 = max(o2_in - o2_req, 0)

    so2 = m_s / gas_flow
    o2 = excess_o2 / gas_flow

    # temperature rise from combustion
    Q = m_s * DELTAH_S_BURN
    dT = Q / (gas_flow * CP_GAS)

    T_out = 400 + dT   # assume air preheated ~400C

    return gas_flow, so2, o2, T_out

# -------------------------
# Converter Bed
# -------------------------
def equilibrium_conversion(T):
    """Simple equilibrium curve — decreases with temperature"""
    return 1 / (1 + np.exp(0.012*(T-440)))

def converter(gas_flow, so2, o2, T_in, activity):

    temps = []
    conversions = []

    for i in range(4):  # 4 beds

        Xeq = equilibrium_conversion(T_in)
        X = activity * Xeq

        reacted = so2 * X

        # temperature rise from reaction
        Q = reacted * gas_flow * DELTAH_SO2_SO3
        dT = Q / (gas_flow * CP_GAS)

        T_out = T_in + dT

        # update composition
        so2 = so2 * (1 - X)
        o2 = max(o2 - reacted*0.5, 0)

        temps.append(T_out)
        conversions.append(X)

        # interpass cooling
        T_in = T_out - 120

    return so2, o2, temps, conversions, T_out

# -------------------------
# Waste Heat Boiler
# -------------------------
def steam_generation(gas_flow, T_hot):
    T_stack = 180
    Q = gas_flow * CP_GAS * (T_hot - T_stack)
    steam = Q / LATENT_STEAM
    return steam
