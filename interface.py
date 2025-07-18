# =========================================
#        interface.py (r4)
# =========================================
# interface.py
"""Functions for calculating conditions at the liquid-gas interface (FVM context)."""

import numpy as np
import cantera as ct
import config
from properties import LiquidProperties, GasProperties, FugacityInterpolator # For type hinting
from numerics import gradient_at_face, harmonic_mean, arithmetic_mean # Import necessary helpers
import numerics
import scipy.optimize as op

def calculate_vapor_pressure(T_s, liquid_props: LiquidProperties):
    """Gets vapor pressure from liquid properties."""
    # Ensure T_s is within the range of the liquid property data
    T_s_phys = np.clip(T_s, liquid_props.T_min, liquid_props.T_max)
    return liquid_props.get_prop('vapor_pressure', T_s_phys) # Pa

def calculate_fuel_mass_fraction_surf(T_s, P, gas_props: GasProperties, liquid_props: LiquidProperties, fugacity_interpolator: FugacityInterpolator):
    """
    フガシティー係数比とポインティング補正を用いて、
    液滴表面の燃料質量分率(Y_f_eq)と組成(Y_eq)を計算する。
    """
    # 物理的な温度範囲にクリップ
    T_s_phys = np.clip(T_s, 200.0, 6000.0)

    # --- 1. 基本物理量の取得 ---
    # 飽和蒸気圧 [Pa]
    P_sat = calculate_vapor_pressure(T_s_phys, liquid_props)
    P_sat = min(max(P_sat, 0.0), P) # 物理的な上限は全圧P

    # 液相のモル体積 [m^3/mol]
    v_L = liquid_props.get_molar_volume(T_s_phys)
    
    # --- 2. 補正項の計算 ---
    # (a) ポインティング補正
    poynting_corr = 1.0
    if not np.isnan(v_L) and T_s_phys > 1e-6:
        try:
            poynting_corr = np.exp(v_L * (P - P_sat) / (config.R_UNIVERSAL * T_s_phys))
        except OverflowError:
            poynting_corr = 1.0 # 発散した場合は補正なしとする

    # (b) フガシティー係数比
    # 理想気体と仮定したモル分率を、混合気体の組成の推定に利用
    X_f_surf_ideal = P_sat / P if P > 1e-6 else 0.0
    
    # 飽和状態（純粋燃料、圧力P_sat）でのフガシティー係数φ_satを取得
    phi_sat = fugacity_interpolator.get_phi(T_s_phys, P_sat, 1.0)
    
    # 混合気体中（圧力P, 推定組成）でのフガシティー係数φ_mixを取得
    phi_mix = fugacity_interpolator.get_phi(T_s_phys, P, X_f_surf_ideal)

    phi_ratio = 1.0
    if phi_mix > 1e-4: # ゼロ割を防止
        phi_ratio = phi_sat / phi_mix

    # --- 3. 補正後の燃料モル分率を計算 ---
    X_f_surf = (P_sat / P) * poynting_corr * phi_ratio
    
    # 物理的な範囲 [0, 1] に収める
    X_f_surf = np.clip(X_f_surf, 0.0, 1.0)
    
    # --- 4. 全組成の計算 (既存ロジックを流用) ---
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx
    mw = gas_props.molecular_weights

    X_surf = np.zeros(nsp)
    X_surf[fuel_idx] = X_f_surf
    
    X_non_fuel_total = 1.0 - X_f_surf
    X_amb = gas_props.X_amb_init
    X_amb_non_fuel_sum = gas_props.X_amb_non_fuel_sum_init

    if X_amb_non_fuel_sum > 1e-9:
        for i in range(nsp):
            if i != fuel_idx:
                X_surf[i] = X_non_fuel_total * (max(0.0, X_amb[i]) / X_amb_non_fuel_sum)
    
    # 正規化と質量分率への変換
    X_surf /= np.sum(X_surf)
    mean_mw_surf = np.sum(X_surf * mw)
    Y_surf = X_surf * mw / mean_mw_surf if mean_mw_surf > 1e-6 else np.zeros(nsp)
    
    Y_f_surf_equil = Y_surf[fuel_idx]

    return Y_f_surf_equil, Y_surf

'''
def calculate_fuel_mass_fraction_surf(T_s, P, gas_props: GasProperties, liquid_props: LiquidProperties, fugacity_interpolator: FugacityInterpolator):
    """
    Calculates fuel mass fraction (Y_f_eq) and equilibrium composition array (Y_eq)
    at the surface assuming phase equilibrium based on vapor pressure.
    Uses pre-calculated ambient mole fractions from gas_props.
    """
    P_sat = calculate_vapor_pressure(T_s, liquid_props)
    # Ensure saturation pressure is physically bounded (0 <= Psat <= P)
    P_sat = min(max(P_sat, 0.0), P * 0.99999) # Avoid Psat = P
    X_f_surf = P_sat / P if P > 1e-6 else 0.0 # Fuel mole fraction at surface
    # 物理的な範囲 [0, 1] に収める
    X_f_surf = np.clip(X_f_surf, 0.0, 1.0) # Ensure physical bounds [0, 1]

    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx
    mw = gas_props.molecular_weights # kg/kmol

    X_surf = np.zeros(nsp)
    if fuel_idx < 0: # Should not happen if config is correct
        print("Error: Fuel index invalid in calculate_fuel_mass_fraction_surf.")
        return 0.0, X_surf # Return zero composition

    X_surf[fuel_idx] = X_f_surf

    # Distribute non-fuel components based on ambient mole fractions
    X_non_fuel_total = 1.0 - X_f_surf
    if X_non_fuel_total < 0.0: X_non_fuel_total = 0.0

    # ========== 変更箇所 START ==========
    # Use pre-calculated ambient mole fractions from gas_props
    X_amb = gas_props.X_amb_init
    X_amb_non_fuel_sum = gas_props.X_amb_non_fuel_sum_init
    # Remove the temporary Cantera object creation and state setting
    # temp_gas = ct.Solution(config.MECH_FILE) # REMOVED
    # try: ... except ... block related to temp_gas is REMOVED
    # ========== 変更箇所 END ==========

    # Distribute non-fuel species according to their relative ambient mole fractions
    if X_amb_non_fuel_sum > 1e-9:
        for i in range(nsp):
            if i != fuel_idx:
                # Ensure X_amb[i] is non-negative
                X_surf[i] = X_non_fuel_total * (max(0.0, X_amb[i]) / X_amb_non_fuel_sum)
    elif X_non_fuel_total > 0: # If ambient was pure fuel initially (unlikely)
        # Default to filling with N2 if available, otherwise the first species
        fill_idx = gas_props.n2_idx if gas_props.n2_idx >= 0 else 0
        if fill_idx == fuel_idx: fill_idx = 1 # Avoid filling with fuel again
        X_surf[fill_idx] = X_non_fuel_total

    # Normalize final mole fractions
    X_surf = np.maximum(X_surf, 0)
    sum_X = np.sum(X_surf)
    if sum_X > 1e-9: X_surf /= sum_X
    else: # Failsafe: if sum is still zero, set to pure N2 or first species
        X_surf.fill(0.0)
        fill_idx = gas_props.n2_idx if gas_props.n2_idx >= 0 else 0
        if fill_idx == fuel_idx: fill_idx = 1
        X_surf[fill_idx] = 1.0

    # Convert mole fractions to mass fractions
    mean_mw_surf = np.sum(X_surf * mw) # kg/kmol
    Y_surf = np.zeros(nsp)
    if mean_mw_surf > 1e-6:
         Y_surf = X_surf * mw / mean_mw_surf # kg/kg
         # Normalize mass fractions
         Y_surf = np.maximum(Y_surf, 0)
         sum_Y = np.sum(Y_surf)
         if abs(sum_Y - 1.0) > 1e-6: Y_surf /= sum_Y # Avoid division by zero or NaN
    else: # Failsafe
        Y_surf.fill(0.0)
        fill_idx = gas_props.n2_idx if gas_props.n2_idx >= 0 else 0
        if fill_idx == fuel_idx: fill_idx = 1
        Y_surf[fill_idx] = 1.0

    Y_f_surf_equil = Y_surf[fuel_idx] if fuel_idx >=0 else 0.0

    # Safety check for NaN Y_f
    if np.isnan(Y_f_surf_equil):
        if config.LOG_LEVEL >=1: print(f"Warning: NaN Y_f_surf_equil computed at T={T_s:.1f}, P={P:.1e}. Returning 0.")
        Y_f_surf_equil = 0.0
        Y_surf.fill(0.0)
        fill_idx = gas_props.n2_idx if gas_props.n2_idx >= 0 else 0
        if fill_idx == fuel_idx: fill_idx = 1
        Y_surf[fill_idx] = 1.0

    return Y_f_surf_equil, Y_surf
'''
def _interface_energy_residual_full(T_s_guess: float,
                              T_l_node_last: float, T_g_node0: float,
                              Y_g_node0: np.ndarray, P: float, R: float,
                              r_l_node_last_center: float, r_g_node0_center: float,
                              gas_props: GasProperties, liquid_props: LiquidProperties, Nl: int,
                              fugacity_interpolator: FugacityInterpolator):
    """
    Calculates the energy balance residual AND the corresponding fluxes/mdot.
    Returns: (residual, q_gas_out, q_liq_out, mdot)
    """
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx
    T_s_phys = np.clip(T_s_guess, 150.0, 6000.0)

    # Initialize return values for error cases
    q_gas_out_at_surf = 0.0
    q_liq_out_at_surf = 0.0
    mdot_double_prime = 0.0
    residual = 1e12 # Default large residual for errors

    try:
        # --- 1. Equilibrium composition and properties at surface ---
        Y_f_eq, Y_eq = calculate_fuel_mass_fraction_surf(T_s_phys, P, gas_props, liquid_props, fugacity_interpolator)
        props_l_s = liquid_props.get_properties(T_s_phys)
        lambda_l_s = props_l_s.get('thermal_conductivity', np.nan)
        Lv = props_l_s.get('heat_of_vaporization', np.nan)
        # ... (Rest of property calculations for Ts, Tg0, face) ...
        if not gas_props.set_state(T_s_phys, P, Y_eq): gas_props.set_state(gas_props.last_T, gas_props.last_P, gas_props.last_Y) # Fallback
        lambda_g_s = gas_props.gas.thermal_conductivity
        rho_g_s = gas_props.get_density(T_s_phys, P, Y_eq)
        Dk_s = gas_props.get_diffusion_coeffs(T_s_phys, P, Y_eq)
        if not gas_props.set_state(T_g_node0, P, Y_g_node0): gas_props.set_state(gas_props.last_T, gas_props.last_P, gas_props.last_Y) # Fallback
        lambda_g_0 = gas_props.gas.thermal_conductivity
        rho_g_0 = gas_props.get_density(T_g_node0, P, Y_g_node0)
        Dk_0 = gas_props.get_diffusion_coeffs(T_g_node0, P, Y_g_node0)
        if np.isnan(lambda_l_s) or np.isnan(Lv) or np.isnan(lambda_g_s) or np.isnan(rho_g_s) or np.any(np.isnan(Dk_s)) or \
           np.isnan(lambda_g_0) or np.isnan(rho_g_0) or np.any(np.isnan(Dk_0)):
            raise ValueError("NaN property calculated.")

        lambda_g_face = harmonic_mean(lambda_g_s, lambda_g_0)
        rho_g_face = arithmetic_mean(rho_g_s, rho_g_0)
        Dk_face = harmonic_mean(Dk_s, Dk_0)
        lambda_l_last = liquid_props.get_prop('thermal_conductivity', T_l_node_last) if Nl > 0 else lambda_l_s
        lambda_l_face = harmonic_mean(lambda_l_s, lambda_l_last) if Nl > 0 else lambda_l_s

        # --- 4. Calculate Gradients ---
        dr_centers_g = r_g_node0_center - R
        dr_centers_l = R - r_l_node_last_center if Nl > 0 else np.inf
        grad_Tg_face = (T_g_node0 - T_s_phys) / dr_centers_g if dr_centers_g > 1e-15 else 0.0
        grad_Yk_face = (Y_g_node0 - Y_eq) / dr_centers_g if dr_centers_g > 1e-15 else np.zeros(nsp)
        grad_Tl_face = (T_s_phys - T_l_node_last) / dr_centers_l if Nl > 0 and dr_centers_l > 1e-15 else 0.0
        #print(f"T_g_node0={T_g_node0} T_s_phys={T_s_phys} dr_centers_g={dr_centers_g}")
        #print(f"T_s_phys={T_s_phys} T_l_node_last={T_l_node_last} dr_centers_l={dr_centers_l}")

        # --- 5. Calculate Heat Fluxes OUTWARD ---
        q_gas_out_at_surf = -lambda_g_face * grad_Tg_face # r=0->R is positive 
        q_liq_out_at_surf = -lambda_l_face * grad_Tl_face # r=0->R is positive 
        #print(f"q_gas_out_at_surf={q_gas_out_at_surf} lambda_g_face={lambda_g_face} grad_Tg_face={grad_Tg_face}")
        #print(f"q_liq_out_at_surf={q_liq_out_at_surf} lambda_l_face={lambda_l_face} grad_Tl_face={grad_Tl_face}")


        # --- 6. Calculate Mass Flux (mdot'') ---
        mdot_double_prime = 0.0
        if fuel_idx >= 0:
             Y_f_eq_safe = min(Y_eq[fuel_idx], 0.99999)
             denominator = 1.0 - Y_f_eq_safe
             if np.isnan(rho_g_face) or np.isnan(Dk_face[fuel_idx]) or np.isnan(grad_Yk_face[fuel_idx]):
                  diff_flux_f_outwards = 0.0
             else:
                  diff_flux_f_outwards = - rho_g_face * Dk_face[fuel_idx] * grad_Yk_face[fuel_idx]

             if denominator > 1e-9:
                 mdot_double_prime = diff_flux_f_outwards / denominator
             else:
                 mdot_double_prime = 0.0
             mdot_double_prime = max(0.0, mdot_double_prime)

        # --- 7. Calculate Residual ---
        if np.isnan(Lv): raise ValueError("NaN Lv detected.") # Check Lv before use
        ###residual = (-q_gas_out_at_surf) + q_liq_out_at_surf - (mdot_double_prime * Lv)
        residual = q_gas_out_at_surf - q_liq_out_at_surf + mdot_double_prime * Lv

        # --- Debug Log ---
        if config.LOG_LEVEL >= 2:
             # (ログ出力コードは変更なし、ただし residual の各項を明確に表示)
             print(f"DEBUG RESIDUAL (Ts_guess={T_s_guess:.3f}K):")
             print(f"  Inputs: Tl_last={T_l_node_last:.2f} Tg0={T_g_node0:.2f} P={P:.3e} R={R:.3e}")
             print(f"  Surf State(Ts={T_s_phys:.3f}): Yf_eq={Y_f_eq:.5f}")
             print(f"  Fluxes Out: q_gas={q_gas_out_at_surf:.4e} q_liq={q_liq_out_at_surf:.4e} Jk_diff={diff_flux_f_outwards:.4e}")
             print(f"  Mdot Calc: denom={denominator:.5f} mdot={mdot_double_prime:.4e}")
             q_gas_in = -q_gas_out_at_surf
             q_liq_in = -q_liq_out_at_surf
             q_evap = mdot_double_prime * Lv
             print(f"  Residual Terms: (qg_in={q_gas_in:.4e}) + (ql_in={q_liq_in:.4e}) + (q_evap={q_evap:.4e}) = {residual:.4e}")
             print("-" * 20)

    except Exception as e_res:
        print(f"ERROR in _interface_energy_residual_full for T_s={T_s_phys:.2f}K: {e_res}")
        # Return safe defaults with large residual
        return 1e12, 0.0, 0.0, 0.0

    # <<< 変更: 計算された流束とmdotも返す >>>
    return residual, q_gas_out_at_surf, q_liq_out_at_surf, mdot_double_prime

# <<< 追加: brentq用のラッパー関数 >>>
def _residual_wrapper_for_solver(T_s_guess, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl, fugacity_interpolator):
    """Wrapper that calls _interface_energy_residual_full and returns only the residual."""
    residual, _, _, _ = _interface_energy_residual_full(T_s_guess, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl, fugacity_interpolator)
    return residual

def solve_interface_conditions(
    T_l_node_last: float, T_g_node0: float,
    Y_g_node0: np.ndarray, P: float, R: float,
    r_l_node_last_center: float, r_g_node0_center: float,
    gas_props: GasProperties, liquid_props: LiquidProperties, Nl: int,
    T_s_previous_step: float, # Use previous step's Ts as initial guess
    initial_Y_array: np.ndarray,
    fugacity_interpolator: FugacityInterpolator
    ):
    """
    Solves for the interface temperature (Ts) and returns consistent fluxes and mdot.
    Uses _interface_energy_residual_full to get all values consistently.
    """
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx

    # --- Define solver bounds and initial guess ---
    T_lower_bound = max(150.0, T_l_node_last - 10.0)
    T_upper_bound = min(T_g_node0 + 50.0, 6000.0)
    T_upper_bound = max(T_upper_bound, T_lower_bound + 0.1)
    T_s_guess_initial = np.clip(T_s_previous_step, T_lower_bound, T_upper_bound)

    T_s_final = T_s_guess_initial # Default value
    q_gas_out_surf_final = 0.0
    q_liq_out_surf_final = 0.0
    mdot_final = 0.0
    solved = False

    try:
        # Check residual at bounds using the wrapper function
        res_low, _, _, _ = _interface_energy_residual_full(T_lower_bound, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl, fugacity_interpolator)
        res_high, _, _, _ = _interface_energy_residual_full(T_upper_bound, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl, fugacity_interpolator)

        if not (np.isfinite(res_low) and np.isfinite(res_high)):
             print(f"Warning: Residual calculation failed at bounds [{T_lower_bound:.1f}, {T_upper_bound:.1f}]. Using guess T_s={T_s_final:.2f}K")
             # Keep default values (T_s_final = guess, fluxes/mdot = 0)
        elif np.sign(res_low) * np.sign(res_high) < 0: # Root is bracketed
            solver_xtol = config.INTERFACE_BRTQ_XTOL_T
            solver_rtol = config.INTERFACE_BRTQ_RTOL_T
            # <<< 変更: ラッパー関数をソルバーに渡す >>>
            T_s_solution, r_info = op.brentq(
                _residual_wrapper_for_solver, # Use the wrapper
                a=T_lower_bound, b=T_upper_bound,
                args=(T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl, fugacity_interpolator),
                xtol=solver_xtol, rtol=solver_rtol,
                maxiter=50, full_output=True
            )
            if r_info.converged:
                T_s_final = np.clip(T_s_solution, 150.0, 6000.0)
                solved = True
            else:
                 T_s_final = np.clip(T_s_solution, 150.0, 6000.0) # Use the unconverged result
                 if config.LOG_LEVEL >= 0: print(f"Warning: Interface Ts solver (brentq) failed. Flag: {r_info.flag}. Using T_s={T_s_final:.2f}K")
        else: # Root not bracketed
            if abs(res_low) < abs(res_high): T_s_final = T_lower_bound
            else: T_s_final = T_upper_bound
            if config.LOG_LEVEL >= 0: print(f"Warning: Interface Ts root not bracketed [{T_lower_bound:.1f}, {T_upper_bound:.1f}]. Res:[{res_low:.2e}, {res_high:.2e}]. Using best bound T_s={T_s_final:.2f}K")

    except Exception as e_solve:
        print(f"ERROR during interface Ts solve process: {e_solve}")
        # Keep default T_s_final = T_s_guess_initial

    # --- Get final consistent values by calling the full function once with T_s_final ---
    try:
        # <<< 変更: _interface_energy_residual_full を呼び出して最終値を取得 >>>
        final_residual_check, q_gas_out_surf_final, q_liq_out_surf_final, mdot_final = _interface_energy_residual_full(
            T_s_final, T_l_node_last, T_g_node0, Y_g_node0, P, R,
            r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl,
            fugacity_interpolator
        )

        # Final consistency check using the returned values
        Lv_final = liquid_props.get_prop('heat_of_vaporization', T_s_final) # Get Lv at final Ts
        q_vap_flux = abs(mdot_final * Lv_final) if not np.isnan(Lv_final) else 0.0
        residual_tolerance = max(q_vap_flux * 0.01, 1.0)
        if abs(final_residual_check) > residual_tolerance and config.LOG_LEVEL >= 0:
             print(f"Warning: Final interface energy balance residual = {final_residual_check:.3e} (Ts={T_s_final:.2f}K, mdot={mdot_final:.2e})")

    except Exception as e_final_call:
        print(f"ERROR getting final values from _interface_energy_residual_full at Ts={T_s_final:.2f}K: {e_final_call}")
        # Set safe defaults on error
        mdot_final = 0.0
        q_gas_out_surf_final = 0.0
        q_liq_out_surf_final = 0.0

    # --- Calculate final Y_eq and v_g_surf ---
    _, Y_eq_final = calculate_fuel_mass_fraction_surf(T_s_final, P, gas_props, liquid_props, fugacity_interpolator)
    rho_g_s_final = gas_props.get_density(T_s_final, P, Y_eq_final)
    v_g_surf_final = mdot_final / rho_g_s_final if rho_g_s_final > 1e-9 else 0.0

    return mdot_final, q_gas_out_surf_final, q_liq_out_surf_final, v_g_surf_final, Y_eq_final, T_s_final
