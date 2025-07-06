# =========================================
#          config.py (r4)
# =========================================
# config.py
"""Simulation configuration parameters."""

import numpy as np

# --- Restart Options ---
USE_RESTART = False  # Trueに設定するとリスタート機能を有効化
RESTART_FILE = 'restart_state.npz' # 状態保存ファイル名

# --- Logging Options ---
LOG_LEVEL = 1 # 0: Basic, 1: Info (Iteration details), 2: Debug (More verbose)
TERMINAL_OUTPUT_INTERVAL_STEP = 50 # Steps (Interval for printing status for LOG_LEVEL 0/1)
# --- Output Options ---
OUTPUT_DIR = 'results_r14_r3' # Changed output dir name
SAVE_INTERVAL_DT = 1e-5 # s (新しい方法: 保存したい時間間隔を指定、例: 1e-5 = 0.01 ms)

# --- Initial Conditions ---
T_INF_INIT = 773.0  # K (Initial ambient gas temperature)
P_INIT = 3.0e6     # Pa (Initial pressure)
X_INF_INIT = {'o2': 0.21, 'n2': 0.79} # Initial ambient gas composition (mole fractions)
###X_INF_INIT = {'nc7h16': 0.0187, 'o2': 0.2061, 'n2': 0.7752} # Initial ambient gas composition (mole fractions)
T_L_INIT = 300.0   # K (Initial liquid temperature)
R0 = 50e-6 #0.05e-3       # m (Initial droplet radius, 50 micronに変更)
FUEL_SPECIES_NAME = 'nc7h16' # Make sure this matches the name in the mechanism file

# --- Parameters for Initial Gas Temperature Profile ---
N_TRANSITION_CELLS = 5 # Number of gas cells near the droplet for the profile
PROFILE_TYPE = 'none' # Options: 'linear', 'tanh', 'none' (for uniform)

# --- Grid Parameters ---
NL = 10             # Number of liquid grid points (cell centers)
NG = 15             # Number of gas grid points (cell centers)
R_RATIO = 9.12      # rmax / R0 (論文の Φ=1 に対応する値)
RMAX = R_RATIO * R0 # Maximum radius of computational domain (m)
GRID_TYPE = 'geometric_series' # Options: 'power_law', 'geometric_series'
# XI_GAS_GRID = 2.0 # Parameter for power_law grid (Not used if GRID_TYPE is 'geometric_series')

# --- Phase Transition ---
###R_TRANSITION_RATIO = 1.05 # Relative radius threshold for switching to gas-only phase
R_TRANSITION_RATIO = 0.05 # Relative radius threshold for switching to gas-only phase
R_TRANSITION_THRESHOLD = R0 * R_TRANSITION_RATIO # m (Radius below which liquid phase calculations stop)

# --- Time Integration Parameters ---
#T_END = 4.58e-3 # s (Simulation end time, 延長)
T_END = 8e-3 # s (Simulation end time, 延長)
DT_INIT = 1e-9               # s (Initial time step)
DT_MAX = 1e-4                # s (Maximum allowed time step)
DT_POST_EVAPORATION = 5.0e-6   # s (例: 0.1マイクロ秒)

# --- Adaptive Time Stepping Parameters ---
USE_ADAPTIVE_DT = True       # True to enable adaptive time stepping
CFL_NUMBER = 5.0             # Courant-Friedrichs-Lewy number (safety factor, typically < 1)
DT_MAX_INCREASE_FACTOR = 1.2 # Maximum factor by which dt can increase per step (e.g., 1.2 = 20% increase)
DT_MIN_VALUE = 1e-12         # Absolute minimum dt allowed (Avoid too small dt)
# TIME_STEPPING_MODE = 'BDF' # Not used in custom loop

# --- Interface Iteration Parameters ---
MAX_INTERFACE_ITER = 50      # Maximum iterations for interface coupling per time step
INTERFACE_ITER_TOL_T = 1 #1e-3  # Convergence tolerance for interface temperature (K)
INTERFACE_ITER_TOL_MDOT = 0.1 #1e-4 # Convergence tolerance for relative mdot change (-)
INTERFACE_RELAX_FACTOR = 0.5 # Relaxation factor for interface iteration (0 < alpha <= 1)
INTERFACE_BRTQ_XTOL_T = 1e-4 # Convergence absolute tolerance for brentq solver
INTERFACE_BRTQ_RTOL_T = 1e-7 # Convergence absolute tolerance for brentq solver

# --- Maximum Number of Steps ---
MAX_STEPS = 500000            # Safety break for total number of steps

# --- Model Options ---
USE_RK_EOS = False            # True to use Redlich-Kwong EOS for density (False recommended initially)
# --- !!! REACTION_TYPE: Choose 'detailed', 'overall', or 'none' !!! ---
REACTION_TYPE = 'detailed'     # Options: 'detailed', 'overall', 'none'
# --- !!! DIFFUSION_OPTION: Choose 'constant', 'Le=1', or 'mixture_averaged' !!! ---
DIFFUSION_OPTION = 'mixture_averaged' # Options: 'constant', 'Le=1', 'mixture_averaged'
# --- !!! ADVECTION_SCHEME: Choose 'upwind' or 'central' !!! ---
ADVECTION_SCHEME = 'upwind'   # Options: 'upwind', 'central' (ユーザー指定: 一次風上)

# --- Overall Reaction Parameters (if REACTION_TYPE = 'overall') ---
OVERALL_B_CM = 4.4e16     # cm^3 / (mol * s)
OVERALL_E_KJ = 209.2      # kJ / mol
OVERALL_E_SI = OVERALL_E_KJ * 1000.0 # J / mol
OVERALL_B_SI = OVERALL_B_CM * 1e-6 # m^3 / (mol * s)
OVERALL_FUEL = FUEL_SPECIES_NAME
OVERALL_OXIDIZER = 'o2'

# --- Diffusion Parameter (if DIFFUSION_OPTION = 'constant') ---
DIFFUSION_CONSTANT_VALUE = 1e-5 # m^2/s

# --- Files ---
LIQUID_PROP_FILE = 'n_heptane_liquid_properties.csv'
MECH_FILE = 'mech_LLNL_reduce.yaml' # ユーザー提供のYAMLファイル
FUGACITY_MAP_FILE = 'fugacity_map_Fluid_n-heptane_SRK.csv'

# --- Termination Criteria ---
IGNITION_CRITERION_DTDT = 1.0e7  # K/s (Max gas temperature rise rate) - Not implemented as termination in this version
IGNITION_CRITERION_TMAX = 1200.0  # K (Max gas temperature)
EXTINCTION_CRITERION_DTDT = 1.0  # K/s (Max gas temperature rise rate threshold for extinction) - Not implemented

# --- Reaction Calculation Cutoff ---
ENABLE_REACTION_CUTOFF = True     # True to enable cutoff, False to calculate everywhere
REACTION_CALC_MIN_TEMP = 600.0    # K (Minimum temperature to calculate reactions)
REACTION_CALC_MIN_FUEL_MOL_FRAC = 1e-8 # mol/mol (Minimum fuel mole fraction)

# --- Physical Constants ---
R_UNIVERSAL = 8.31446261815324 # J/(mol·K)

# --- Numerical Parameters ---
# SOLVER_TOL = 1e-4 # Relative tolerance for solve_ivp (Not used)
# ATOL_FACTOR = 1e-6 # Absolute tolerance factor (Not used)
MAX_ITER_RK = 10   # Max iterations for RK EOS solver
RK_SOLVER_TOL = 1e-7 # Tolerance for solving cubic EOS

# --- Redlich-Kwong Parameters (Only needed if USE_RK_EOS = True) ---
# Critical Temp (K), Critical Press (Pa), Acentric factor (-)
# Values from NIST, Dortmund Data Bank, Reid et al. (verify consistency)
RK_PARAMS = {
    'n2':     {'Tc': 126.19, 'Pc': 3.3958e6, 'omega': 0.0372},
    'o2':     {'Tc': 154.58, 'Pc': 5.043e6,  'omega': 0.0222},
    'nc7h16': {'Tc': 540.2,  'Pc': 2.74e6,   'omega': 0.351}, # n-Heptane
    'co2':    {'Tc': 304.13, 'Pc': 7.3773e6, 'omega': 0.2239},
    'h2o':    {'Tc': 647.10, 'Pc': 22.064e6, 'omega': 0.3449}
    # Add others if significant concentrations are expected AND reliable data exists
}
# RK_MIXING_RULE = 'van_der_waals' # Simple VdW mixing rules (Currently only used in properties.py if RK is enabled)
# Radial Profile Output Settings (課題①)
# --------------------------------------------------------------------------
# 半径方向プロファイル（温度分布など）の出力を有効にするか
SAVE_RADIAL_PROFILES = True

# プロファイルを出力する時間間隔 [s] (例: 2.0e-4 s = 0.2 ms ごと)
OUTPUT_TIME_INTERVAL = 0.5e-3

# 半径方向プロファイルのCSVが保存されるサブディレクトリ名
RADIAL_CSV_SUBDIR = 'radial_profiles'

# CSVに出力したい化学種のリスト (Canteraでの名称と一致させること)
SPECIES_TO_OUTPUT = [
    'nc7h16', 'o2', 'n2', 'co2', 'h2o',
    'oh', 'h2o2', 'ch2o'
]