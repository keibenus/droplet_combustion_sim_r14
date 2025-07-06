# =========================================
#         plotting.py (r4)
# =========================================
# plotting.py
"""Functions for plotting simulation results (works with FVM cell center data)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # For formatting axes
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd
import config
import grid # To regenerate grids for plotting
from properties import GasProperties # For type hinting

plt.rcParams.update({'font.size': 12}) # Adjust font size for readability

def plot_results(times, results_list, output_dir, nsp, Nl_init, Ng_init, gas_props: GasProperties):
    """
    Generates and saves plots from a list of state dictionaries (cell center values).
    Uses initial Nl/Ng values (Nl_init, Ng_init) for consistency in array access,
    but handles the transition to gas-only phase (Nl becomes 0).
    """
    print("Generating final plots...")
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    times_array = np.array(times)
    num_times = len(results_list)
    if num_times <= 1: # Need at least initial and one more point
        print("Not enough time points saved for plotting meaningful history.")
        return

    # --- Extract History Data ---
    Tl_surf_hist = np.full(num_times, np.nan) # Use NaN as default
    Tg_surf_hist = np.full(num_times, np.nan)
    T_interface_hist = np.full(num_times, np.nan) # <<< 追加 >>>
    R_hist = np.full(num_times, np.nan)
    P_hist = np.full(num_times, np.nan)
    maxTgas_hist = np.full(num_times, np.nan)
    fuel_idx = gas_props.fuel_idx if gas_props else -1

    # --- Extract data from results_list ---
    # We need T_s (interface temperature) history if available, but it wasn't explicitly saved
    # Let's try to get it from the CSV log file instead, as it's saved there.
    csv_log_file = os.path.join(config.OUTPUT_DIR, 'time_history_live.csv')
    log_data = None
    log_times = np.array([])
    try:
        log_data = pd.read_csv(csv_log_file)
        # Ensure log_times aligns with simulation times (using tolerance)
        log_times = log_data['Time (s)'].values
        T_interface_log = log_data['T_solved_interface (K)'].values
        # Interpolate log data onto simulation save times
        if len(log_times) > 1:
             interp_Ts = interp1d(log_times, T_interface_log, kind='linear', bounds_error=False, fill_value=np.nan)
             T_interface_hist = interp_Ts(times_array)
        else:
             T_interface_hist.fill(np.nan) # Not enough log points to interpolate

    except FileNotFoundError:
        print(f"Warning: Log file '{csv_log_file}' not found. Interface temperature history plot unavailable.")
        T_interface_hist.fill(np.nan)
    except Exception as e:
        print(f"Warning: Error reading or interpolating log file '{csv_log_file}': {e}")
        T_interface_hist.fill(np.nan)


    for i, state_dict in enumerate(results_list):
        # Check if keys exist and arrays have expected length based on initial grid sizes
        # Handle potential transition to gas-only phase where T_l might be empty
        T_l_state = state_dict.get('T_l', np.array([]))
        T_g_state = state_dict.get('T_g', np.array([]))
        R_state = state_dict.get('R', np.nan)
        P_state = state_dict.get('P', np.nan)

        if len(T_l_state) > 0: # Check if liquid phase exists in this state
            Tl_surf_hist[i] = T_l_state[-1]
        #else: Tl_surf_hist[i] = np.nan # Already initialized to NaN

        if len(T_g_state) > 0:
            Tg_surf_hist[i] = T_g_state[0]
            maxTgas_hist[i] = np.max(T_g_state)
        #else: Tg_surf_hist[i] = np.nan; maxTgas_hist[i] = np.nan

        R_hist[i] = R_state
        P_hist[i] = P_state

    # --- Plot 1: Surface/Interface Temperatures vs Time ---
    plt.figure(figsize=(10, 6))
    valid_Tl = ~np.isnan(Tl_surf_hist)
    valid_Tg = ~np.isnan(Tg_surf_hist)
    valid_Ts = ~np.isnan(T_interface_hist)

    if np.any(valid_Tl): plt.plot(times_array[valid_Tl], Tl_surf_hist[valid_Tl], 'b.-', markersize=4, label=f'Liquid Surf Cell (j={Nl_init-1})')
    if np.any(valid_Tg): plt.plot(times_array[valid_Tg], Tg_surf_hist[valid_Tg], 'r.-', markersize=4, label=f'Gas Surf Cell (i=0)')
    if np.any(valid_Ts): plt.plot(times_array[valid_Ts], T_interface_hist[valid_Ts], 'g.--', markersize=4, label='Interface Temp (Solved)')
    if np.any(valid_Tg): plt.plot(times_array[valid_Tg], maxTgas_hist[valid_Tg], 'm:', markersize=4, label='Max Gas Temp')

    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Characteristic Temperatures vs Time')
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    # Determine reasonable Y limits
    all_temps = np.concatenate([Tl_surf_hist, Tg_surf_hist, T_interface_hist, maxTgas_hist])
    min_temp = np.nanmin(all_temps) if not np.all(np.isnan(all_temps)) else config.T_L_INIT
    max_temp = np.nanmax(all_temps) if not np.all(np.isnan(all_temps)) else config.T_INF_INIT
    plt.ylim(bottom=max(0, min_temp - 100), top=max_temp + 200)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e')) # Scientific notation for time
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_history.png'))
    plt.close()

    # --- Plot 2: Droplet Radius vs Time ---
    valid_R = ~np.isnan(R_hist)
    if np.any(valid_R):
        plt.figure(figsize=(10, 6))
        plt.plot(times_array[valid_R], R_hist[valid_R] * 1e6, 'k.-', markersize=4) # Radius in micrometers
        plt.xlabel('Time (s)')
        plt.ylabel('Droplet Radius (μm)')
        plt.title(f'Droplet Radius vs Time (Initial R0 = {config.R0*1e6:.1f} μm)')
        plt.grid(True, which='both', linestyle=':')
        plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'droplet_radius.png'))
        plt.close()

    # --- Plot 3: Pressure vs Time ---
    valid_P = ~np.isnan(P_hist)
    if np.any(valid_P):
        plt.figure(figsize=(10, 6))
        plt.plot(times_array[valid_P], P_hist[valid_P] / 1e6, 'k.-', markersize=4) # Pressure in MPa
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (MPa)')
        plt.title(f'Vessel Pressure vs Time (Initial P = {config.P_INIT/1e6:.2f} MPa)')
        plt.grid(True, which='both', linestyle=':')
        # plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pressure_history.png'))
        plt.close()


    # --- Plot 4: Temperature Profiles at Different Times ---
    plt.figure(figsize=(12, 7))
    num_plots = min(num_times, 7) # Show up to 7 profiles
    plot_indices = []
    if num_times > 1:
        # Select indices: first, last, and some in between (logarithmically spaced in index)
        plot_indices = np.unique(np.geomspace(1, num_times, num=num_plots, dtype=int)) - 1
        if 0 not in plot_indices: plot_indices = np.insert(plot_indices, 0, 0)
        if num_times - 1 not in plot_indices: plot_indices = np.append(plot_indices, num_times - 1)
        plot_indices = np.unique(plot_indices) # Ensure unique indices
    elif num_times == 1:
        plot_indices = [0]

    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
    plotted_count = 0
    for k, i in enumerate(plot_indices):
        t_plot = times_array[i]
        state_dict = results_list[i]
        T_l_plot = state_dict.get('T_l', np.array([]))
        T_g_plot = state_dict.get('T_g', np.array([]))
        R_plot = state_dict.get('R', np.nan)

        if len(T_g_plot) != Ng_init or np.isnan(R_plot): # Need gas temp and radius
            print(f"Warning: Missing/invalid T_g or R data at index {i} (t={t_plot:.3e}). Skipping profile plot.")
            continue

        R_plot = max(R_plot, 1e-12) # Use very small R if liquid vanished

        # Regenerate grids (cell centers) for this radius
        Nl_plot = len(T_l_plot) # Current number of liquid points
        r_l_centers_plot, _, _ = grid.liquid_grid_fvm(R_plot, Nl_plot)
        r_g_centers_plot, _, _ = grid.gas_grid_fvm(R_plot, config.RMAX, Ng_init)

        # Combine for plotting: liquid centers + gas centers
        if Nl_plot > 0:
             r_combined = np.concatenate((r_l_centers_plot, r_g_centers_plot))
             T_combined = np.concatenate((T_l_plot, T_g_plot))
        else: # Gas phase only
             r_combined = r_g_centers_plot
             T_combined = T_g_plot

        plt.plot(r_combined / config.R0, T_combined, '.-', color=colors[k], label=f't = {t_plot:.3e} s', markersize=4, linewidth=1.5)
        plotted_count += 1

    if plotted_count > 0:
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Profiles at Cell Centers (Rmax/R0 = {config.R_RATIO:.2f})')
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        # plt.axvline(0, color='k', linestyle='--', linewidth=0.8) # Redundant if xlim starts at 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperature_profiles.png'))
    plt.close()

    # --- Plot 5: Fuel Mass Fraction Profile ---
    plt.figure(figsize=(12, 7))
    plotted_count = 0
    if fuel_idx != -1 and num_times > 1: # Need fuel index and data
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices))) # Use same indices/colors as T profile
        for k, i in enumerate(plot_indices):
            t_plot = times_array[i]
            state_dict = results_list[i]
            Y_g_plot = state_dict.get('Y_g')
            R_plot = state_dict.get('R', np.nan)

            if Y_g_plot is None or len(Y_g_plot) == 0 or Y_g_plot.shape != (nsp, Ng_init) or np.isnan(R_plot):
                # print(f"Warning: Missing/invalid Y_g or R data at index {i}. Skipping fuel profile.")
                continue # Skip if gas data is missing

            R_plot = max(R_plot, 1e-12)
            r_g_centers_plot, _, _ = grid.gas_grid_fvm(R_plot, config.RMAX, Ng_init)

            plt.plot(r_g_centers_plot / config.R0, Y_g_plot[fuel_idx, :], '.-', color=colors[k], label=f't = {t_plot:.3e} s', markersize=4, linewidth=1.5)
            plotted_count += 1

    if plotted_count > 0:
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel(f'Fuel ({config.FUEL_SPECIES_NAME}) Mass Fraction')
        plt.title('Fuel Mass Fraction Profiles at Cell Centers')
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        plt.ylim(bottom=-0.05, top=1.05) # Allow slightly below 0 for visibility
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fuel_fraction_profiles.png'))
    elif fuel_idx == -1:
        print("Skipping fuel fraction plot: fuel index not found.")
    else:
        print("No valid data points to plot fuel fraction profiles.")
    plt.close()

    # --- Plot 6: Other Species Profiles (e.g., O2, CO2, H2O) ---
    species_to_plot = ['o2', 'co2', 'h2o']
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
    plot_handles = {} # Store handles for legend

    for k, i in enumerate(plot_indices): # Iterate through selected time steps
        t_plot = times_array[i]
        state_dict = results_list[i]
        Y_g_plot = state_dict.get('Y_g')
        R_plot = state_dict.get('R', np.nan)

        if Y_g_plot is None or len(Y_g_plot) == 0 or Y_g_plot.shape != (nsp, Ng_init) or np.isnan(R_plot):
            continue # Skip if gas data is missing

        R_plot = max(R_plot, 1e-12)
        r_g_centers_plot, _, _ = grid.gas_grid_fvm(R_plot, config.RMAX, Ng_init)

        # Plot selected species at this time step
        for sp_name in species_to_plot:
            try:
                 sp_idx = gas_props.gas.species_index(sp_name)
                 line_style = '-' # Default solid line
                 if sp_name == 'o2': line_style = '--'
                 elif sp_name == 'co2': line_style = ':'
                 elif sp_name == 'h2o': line_style = '-.'

                 # Plot with color representing time, label representing species
                 handle, = plt.plot(r_g_centers_plot / config.R0, Y_g_plot[sp_idx, :],
                                    linestyle=line_style, color=colors[k],
                                    label=f'{sp_name} @ t={t_plot:.3e}s' if sp_name not in plot_handles else "", # Label only once per species
                                    markersize=4, linewidth=1.5)
                 if sp_name not in plot_handles:
                     plot_handles[sp_name] = handle # Store handle for combined legend

            except ValueError:
                 if config.LOG_LEVEL >= 0 and i == plot_indices[0]: print(f"Warning: Species '{sp_name}' not in mechanism, cannot plot profile.")
            except Exception as e_sp:
                 print(f"Error plotting species {sp_name} at t={t_plot:.3e}: {e_sp}")

    # Create a combined legend
    if plot_handles:
        # Add time legend separately? Or combine cleverly.
        # Use handles for species, maybe add time color bar?
        # Simple approach: legend showing species linestyle, mention color indicates time.
        time_label = f"Color: Early (Purple) -> Late (Yellow)"
        plt.legend(handles=list(plot_handles.values()), title=f"Species Profiles\n({time_label})", fontsize=10)
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel('Mass Fraction')
        plt.title('Key Species Mass Fraction Profiles')
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        plt.ylim(bottom=-0.05, top=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'other_species_profiles.png'))

    plt.close() # Close the species plot figure

    print(f"Plots saved in '{output_dir}' directory.")