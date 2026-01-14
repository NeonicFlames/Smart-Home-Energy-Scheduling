import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ==========================================
# 1. SETUP: DATA & CONSTANTS
# ==========================================

# Malaysian Tariff (Tariff B - Domestic)
OFF_PEAK_RATE = 0.4183  # RM/kWh (22:00 - 08:00)
PEAK_RATE = 0.4592      # RM/kWh (08:00 - 22:00)

# Benchmark Dataset (Standard for Project)
TASKS_DATA = [
    # Non-Shiftable Appliances (Must run at preferred time)
    {'id': 1,  'name': 'Fridge',           'power': 0.30, 'pref': 0,  'dur': 24, 'shift': False},
    {'id': 2,  'name': 'Lights (AM)',      'power': 1.09, 'pref': 6,  'dur': 2,  'shift': False},
    {'id': 3,  'name': 'Lights (PM)',      'power': 1.09, 'pref': 18, 'dur': 5,  'shift': False},
    {'id': 4,  'name': 'TV',               'power': 1.10, 'pref': 19, 'dur': 3,  'shift': False},
    {'id': 5,  'name': 'Microwave (L)',    'power': 1.10, 'pref': 12, 'dur': 1,  'shift': False},
    {'id': 6,  'name': 'Microwave (D)',    'power': 1.10, 'pref': 19, 'dur': 1,  'shift': False},
    {'id': 7,  'name': 'AC',               'power': 3.50, 'pref': 13, 'dur': 4,  'shift': False},
    
    # Shiftable Appliances (Optimization Targets)
    {'id': 8,  'name': 'Washing Machine',  'power': 1.10, 'pref': 9,  'dur': 2,  'shift': True},
    {'id': 9,  'name': 'Dishwasher',       'power': 1.10, 'pref': 20, 'dur': 2,  'shift': True},
    {'id': 10, 'name': 'Heater',           'power': 3.49, 'pref': 18, 'dur': 4,  'shift': True}
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_hourly_rate(hour):
    if 0 <= hour < 8 or 22 <= hour <= 23:
        return OFF_PEAK_RATE
    else:
        return PEAK_RATE

def calculate_task_cost(power, start_hour, duration):
    cost = 0
    for h in range(start_hour, start_hour + duration):
        current_hour = h % 24
        cost += power * get_hourly_rate(current_hour)
    return cost

def check_peak_power_constraint(schedule_dict, max_peak_power):
    hourly_load = [0.0] * 24
    for task in TASKS_DATA:
        tid = task['id']
        start_time = schedule_dict.get(tid, task['pref'])
        for h in range(task['dur']):
            hour = (start_time + h) % 24
            hourly_load[hour] += task['power']
    
    max_power = max(hourly_load)
    is_valid = max_power <= max_peak_power
    return is_valid, max_power, hourly_load

def evaluate_schedule(schedule_dict, max_peak_power):
    total_cost = 0
    total_discomfort = 0
    
    for task in TASKS_DATA:
        tid = task['id']
        start_time = schedule_dict.get(tid, task['pref'])
        total_cost += calculate_task_cost(task['power'], start_time, task['dur'])
        diff = abs(start_time - task['pref'])
        total_discomfort += diff
    
    is_valid, max_power, _ = check_peak_power_constraint(schedule_dict, max_peak_power)
    return total_cost, total_discomfort, is_valid, max_power

# ==========================================
# 3. ACO ALGORITHM
# ==========================================

def run_aco(num_ants, num_iterations, alpha, beta, evaporation_rate, w_discomfort, max_peak_power, progress_bar=None):
    shiftable_ids = [t['id'] for t in TASKS_DATA if t['shift']]
    pheromone = np.ones((len(shiftable_ids), 24)) * 0.1
    
    best_overall_fitness = float('inf')
    best_overall_schedule = {}
    best_overall_metrics = (0, 0, False, 0) # cost, discomfort, valid, max_power
    
    history = {
        'iteration': [],
        'cost': [],
        'discomfort': [],
        'fitness': [],
        # Enhanced tracking for iteration analysis
        'valid_count': [],      # Number of valid solutions per iteration
        'invalid_count': [],    # Number of invalid solutions per iteration
        'avg_violation': [],    # Average constraint violation in kW
        'global_best_cost': [], # Global best cost up to this iteration (cumulative)
        'iter_best_cost': [],   # Best ant cost in this iteration only
        'iter_worst_cost': []   # Worst ant cost in this iteration only
    }
    
    Q = 100.0
    global_best_cost = float('inf')  # Track cumulative best

    for iteration in range(num_iterations):
        ant_paths = []
        
        # Track iteration statistics
        valid_ants = 0
        total_violation = 0
        ant_costs = []
        
        for ant in range(num_ants):
            current_schedule = {}
            
            for idx, tid in enumerate(shiftable_ids):
                task = next(t for t in TASKS_DATA if t['id'] == tid)
                probabilities = []
                
                for h in range(24):
                    tau = pheromone[idx][h]
                    cost = calculate_task_cost(task['power'], h, task['dur'])
                    discomfort = abs(h - task['pref']) * w_discomfort
                    eta = 1.0 / (cost + discomfort + 0.001)
                    prob = (tau ** alpha) * (eta ** beta)
                    probabilities.append(prob)
                
                probabilities = np.array(probabilities)
                probabilities = probabilities / probabilities.sum()
                selected_hour = np.random.choice(range(24), p=probabilities)
                current_schedule[tid] = selected_hour
            
            cost, discomfort, is_valid, max_power = evaluate_schedule(current_schedule, max_peak_power)
            
            if is_valid:
                fitness = cost + (discomfort * w_discomfort)
            else:
                violation = max_power - max_peak_power
                penalty = 1000 * violation
                fitness = cost + (discomfort * w_discomfort) + penalty
            
            ant_paths.append((current_schedule, fitness, cost, discomfort, is_valid, max_power))
            
            # Track statistics for this ant
            if is_valid:
                valid_ants += 1
            else:
                total_violation += (max_power - max_peak_power)
            ant_costs.append(cost)
            
            if is_valid and fitness < best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_schedule = current_schedule.copy()
                best_overall_metrics = (cost, discomfort, is_valid, max_power)
        
        # Pheromone Update
        pheromone *= (1 - evaporation_rate)
        for schedule, fitness, cost, discomfort, is_valid, max_power in ant_paths:
            if is_valid:
                deposit = Q / fitness
                for idx, tid in enumerate(shiftable_ids):
                    hour = schedule[tid]
                    pheromone[idx][hour] += deposit
        
        # Record History (Enhanced)
        # Track iteration statistics
        iter_best = min(ant_costs) if ant_costs else float('inf')
        iter_worst = max(ant_costs) if ant_costs else 0
        
        # Update global best (cumulative - can only improve)
        if iter_best < global_best_cost:
            global_best_cost = iter_best
        
        history['iteration'].append(iteration + 1)
        history['valid_count'].append(valid_ants)
        history['invalid_count'].append(num_ants - valid_ants)
        history['avg_violation'].append(total_violation / num_ants if num_ants > 0 else 0)
        history['global_best_cost'].append(global_best_cost)  # Cumulative best
        history['iter_best_cost'].append(iter_best)           # This iteration's best
        history['iter_worst_cost'].append(iter_worst)         # This iteration's worst
        
        if best_overall_metrics[2]: # If valid solution found yet
           history['cost'].append(best_overall_metrics[0])
           history['discomfort'].append(best_overall_metrics[1])
           history['fitness'].append(best_overall_fitness)
        else:
           # No valid solution yet, append placeholder
           history['cost'].append(None)
           history['discomfort'].append(None)
           history['fitness'].append(None)
        
        if progress_bar:
            progress_bar.progress((iteration + 1) / num_iterations)
            
    return best_overall_schedule, best_overall_metrics, history

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Smart Home Energy Scheduler", layout="wide")

st.title("Smart Home Energy Scheduling using Ant Colony Optimization")


# Data Overview
st.subheader("System Data & Tariffs")

# 1. Tariff Rates (Top)
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Off-Peak Rate** (22:00-08:00): **RM {OFF_PEAK_RATE:.3f}/kWh**")
with col2:
    st.warning(f"**Peak Rate** (08:00-22:00): **RM {PEAK_RATE:.3f}/kWh**")



# Initialize session state
if 'shiftable_states' not in st.session_state:
    st.session_state.shiftable_states = {task['id']: task['shift'] for task in TASKS_DATA}

# Create custom table with checkboxes
# Table Header
header_cols = st.columns([3, 1.5, 2, 1.5, 1.5])
with header_cols[0]:
    st.markdown("**Appliance**")
with header_cols[1]:
    st.markdown("**Power (kW)**")
with header_cols[2]:
    st.markdown("**Preferred Time**")
with header_cols[3]:
    st.markdown("**Duration**")
with header_cols[4]:
    st.markdown("**Shiftable**")

st.markdown("---")

# Table Rows
for task in TASKS_DATA:
    row_cols = st.columns([3, 1.5, 2, 1.5, 1.5])
    
    with row_cols[0]:
        st.write(task['name'])
    with row_cols[1]:
        st.write(f"{task['power']:.2f}")
    with row_cols[2]:
        st.write(f"{task['pref']:02d}:00")
    with row_cols[3]:
        st.write(f"{task['dur']}h")
    with row_cols[4]:
        # Checkbox in the table
        is_shiftable = st.checkbox(
            "✓",
            value=st.session_state.shiftable_states[task['id']],
            key=f"shift_{task['id']}",
            label_visibility="collapsed"
        )
        st.session_state.shiftable_states[task['id']] = is_shiftable

# Update TASKS_DATA based on checkbox states
for task in TASKS_DATA:
    task['shift'] = st.session_state.shiftable_states[task['id']]



# Baseline Metrics (Before Optimization)
st.markdown("**Baseline (No Optimization):**")
baseline_schedule = {t['id']: t['pref'] for t in TASKS_DATA if t['shift']}
base_cost, base_disc, base_valid, base_peak = evaluate_schedule(baseline_schedule, 5.0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline Cost", f"RM {base_cost:.2f}")
with col2:
    st.metric("Baseline Discomfort", f"{base_disc} hrs")
with col3:
    st.metric("Baseline Peak Power", f"{base_peak:.2f} kW")

# Baseline Schedule Visualization (Gantt Chart)
st.markdown("**Baseline Schedule Timeline:**")
fig_baseline, ax_baseline = plt.subplots(figsize=(15, 6))

# Background shading for peak/off-peak
ax_baseline.axvspan(0, 8, alpha=0.15, color='green', label='Off-Peak Hours')
ax_baseline.axvspan(22, 24, alpha=0.15, color='green')
ax_baseline.axvspan(8, 22, alpha=0.15, color='red', label='Peak Hours')

# Plot each appliance
y_pos = 0
yticks = []
ylabels = []

for task in TASKS_DATA:
    start_hour = task['pref']
    duration = task['dur']
    
    # Determine color based on shiftable status
    if task['shift']:
        color = '#3498db'  # Blue for shiftable
        label = 'Shiftable' if y_pos == 0 else ""
    else:
        color = '#95a5a6'  # Gray for non-shiftable
        label = 'Non-Shiftable' if y_pos == 0 else ""
    
    # Draw horizontal bar for appliance runtime
    ax_baseline.barh(y_pos, duration, left=start_hour, height=0.8, 
                     color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    yticks.append(y_pos)
    ylabels.append(f"{task['name']} ({task['power']} kW)")
    y_pos += 1

ax_baseline.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax_baseline.set_ylabel('Appliances', fontsize=12, fontweight='bold')
ax_baseline.set_title('Baseline Appliance Schedule (24-Hour View)', fontsize=14, fontweight='bold')
ax_baseline.set_yticks(yticks)
ax_baseline.set_yticklabels(ylabels, fontsize=10)
ax_baseline.set_xticks(range(0, 25, 2))
ax_baseline.set_xlim(0, 24)
ax_baseline.grid(axis='x', alpha=0.3, linestyle='--')
ax_baseline.legend(loc='upper right', fontsize=10)

plt.tight_layout()
st.pyplot(fig_baseline)



# Sidebar

st.sidebar.header("Optimization Settings")

# Parameter Presets
st.sidebar.subheader("Quick Presets")
col1, col2, col3 = st.sidebar.columns(3)



with col1:
    if st.button("Conservative", use_container_width=True, help="Fewer ants, less exploration"):
        st.session_state.preset = "conservative"
        
with col2:
    if st.button("Balanced", use_container_width=True, help="Default balanced settings"):
        st.session_state.preset = "balanced"
        
with col3:
    if st.button("Aggressive", use_container_width=True, help="More ants, deeper exploration"):
        st.session_state.preset = "aggressive"

# Defaulted on Balanced
if 'preset' not in st.session_state:
    st.session_state.preset = "balanced"

# Define preset configurations
PRESETS = {
    "conservative": {
        "ants": 10,
        "iterations": 30,
        "alpha": 1.0,
        "beta": 2.5,
        "evaporation": 0.3
    },
    "balanced": {
        "ants": 20,
        "iterations": 50,
        "alpha": 1.0,
        "beta": 2.0,
        "evaporation": 0.5
    },
    "aggressive": {
        "ants": 50,
        "iterations": 100,
        "alpha": 1.5,
        "beta": 1.5,
        "evaporation": 0.7
    }
}

# Get current preset
current = PRESETS[st.session_state.preset]

with st.sidebar.expander("ACO Hyperparameters", expanded=True):
    NUM_ANTS = st.number_input("Number of Ants", min_value=5, max_value=100, value=current["ants"], step=5)
    NUM_ITERATIONS = st.number_input("Iterations", min_value=10, max_value=500, value=current["iterations"], step=10)
    ALPHA = st.slider("Alpha (Pheromone Importance)", 0.0, 5.0, current["alpha"], 0.1)
    BETA = st.slider("Beta (Heuristic Importance)", 0.0, 5.0, current["beta"], 0.1)
    EVAPORATION_RATE = st.slider("Evaporation Rate", 0.0, 1.0, current["evaporation"], 0.05)

with st.sidebar.expander("Constraints & Weights", expanded=True):
    MAX_PEAK_POWER = st.number_input("Max Peak Power (kW)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    W_DISCOMFORT = st.slider("Discomfort Weight (Cost of 1hr Delay)", 0.0, 2.0, 0.1, 0.1)

# Main Execution
if st.button("Start Optimization", type="primary"):
    
    # Baseline Calculation
    baseline_schedule = {t['id']: t['pref'] for t in TASKS_DATA if t['shift']}
    base_cost, base_disc, base_valid, base_peak = evaluate_schedule(baseline_schedule, MAX_PEAK_POWER) # Pass MAX_PEAK_POWER
    
    # Run Optimization
    with st.spinner("Running ACO Algorithm..."):
        progress_bar = st.progress(0)
        best_schedule, metrics, history = run_aco(
            NUM_ANTS, NUM_ITERATIONS, ALPHA, BETA, EVAPORATION_RATE, W_DISCOMFORT, MAX_PEAK_POWER, progress_bar
        )
    
    # Results Section
    st.divider()
    st.header("Optimization Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"RM {metrics[0]:.2f}", delta=f"{metrics[0] - base_cost:.2f} RM", delta_color="inverse")
    with col2:
        st.metric("Total Discomfort", f"{metrics[1]} hrs", delta=f"{metrics[1] - base_disc} hrs", delta_color="inverse")
    with col3:
        st.metric("Peak Power", f"{metrics[3]:.2f} kW", f"Limit: {MAX_PEAK_POWER} kW")


  
    tab1, tab2, tab3, tab4 = st.tabs(["Convergence", "Load Profile", "Schedule Details", "Iteration Details"])
    
    with tab1:
        if history['iteration']:
            # Filter out None values for plotting (occurs when no valid solutions found)
            valid_indices = [i for i, c in enumerate(history['cost']) if c is not None]
            
            if valid_indices:
                # Create filtered arrays with only valid data points
                valid_iterations = [history['iteration'][i] for i in valid_indices]
                valid_cost = [history['cost'][i] for i in valid_indices]
                valid_discomfort = [history['discomfort'][i] for i in valid_indices]
                valid_fitness = [history['fitness'][i] for i in valid_indices]
                
                fig, ax = plt.subplots(2, 2, figsize=(16, 10))
                
                # Plot 1: Cost Convergence
                ax[0, 0].plot(valid_iterations, valid_cost, 'g-', linewidth=2, label='Best Cost')
                ax[0, 0].axhline(y=base_cost, color='r', linestyle='--', linewidth=2, label='Baseline')
                ax[0, 0].fill_between(valid_iterations, valid_cost, base_cost, alpha=0.3, color='green')
                ax[0, 0].set_xlabel('Iteration', fontsize=11)
                ax[0, 0].set_ylabel('Cost (RM)', fontsize=11)
                ax[0, 0].set_title('Cost Optimization Over Iterations', fontsize=13, fontweight='bold')
                ax[0, 0].legend(fontsize=10)
                ax[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Discomfort Evolution
                ax[0, 1].plot(valid_iterations, valid_discomfort, 'b-', linewidth=2, marker='o', markersize=4)
                ax[0, 1].set_xlabel('Iteration', fontsize=11)
                ax[0, 1].set_ylabel('Discomfort (Hours)', fontsize=11)
                ax[0, 1].set_title('User Discomfort Evolution', fontsize=13, fontweight='bold')
                ax[0, 1].grid(True, alpha=0.3)
                ax[0, 1].fill_between(valid_iterations, 0, valid_discomfort, alpha=0.3, color='blue')
                
                # Plot 3: Fitness Convergence (Combined Objective)
                ax[1, 0].plot(valid_iterations, valid_fitness, 'purple', linewidth=2, marker='s', markersize=4)
                ax[1, 0].set_xlabel('Iteration', fontsize=11)
                ax[1, 0].set_ylabel('Fitness (Lower is Better)', fontsize=11)
                ax[1, 0].set_title('Combined Fitness Function', fontsize=13, fontweight='bold')
                ax[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Pareto Front (Cost vs Discomfort Trade-off)
                scatter = ax[1, 1].scatter(valid_cost, valid_discomfort, 
                                          c=valid_iterations, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
                ax[1, 1].scatter(base_cost, base_disc, color='red', marker='*', s=400, 
                               label='Baseline', zorder=5, edgecolors='darkred', linewidths=2)
                ax[1, 1].set_xlabel('Cost (RM)', fontsize=11)
                ax[1, 1].set_ylabel('Discomfort (Hours)', fontsize=11)
                ax[1, 1].set_title('Pareto Front: Multi-Objective Trade-off', fontsize=13, fontweight='bold')
                ax[1, 1].legend(fontsize=10)
                ax[1, 1].grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax[1, 1])
                cbar.set_label('Iteration', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("**No valid solutions found** Minimum Required Power is too low")
        else:
            st.warning("No valid solutions found to plot convergence.")

    with tab2:
        # Load Profile Plot
        hours = range(24)
        _, _, baseline_load = check_peak_power_constraint(baseline_schedule, MAX_PEAK_POWER)
        
        final_full_schedule = baseline_schedule.copy()
        final_full_schedule.update(best_schedule)
        _, _, optimized_load = check_peak_power_constraint(final_full_schedule, MAX_PEAK_POWER)
        
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        
        width = 0.35
        x = np.arange(len(hours))
        ax2.bar(x - width/2, baseline_load, width, label='Baseline', alpha=0.7)
        ax2.bar(x + width/2, optimized_load, width, label='Optimized', alpha=0.7)
        ax2.axhline(y=MAX_PEAK_POWER, color='red', linestyle='--', label='Max Power Limit')
        
        # Shade regions
        ax2.axvspan(0, 8, alpha=0.1, color='green', label='Off-Peak')
        ax2.axvspan(22, 24, alpha=0.1, color='green')
        ax2.axvspan(8, 22, alpha=0.1, color='red', label='Peak')

        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('24-Hour Load Profile')
        ax2.set_xticks(hours)
        ax2.legend()
        
        st.pyplot(fig2)

    with tab3:
        # Detailed Schedule Table
        data = []
        for task in TASKS_DATA:
            if task['shift']:
                tid = task['id']
                old_start = task['pref']
                new_start = best_schedule.get(tid, old_start)
                old_cost = calculate_task_cost(task['power'], old_start, task['dur'])
                new_cost = calculate_task_cost(task['power'], new_start, task['dur'])
                
                data.append({
                    "Appliance": task['name'],
                    "Preferred Start": f"{old_start:02d}:00",
                    "Optimized Start": f"{new_start:02d}:00",
                    "Delay (Hrs)": abs(new_start - old_start),
                    "Original Cost (RM)": f"{old_cost:.2f}",
                    "New Cost (RM)": f"{new_cost:.2f}",
                    "Savings (RM)": f"{old_cost - new_cost:.2f}"
                })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    with tab4:
        if history['iteration']:
            # Create detailed DataFrame
            iteration_data = []
            for i in range(len(history['iteration'])):
                iteration_data.append({
                    'Iteration': history['iteration'][i],
                    'Valid Ants': history['valid_count'][i],
                    'Invalid Ants': history['invalid_count'][i],
                    'Success Rate': f"{history['valid_count'][i]/NUM_ANTS*100:.0f}%",
                    'Avg Violation (kW)': f"{history['avg_violation'][i]:.2f}",
                    'Global Best (RM)': f"{history['global_best_cost'][i]:.2f}",  # Changed: Global cumulative best
                    'This Iter Best (RM)': f"{history['iter_best_cost'][i]:.2f}",  # New: This iteration's best
                    'This Iter Worst (RM)': f"{history['iter_worst_cost'][i]:.2f}" # Changed: This iteration's worst
                })
            
            iteration_df = pd.DataFrame(iteration_data)
           
            
            # Detailed Table (expandable)
            st.dataframe(iteration_df, use_container_width=True, height=400)
            
        else:
            st.warning("Data not available. run it brother")

