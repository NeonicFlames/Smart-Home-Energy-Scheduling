import pandas as pd

def generate_benchmark_from_raw():
    # 1. Load the Raw Data
    # Ensure this filename matches your uploaded file
    raw_file = 'smart_home_energy_consumption_large.csv'
    try:
        df = pd.read_csv(raw_file)
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {raw_file} not found. Please make sure the file is in the folder.")
        return

    # 2. Analyze Real Data to get Average Power Consumption
    # We group by appliance to get the real-world usage patterns from the dataset
    appliance_stats = df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().to_dict()
    print("Calculated appliance power profiles from raw data.")

    # 3. Define the Standardized Scenario (The "Benchmark Problem")
    # We use the 'Appliance Type' from the raw data to look up the power,
    # but we manually define the 'Time' and 'Duration' to create a challenging schedule.
    # This ensures all 4 team members are solving the EXACT same problem.
    
    scenario_tasks = [
        # NON-SHIFTABLE (Fixed Constraints - The "Must-Haves")
        # These appliances create the "Base Load"
        {'Appliance': 'Fridge',           'Preferred_Start': 0,  'Duration': 24, 'Shiftable': False},
        {'Appliance': 'Lights',           'Preferred_Start': 6,  'Duration': 2,  'Shiftable': False}, # Morning Routine
        {'Appliance': 'Lights',           'Preferred_Start': 18, 'Duration': 5,  'Shiftable': False}, # Evening Routine
        {'Appliance': 'TV',               'Preferred_Start': 19, 'Duration': 3,  'Shiftable': False}, # Prime Time
        {'Appliance': 'Microwave',        'Preferred_Start': 12, 'Duration': 1,  'Shiftable': False}, # Lunch
        {'Appliance': 'Microwave',        'Preferred_Start': 19, 'Duration': 1,  'Shiftable': False}, # Dinner
        {'Appliance': 'Air Conditioning', 'Preferred_Start': 13, 'Duration': 4,  'Shiftable': False}, # Afternoon Heat Peak
        
        # SHIFTABLE (Optimization Targets)
        # These are the tasks your ACO, GA, and PSO algorithms must optimize.
        # They have high power consumption and flexible times.
        {'Appliance': 'Washing Machine',  'Preferred_Start': 9,  'Duration': 2,  'Shiftable': True},  # Pref: 9am (Peak)
        {'Appliance': 'Dishwasher',       'Preferred_Start': 20, 'Duration': 2,  'Shiftable': True},  # Pref: 8pm (Peak)
        {'Appliance': 'Heater',           'Preferred_Start': 18, 'Duration': 4,  'Shiftable': True}   # Pref: 6pm (Peak)
    ]

    # 4. Build the DataFrame
    benchmark_data = []
    for i, task in enumerate(scenario_tasks, 1):
        appliance_name = task['Appliance']
        
        # Get real power from data (fallback to 0.5 if missing)
        avg_power = appliance_stats.get(appliance_name, 0.5)
        
        benchmark_data.append({
            'Task_ID': i,
            'Appliance': appliance_name,
            'Avg_Power_kW': round(avg_power, 2), # Rounded for cleaner data
            'Preferred_Start_Hour': task['Preferred_Start'],
            'Duration_Hours': task['Duration'],
            'Is_Shiftable': task['Shiftable']
        })

    # 5. Save to CSV
    output_filename = 'project_benchmark_data.csv'
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(output_filename, index=False)

    print(f"\nSUCCESS: '{output_filename}' has been generated.")
    print("-" * 50)
    print(benchmark_df.to_string())
    print("-" * 50)
    print("Distribute this file to your team members for their algorithms.")

if __name__ == "__main__":
    generate_benchmark_from_raw()