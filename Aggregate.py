import os
import pandas as pd

battery_label = "G1"
base_dir = "Data/Output/LGM50/Optimization_Results"
cycle = 0

# Initialize combined DataFrame
combined_df = pd.DataFrame()

csv_path = os.path.join(base_dir, battery_label, str(cycle), f"{battery_label}_{cycle}_ecm_lut_table.csv")

if os.path.exists(csv_path):
    try:
        # Read CSV and ensure proper columns
        df = pd.read_csv(csv_path)

        # Add missing cycle column if not present
        if 'cycle' not in df.columns:
            df['cycle'] = cycle

        # Standardize column names (optional)
        df.columns = df.columns.str.lower()  # Ensure lowercase

        combined_df = df.copy()
        print(f"Successfully processed Cycle {cycle}")

    except Exception as e:
        print(f"Error processing Cycle {cycle}: {str(e)}")
else:
    print(f"File not found for Cycle {cycle}: {csv_path}")

# Sort by pulse number if available
if not combined_df.empty:
    if 'pulse_number' in combined_df.columns:
        combined_df.sort_values('pulse_number', inplace=True)

    # Save results
    output_path = os.path.join(base_dir, battery_label, str(cycle), f"{battery_label}_cycle_{cycle}_ecm_results.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"\nCycle {cycle} aggregation complete! Results saved to:\n{output_path}")

    # Show summary
    print("\nSummary Statistics:")
    print(combined_df.describe())

    print("\nFirst 5 rows:")
    print(combined_df.head())
else:
    print("No data was processed. Check file path or file format.")
