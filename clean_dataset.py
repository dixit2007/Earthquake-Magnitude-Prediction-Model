import pandas as pd
import os

# 1. Define filenames
input_file = 'earthquake1826_2026.csv'
output_file = 'earthquake_clean.csv'

# 2. Check if file exists
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' was not found.")
else:
    print("Reading dataset...")
    df = pd.read_csv(input_file)

    # 3. Define the VIP columns (The only ones we want to keep)
    # We select these 4 because they are the physical drivers of the model.
    cols_to_keep = ['latitude', 'longitude', 'depth', 'mag']

    # 4. Create a new dataframe with ONLY these columns
    df_clean = df[cols_to_keep]

    # 5. Remove any empty rows (Data Cleaning)
    # This removes rows where 'mag' or 'depth' might be blank
    initial_count = len(df)
    df_clean = df_clean.dropna()
    final_count = len(df_clean)

    # 6. Save the new clean file
    df_clean.to_csv(output_file, index=False)

    print(f"✅ Success! Clean file saved as: {output_file}")
    print(f"Original Rows: {initial_count}")
    print(f"Clean Rows:    {final_count}")
    print(f"Removed Rows:  {initial_count - final_count} (empty/bad data)")