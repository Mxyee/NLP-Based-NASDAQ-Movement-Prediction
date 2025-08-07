import pandas as pd

# --- Process nasdaq_2025.csv ---
print("Processing nasdaq_2025.csv...")
df_2025 = pd.read_csv("data/raw/nasdaq_2025.csv")

# Ensure 'Close' and 'Open' are numeric types, remove commas and handle NaNs
df_2025['Close'] = df_2025['Close'].astype(str).str.replace(',', '', regex=False)
df_2025['Open'] = df_2025['Open'].astype(str).str.replace(',', '', regex=False)
df_2025['Close'] = pd.to_numeric(df_2025['Close'], errors='coerce')
df_2025['Open'] = pd.to_numeric(df_2025['Open'], errors='coerce')
df_2025 = df_2025.dropna(subset=['Close', 'Open'])  # Remove rows where conversion failed

# Add 'label' column (1 if Close > Open, else 0)
df_2025["label"] = (df_2025["Close"] > df_2025["Open"]).astype(int)
print("Finished processing nasdaq_2025.csv.")
print(df_2025.head())
print("-" * 30)

# --- Process nasdaq_2008-2016.csv ---
print("Processing nasdaq_2008-2016.csv...")
df_2008_2016 = pd.read_csv("data/raw/nasdaq_2008-2016.csv")

# Ensure 'Close' and 'Open' are numeric types, remove commas and handle NaNs
df_2008_2016['Close'] = df_2008_2016['Close'].astype(str).str.replace(',', '', regex=False)
df_2008_2016['Open'] = df_2008_2016['Open'].astype(str).str.replace(',', '', regex=False)
df_2008_2016['Close'] = pd.to_numeric(df_2008_2016['Close'], errors='coerce')
df_2008_2016['Open'] = pd.to_numeric(df_2008_2016['Open'], errors='coerce')
df_2008_2016 = df_2008_2016.dropna(subset=['Close', 'Open'])  # Remove rows where conversion failed

# Add 'label' column (1 if Close > Open, else 0)
df_2008_2016["label"] = (df_2008_2016["Close"] > df_2008_2016["Open"]).astype(int)
print("Finished processing nasdaq_2008-2016.csv.")
print(df_2008_2016.head())
print("-" * 30)

# --- Combine the two DataFrames ---
print("Combining processed data...")
# Make sure both DataFrames have the same column order to merge properly
# Assuming both CSV files have the same column structure
combined_df = pd.concat([df_2025, df_2008_2016], ignore_index=True)
print("Data combined successfully.")
print(f"Total number of rows after merge: {len(combined_df)}")
print(combined_df.head())
print(combined_df.tail())
print("-" * 30)

# --- Save the new CSV ---
# Save the combined data with labels into nasdaq_labels.csv
# This will contain NASDAQ data with labels for all years
print("Saving combined data to nasdaq_labels.csv...")
combined_df.to_csv("data/processed/nasdaq_labels.csv", index=False)
print("nasdaq_labels.csv saved successfully!")

