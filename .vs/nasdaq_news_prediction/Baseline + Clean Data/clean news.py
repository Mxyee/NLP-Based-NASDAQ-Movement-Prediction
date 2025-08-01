import pandas as pd
from dateutil import parser

# Input and output paths
input_file = "all_raw_news.csv"
output_file = "all_news_cleaned_sorted.csv"

# Read the input CSV file
df = pd.read_csv(input_file)

# Back up the original date column
df['date_raw'] = df['date']

# Try to fix the date field format: use dateutil.parser to handle non-standard formats
def parse_date_safe(date_str):
    try:
        return parser.parse(str(date_str))
    except:
        return pd.NaT

df['date'] = df['date_raw'].apply(parse_date_safe)

# Print which fields were originally problematic
invalid_dates = df[df['date'].isna()]
print(f"Number of dates that could not be parsed:{len(invalid_dates)}")

# Clean up titles (remove - CNBC etc)
# Remove leading b' or b" and trailing news sources like " - CNBC"
df['title_clean'] = df['title'].str.replace(r"^b[\"']", "", regex=True).str.replace(r"\s*-\s*[A-Za-z\s]+$", "", regex=True)

# Filter out empty or short titles
df = df[df['title_clean'].str.len() > 15]

# Remove duplicate titles



# Sort by date (NaT comes last)
df = df.sort_values(by='date').reset_index(drop=True)

# Store the result
df.to_csv(output_file, index=False)
print(f" Cleaning completed: {len(df)} records in total, output to {output_file}")
