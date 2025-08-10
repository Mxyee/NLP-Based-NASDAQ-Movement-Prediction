import sys
import pandas as pd

output = 'demo/data/processed/nasdaq_labels.csv'

def fetch_ixic_yf():
    import yfinance as yf
    print("Fetching NASDAQ Composite Index data from Yahoo Finance...")
    ixic = yf.download('^IXIC', period='20y', interval='1d', auto_adjust=True, progress=False)
    ixic = ixic.dropna().rename_axis("Date").reset_index()
    ixic["label"] = (ixic["Close"] > ixic["Open"]).astype(int)
    ixic.to_csv(output, index=False)
    print("Saved:", output,"|rows:", len(ixic))

def main():
    try:
        fetch_ixic_yf()
    except Exception as e:
        print("yfinance download failed:", e)
        print("Please provide a callback CSV file if needed and write it to", output)
        sys.exit(1)

if "__main__" == __name__:
    main()
