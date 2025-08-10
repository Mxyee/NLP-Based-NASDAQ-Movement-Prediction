import pandas as pd

news = pd.read_csv('data/processed/all_news_cleaned_sorted.csv')
market = pd.read_csv('data/processed/nasdaq_labels.csv')
print(news.columns)
# Ensure date columns are in datetime format
news['date'] = pd.to_datetime(news['date'])
market['Date'] = pd.to_datetime(market['Date'], dayfirst=True)

# Merge the two DataFrames on the date columns into a list
market_dates = sorted(market['Date'].unique().tolist())
news['market_date'] = news['date'].apply(lambda d: next(( md for md in market_dates if md >= d), pd.NaT))
# Align the news DataFrame with the market dates
def align_date(news_date):
    future_date = [d for d in market_dates if d >= news_date]
    return future_date[0] if future_date else None

news['merged_date'] = news['date'].apply(align_date)

merged = pd.merge(news, market, left_on = 'market_date', right_on = 'Date', how = 'inner')
merged = merged.drop(columns=['merged_date', 'date_raw', 'title','date'])
final = merged[['Date', 'title_clean', 'market_date', 'Open', 'Close', 'label']]
# Save final cleaned merged data
final.to_csv('data/processed/news_market_merged.csv', index=False)