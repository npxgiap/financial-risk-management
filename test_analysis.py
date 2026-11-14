import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['VCB', 'BID', 'CTG', 'HPG', 'GVR', 'VHM', 'MSN', 'VNM', 'SAB', 'GAS', 'POW', 'PLX']
yf_tickers = [f"{t}.VN" for t in tickers]

df = yf.download(yf_tickers, start='2023-07-31', end='2026-08-01', auto_adjust=True)['Close']
df.columns = [c.replace('.VN', '') for c in df.columns]
df = df[tickers] # keep original order

# Drop dates where all stocks are NaN
df_cleaned = df.dropna(how='all')

print(f"Total dates: {len(df)}")
print(f"Valid trading dates (at least 1 stock): {len(df_cleaned)}")
print(f"Common sample dates (all 12 stocks): {len(df_cleaned.dropna())}")
print("\nMissing values per ticker:")
print(df_cleaned.isnull().sum())

# Returns
returns = df_cleaned.pct_change()
log_returns = np.log(df_cleaned / df_cleaned.shift(1))

# Pairwise correlation
corr_pairwise = returns.corr(method='pearson')
print("\nPairwise Pearson correlation matrix:")
print(corr_pairwise.round(4))

# Find lowest correlation pair
pairs = []
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        t1, t2 = tickers[i], tickers[j]
        # Valid observations for pair
        valid_obs = returns[[t1, t2]].dropna()
        n_obs = len(valid_obs)
        p_corr = valid_obs[t1].corr(valid_obs[t2], method='pearson')
        s_corr = valid_obs[t1].corr(valid_obs[t2], method='spearman')
        pairs.append({
            'Stock 1': t1,
            'Stock 2': t2,
            'Pair': f"{t1} - {t2}",
            'Pearson_Corr': p_corr,
            'Spearman_Corr': s_corr,
            'Observations': n_obs
        })

df_pairs = pd.DataFrame(pairs).sort_values('Pearson_Corr').reset_index(drop=True)
df_pairs['Rank'] = df_pairs.index + 1

print("\nTop 5 Lowest Correlation Pairs:")
print(df_pairs[['Rank', 'Stock 1', 'Stock 2', 'Pearson_Corr', 'Spearman_Corr', 'Observations']].head(10))

print("\nLowest correlation pair overall:")
lowest_pair = df_pairs.iloc[0]
print(lowest_pair)

