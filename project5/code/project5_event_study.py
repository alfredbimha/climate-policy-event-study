"""
===============================================================================
PROJECT 5: Climate Policy Event Study
===============================================================================
RESEARCH QUESTION:
    How do major climate policy announcements affect green and brown stock returns?
METHOD:
    Event study methodology — estimate abnormal returns (AR) and 
    cumulative abnormal returns (CAR) around key climate policy dates.
DATA:
    Yahoo Finance for stock returns, hand-collected policy event dates
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 1: Define climate policy events
# =============================================================================
print("STEP 1: Defining climate policy events...")
events = pd.DataFrame([
    {'date':'2021-01-20','event':'Biden Inauguration / Paris Rejoining','direction':'green_positive'},
    {'date':'2021-04-22','event':'Biden Climate Summit 2030 Target','direction':'green_positive'},
    {'date':'2021-11-01','event':'COP26 Glasgow Opens','direction':'green_positive'},
    {'date':'2022-08-16','event':'Inflation Reduction Act Signed','direction':'green_positive'},
    {'date':'2022-06-30','event':'SCOTUS West Virginia v EPA','direction':'brown_positive'},
    {'date':'2023-03-20','event':'IPCC AR6 Synthesis Report','direction':'green_positive'},
    {'date':'2023-09-20','event':'Biden UN Climate Ambition Summit','direction':'green_positive'},
    {'date':'2024-01-26','event':'Biden LNG Export Pause','direction':'green_positive'},
    {'date':'2024-03-06','event':'SEC Climate Disclosure Rule','direction':'green_positive'},
])
events['date'] = pd.to_datetime(events['date'])
events.to_csv('data/climate_events.csv', index=False)
print(f"  {len(events)} climate policy events defined")

# =============================================================================
# STEP 2: Download stock data
# =============================================================================
print("\nSTEP 2: Downloading stock data...")
assets = {'ICLN':'Clean Energy','XLE':'Fossil Fuels','SPY':'Market'}
prices = {}
for t in assets:
    df = yf.download(t, start='2020-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    prices[t] = df['Close']
    print(f"  {t}: {len(df)} obs")

prices = pd.DataFrame(prices).dropna()
returns = np.log(prices/prices.shift(1)).dropna() * 100
returns.to_csv('data/daily_returns.csv')

# =============================================================================
# STEP 3: Event study — market model estimation
# =============================================================================
print("\nSTEP 3: Running event study analysis...")

estimation_window = 120  # 120 trading days before event window
event_window = (-5, 10)   # 5 days before to 10 days after event

all_car = []
for _, ev in events.iterrows():
    event_date = ev['date']
    # Find nearest trading day
    idx = returns.index.searchsorted(event_date)
    if idx >= len(returns.index) or idx < estimation_window + abs(event_window[0]):
        continue
    actual_date = returns.index[idx]
    
    for asset in ['ICLN','XLE']:
        # Estimation window
        est_start = idx - estimation_window - abs(event_window[0])
        est_end = idx + event_window[0]
        
        y_est = returns[asset].iloc[est_start:est_end]
        x_est = add_constant(returns['SPY'].iloc[est_start:est_end])
        
        try:
            mkt_model = OLS(y_est, x_est).fit()
        except:
            continue
        
        # Event window
        ev_start = idx + event_window[0]
        ev_end = min(idx + event_window[1] + 1, len(returns))
        
        actual_ret = returns[asset].iloc[ev_start:ev_end]
        expected_ret = mkt_model.predict(add_constant(returns['SPY'].iloc[ev_start:ev_end]))
        
        ar = actual_ret - expected_ret  # Abnormal returns
        car = ar.cumsum()  # Cumulative abnormal returns
        
        days = list(range(event_window[0], event_window[0] + len(ar)))
        
        for d, a, c in zip(days, ar, car):
            all_car.append({
                'event': ev['event'], 'event_date': str(event_date.date()),
                'asset': asset, 'asset_name': assets[asset],
                'day': d, 'AR': a, 'CAR': c, 'direction': ev['direction']
            })

car_df = pd.DataFrame(all_car)
car_df.to_csv('output/tables/event_study_results.csv', index=False)

# =============================================================================
# STEP 4: Summary statistics
# =============================================================================
print("\nSTEP 4: Computing summary statistics...")
summary = car_df.groupby(['asset_name','day']).agg(
    mean_AR=('AR','mean'), mean_CAR=('CAR','mean'),
    std_AR=('AR','std'), n=('AR','count')
).reset_index()
summary['t_stat'] = summary['mean_AR'] / (summary['std_AR'] / np.sqrt(summary['n']))
summary['significant'] = abs(summary['t_stat']) > 1.96

# Print key results
for asset in ['Clean Energy','Fossil Fuels']:
    sub = summary[summary['asset_name']==asset]
    day0 = sub[sub['day']==0]
    if not day0.empty:
        print(f"  {asset} Day 0: AR={day0['mean_AR'].values[0]:.4f}% "
              f"(t={day0['t_stat'].values[0]:.2f})")
    day5 = sub[sub['day']==5]
    if not day5.empty:
        print(f"  {asset} CAR(0,5): {day5['mean_CAR'].values[0]:.4f}%")

summary.to_csv('output/tables/car_summary.csv', index=False)

# =============================================================================
# STEP 5: Visualizations
# =============================================================================
print("\nSTEP 5: Creating visualizations...")

# Fig 1: Average CAR for green vs brown
fig, ax = plt.subplots(figsize=(12, 6))
for asset, color in [('Clean Energy','#2ecc71'),('Fossil Fuels','#e74c3c')]:
    sub = summary[summary['asset_name']==asset]
    ax.plot(sub['day'], sub['mean_CAR'], label=asset, color=color, linewidth=2)
    ax.fill_between(sub['day'], sub['mean_CAR']-1.96*sub['std_AR']/np.sqrt(sub['n']),
                    sub['mean_CAR']+1.96*sub['std_AR']/np.sqrt(sub['n']),
                    alpha=0.2, color=color)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Event Day')
ax.set_title('Average Cumulative Abnormal Returns Around Climate Policy Events', fontweight='bold')
ax.set_xlabel('Days Relative to Event')
ax.set_ylabel('CAR (%)')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig1_average_car.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Event-by-event CAR at day +5
fig, ax = plt.subplots(figsize=(14, 7))
car5 = car_df[car_df['day']==5][['event','asset_name','CAR']].pivot(
    index='event', columns='asset_name', values='CAR')
if not car5.empty:
    car5.plot(kind='barh', ax=ax, color=['#2ecc71','#e74c3c'])
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title('CAR(0,+5) by Event', fontweight='bold')
    ax.set_xlabel('Cumulative Abnormal Return (%)')
plt.tight_layout()
plt.savefig('output/figures/fig2_event_car.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
