import os
import fastf1
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Ensure cache directory exists
cache_dir = "f1_cache"
os.makedirs(cache_dir, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

# Function to fetch race results and qualifying data
def get_race_data(year, gp="Australian Grand Prix"):
    try:
        schedule = fastf1.get_event_schedule(year)
        race_event = schedule[schedule['EventName'].str.contains(gp, case=False, na=False)]

        if race_event.empty:
            print(f"No {gp} found in {year}. Skipping...")
            return None, None

        round_number = race_event.iloc[0]['RoundNumber']
        race = fastf1.get_session(year, round_number, 'Race')
        race.load()
        quali = fastf1.get_session(year, round_number, 'Qualifying')
        quali.load()

        # Extract race results
        race_results = race.results[['DriverNumber', 'FullName', 'TeamName', 'Position', 'GridPosition', 'Points']]
        race_results['Year'] = year

        # Extract Qualifying Times & Sector Data
        quali_results = quali.laps.groupby("DriverNumber").agg(
            avg_sector1=('Sector1Time', 'mean'),
            avg_sector2=('Sector2Time', 'mean'),
            avg_sector3=('Sector3Time', 'mean'),
            avg_lap_time=('LapTime', 'mean')
        ).reset_index()

        return race_results, quali_results

    except Exception as e:
        print(f"Error fetching data for {gp} in {year}: {str(e)}")
        return None, None

# Function to calculate team performance trends
def get_team_performance(year):
    season_races = [get_race_data(year, gp) for gp in fastf1.get_event_schedule(year)['EventName']]
    race_results = [r[0] for r in season_races if r[0] is not None]
    
    df_season = pd.concat(race_results, ignore_index=True)
    df_team_perf = df_season.groupby('TeamName').agg(
        early_avg=('Position', lambda x: x[:5].mean()),
        late_avg=('Position', lambda x: x[-5:].mean())
    ).reset_index()
    df_team_perf['TeamPerformanceScore'] = df_team_perf['early_avg'] / df_team_perf['late_avg']
    return df_team_perf[['TeamName', 'TeamPerformanceScore']]

# Use only 2024 for training
train_year = 2024
race_results, quali_results = get_race_data(train_year)
team_perf = get_team_performance(train_year)

if race_results is None or quali_results is None:
    print("❌ Failed to load 2024 data. Cannot proceed with training.")
    exit()

# Merge race results, qualifying, and team performance data
df = pd.merge(race_results, quali_results, on=['DriverNumber'], how='left')
df = pd.merge(df, team_perf, on='TeamName', how='left')

df['PrevPosition'] = df['Position']
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
df['PrevPosition'] = pd.to_numeric(df['PrevPosition'], errors='coerce')

driver_encoder = LabelEncoder()
team_encoder = LabelEncoder()
df['DriverID'] = driver_encoder.fit_transform(df['FullName'])
df['TeamID'] = team_encoder.fit_transform(df['TeamName'])

time_columns = ['avg_sector1', 'avg_sector2', 'avg_sector3', 'avg_lap_time']
for col in time_columns:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col])
        df[col] = df[col].dt.total_seconds().fillna(0)

df = df.dropna()

features = ['GridPosition', 'PrevPosition', 'TeamID', 'DriverID', 'TeamPerformanceScore', 'avg_sector1', 'avg_sector2', 'avg_sector3', 'avg_lap_time']
target = 'Position'
df[target] = df[target] - 1

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

latest_race, latest_quali = get_race_data(2024)
latest_team_perf = get_team_performance(2024)

if latest_race is not None and latest_quali is not None:
    latest_data = pd.merge(latest_race, latest_quali, on=['DriverNumber'], how='left')
    latest_data = pd.merge(latest_data, latest_team_perf, on='TeamName', how='left')

    latest_data['PrevPosition'] = latest_data.get('Position', latest_data['GridPosition'])

    for col in time_columns:
        if col in latest_data.columns:
            latest_data[col] = pd.to_timedelta(latest_data[col])
            latest_data[col] = latest_data[col].dt.total_seconds().fillna(0)

    latest_data['DriverID'] = driver_encoder.transform(latest_data['FullName'])
    latest_data['TeamID'] = team_encoder.transform(latest_data['TeamName'])

    X_2025 = latest_data[features].dropna()
    predicted_positions = model.predict(X_2025)
    latest_data['PredictedPosition'] = predicted_positions
    latest_data = latest_data.sort_values(by='PredictedPosition')

    print("\n🏁 **Predicted Race Results for 2025 Australian Grand Prix** 🏁")
    print(f"{'Position':<10} {'Driver':<20} {'Team':<20} {'Avg Lap Time (s)'}")
    print("="*65)
    for index, row in latest_data.iterrows():
        avg_lap_time = f"{row['avg_lap_time']:.3f}" if pd.notna(row['avg_lap_time']) else "N/A"
        print(f"{int(row['PredictedPosition'] + 1):<10} {row['FullName']:<20} {row['TeamName']:<20} {avg_lap_time}")

    print(f"\n🏆 Predicted Winner of the 2025 Australian Grand Prix: {latest_data.iloc[0]['FullName']} 🏆")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("\n📊 **Model Evaluation:**")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
