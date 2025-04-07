import os
import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Ensure cache directory exists
cache_dir = "f1_cache"
os.makedirs(cache_dir, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

st.title("🏎️ F1 Race Predictor")

# List of 2025 Grand Prix races
grand_prix_2025 = [
    "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix", "Bahrain Grand Prix", 
    "Saudi Arabian Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix", "Monaco Grand Prix", 
    "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix", 
    "Belgian Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", 
    "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", 
    "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
]
selected_gp = st.selectbox("Select the Grand Prix you want to predict:", grand_prix_2025)

train_year = 2024  # Default training year

# Function to fetch race and qualifying data
def get_race_data(year, gp):
    try:
        schedule = fastf1.get_event_schedule(year)
        race_event = schedule[schedule['EventName'].str.contains(gp, case=False, na=False)]
        if race_event.empty:
            return None, None
        round_number = race_event.iloc[0]['RoundNumber']
        race = fastf1.get_session(year, round_number, 'Race')
        race.load()
        quali = fastf1.get_session(year, round_number, 'Qualifying')
        quali.load()

        # Corrected: Use 'FullName' from race results, 'Driver' from qualifying data
        race_results = race.results[['DriverNumber', 'FullName', 'TeamName', 'Position', 'GridPosition', 'Points']]

        quali_results = quali.laps.groupby(["DriverNumber", "Driver"]).agg(
            avg_sector1=('Sector1Time', 'mean'),
            avg_sector2=('Sector2Time', 'mean'),
            avg_sector3=('Sector3Time', 'mean'),
            min_lap_time=('LapTime', 'min')  # Use Fastest lap time
        ).reset_index()

        # Merge race and qualifying data using DriverNumber
        quali_results.rename(columns={"Driver": "FullName"}, inplace=True)

        return race_results, quali_results

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

st.write("Fetching data...")
race_results, quali_results = get_race_data(train_year, selected_gp)
if race_results is None or quali_results is None:
    st.error("Failed to load data.")
    st.stop()

st.write("✅ Data Loaded Successfully")

# Display 2024 Qualifying & Race results side by side
col1, col2 = st.columns(2)
with col1:
    st.write(f"📊 **2024 Qualifying Results - {selected_gp}**")
    st.dataframe(quali_results[['FullName', 'min_lap_time']].rename(columns={'min_lap_time': 'Fastest Lap Time'}))
with col2:
    st.write(f"🏁 **2024 Race Results - {selected_gp}**")
    st.dataframe(race_results[['FullName', 'TeamName', 'Position']].rename(columns={'Position': 'Final Position'}))

# Adding new 2025 drivers
new_drivers = ["Andrea Kimi Antonelli", "Liam Lawson", "Isack Hadjar", "Franco Colapinto", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto"]
for driver in new_drivers:
    if driver not in race_results['FullName'].values:
        new_entry = pd.DataFrame([[None, driver, "Unknown Team", None, None, None]], columns=race_results.columns)
        race_results = pd.concat([race_results, new_entry], ignore_index=True)

# Preprocessing
df = pd.merge(race_results, quali_results, on=['DriverNumber', 'FullName'], how='left')
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df['Points'] = pd.to_numeric(df['Points'], errors='coerce')

df['FullName'] = df['FullName'].fillna("Unknown Driver")
df['TeamName'] = df['TeamName'].fillna("Unknown Team")

# Ensure LabelEncoder sees all possible labels
combined_teams = np.unique(pd.concat([df['TeamName'], race_results['TeamName']]).fillna("Unknown Team"))
combined_drivers = np.unique(pd.concat([df['FullName'], race_results['FullName']]).fillna("Unknown Driver"))

driver_encoder = LabelEncoder()
team_encoder = LabelEncoder()
driver_encoder.fit(combined_drivers)
team_encoder.fit(combined_teams)

df['DriverID'] = driver_encoder.transform(df['FullName'])
df['TeamID'] = team_encoder.transform(df['TeamName'])

for col in ['avg_sector1', 'avg_sector2', 'avg_sector3', 'min_lap_time']:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col])
        df[col] = df[col].dt.total_seconds().fillna(0)

df = df.dropna()
features = ['GridPosition', 'TeamID', 'DriverID', 'avg_sector1', 'avg_sector2', 'avg_sector3', 'min_lap_time']
target = 'Position'
X = df[features]
y = df[target] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"📊 Model Trained! Mean Absolute Error: {mae:.2f}")

# Predicting 2025 Race Results
st.write(f"🔮 Predicting 2025 Results for {selected_gp}...")
latest_race, latest_quali = get_race_data(2025, selected_gp)
if latest_race is not None and latest_quali is not None:
    latest_data = pd.merge(latest_race, latest_quali, on=['DriverNumber', 'FullName'], how='left')

    st.write(f"📊 **2025 Qualifying Times - {selected_gp}**")
    st.dataframe(latest_data[['FullName', 'min_lap_time']].rename(columns={'min_lap_time': 'Fastest Lap Time'}))

    latest_data['DriverID'] = driver_encoder.transform(latest_data['FullName'])
    latest_data['TeamID'] = team_encoder.transform(latest_data['TeamName'])

    X_predict = latest_data[features].dropna()
    latest_data['PredictedPosition'] = model.predict(X_predict)
    latest_data = latest_data.sort_values(by='PredictedPosition')

    st.write(f"🏁 **Predicted Race Results for 2025 {selected_gp}**")
    st.dataframe(latest_data[['FullName', 'TeamName', 'PredictedPosition']].rename(columns={'PredictedPosition': 'Predicted Position'}))
