# aqi_prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# import os # જો ડેટાફાઇલ પાથ હેન્ડલિંગ કરવું હોય તો, અત્યારે જરૂર નથી
# import requests # જો API નો ઉપયોગ કરવો હોય તો, અત્યારે જરૂર નથી
# import json # જો API નો ઉપયોગ કરવો હોય તો, અત્યારે જરૂર નથી

print("AQI Prediction Project Started...")

# --------------------------------------------------------------------------------
# પગલું 2: વાસ્તવિક AQI ડેટા લોડ કરવામાં આવી રહ્યો છે (તાલીમ માટે)
# --------------------------------------------------------------------------------
print("\nપગલું 2: ઐતિહાસિક AQI ડેટા (તાલીમ માટે) લોડ કરવામાં આવી રહ્યો છે...")

# ડેટાફાઇલનું નામ અને પાથ
# ખાતરી કરો કે city_day.csv ફાઇલ તમારા પ્રોજેક્ટ ફોલ્ડરમાં અથવા './archive/' માં છે.
DATA_FILE = './archive/city_day.csv' # જો ફાઇલ સીધી પ્રોજેક્ટ ફોલ્ડરમાં હોય તો
# DATA_FILE = './archive/city_day.csv' # જો ફાઇલ archive ફોલ્ડરમાં હોય તો (તમારા અગાઉના આઉટપુટ મુજબ)

try:
    data = pd.read_csv(DATA_FILE)
    print(f"'{DATA_FILE}' માંથી ડેટા સફળતાપૂર્વક લોડ થયો.")
except FileNotFoundError:
    print(f"ભૂલ: '{DATA_FILE}' ફાઇલ મળી નથી. કૃપા કરીને ખાતરી કરો કે ફાઇલ પ્રોજેક્ટ ફોલ્ડરમાં છે.")
    print("તમારે Kaggle પરથી 'Air Quality in India' ડેટાસેટ ડાઉનલોડ કરીને 'city_day.csv' ફાઇલને પ્રોજેક્ટ ફોલ્ડરમાં અથવા તેના 'archive' સબ-ફોલ્ડરમાં મૂકવી પડશે.")
    exit() # જો ફાઇલ ન મળે તો પ્રોગ્રામ બંધ કરો

print("\nપ્રથમ 5 ડેટા એન્ટ્રીઓ:")
print(data.head())

print("\nડેટા સેટની માહિતી:")
print(data.info())

print("\nડેટા સેટનું વર્ણન:")
print(data.describe())

print("\nખૂટતા મૂલ્યો (Missing Values) તપાસી રહ્યા છીએ:")
print(data.isnull().sum())

# બિનજરૂરી કોલમ છોડી દેવા
# આપણે અહીં City, Date, AQI_Bucket ને ડ્રોપ કરી રહ્યા છીએ
data = data.drop(columns=['AQI_Bucket', 'City', 'Date'], errors='ignore')

# ખૂટતા મૂલ્યોને ફીચર્સના સરેરાશ (mean) વડે ભરો
# આ લૂપ બધા ન્યુમેરિક કૉલમ્સમાંના NaN મૂલ્યોને ભરશે.
for col in data.columns:
    if data[col].dtype in ['float64', 'int64'] and data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mean())

print("\nખૂટતા મૂલ્યો ભર્યા પછી (NaNs ભરાઈ ગયા પછી):")
print(data.isnull().sum()) # હવે અહીં કોઈ ન્યુમેરિક કોલમમાં NaN ન હોવા જોઈએ

# ખાતરી કરો કે AQI 0 થી 500 ની રેન્જમાં છે
data['AQI'] = np.clip(data['AQI'], 0, 500)

# --------------------------------------------------------------------------------
# પગલું 3: ડેટા પ્રીપ્રોસેસિંગ
# --------------------------------------------------------------------------------
print("\nપગલું 3: ડેટા પ્રીપ્રોસેસિંગ...")

# લક્ષણો (X) અને ટાર્ગેટ (y) ને અલગ કરો
X = data.drop('AQI', axis=1) # AQI સિવાયના બધા કોલમ લક્ષણો છે
y = data['AQI']

# X_train ના કૉલમ નામ સ્ટોર કરો. ભવિષ્યવાણી માટે આ જ ફીચર્સનો ઉપયોગ થશે.
X_feature_names_for_prediction = X.columns.tolist()

# તાલીમ અને પરીક્ષણ સેટમાં ડેટાને વિભાજિત કરો
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train આકાર: {X_train.shape}")
print(f"X_test આકાર: {X_test.shape}")
print(f"y_train આકાર: {y_train.shape}")
print(f"y_test આકાર: {y_test.shape}")

# ડેટા સ્કેલિંગ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nપ્રથમ 5 સ્કેલ્ડ તાલીમ ડેટા (X_train_scaled) એન્ટ્રીઓ:")
print(X_train_scaled[:5])

# --------------------------------------------------------------------------------
# પગલું 4: મોડેલ પસંદગી (RandomForestRegressor)
# --------------------------------------------------------------------------------
print("\nપગલું 4: મોડેલ પસંદગી (RandomForestRegressor)...")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("પસંદ કરેલ મોડેલ: RandomForestRegressor")

# --------------------------------------------------------------------------------
# પગલું 5: મોડેલ તાલીમ
# --------------------------------------------------------------------------------
print("\nપગલું 5: મોડેલને તાલીમ આપવામાં આવી રહી છે...")
model.fit(X_train_scaled, y_train)
print("મોડેલ તાલીમ પૂર્ણ.")

# --------------------------------------------------------------------------------
# પગલું 6: મોડેલ મૂલ્યાંકન
# --------------------------------------------------------------------------------
print("\nપગલું 6: મોડેલ મૂલ્યાંકન...")

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nમોડેલ મૂલ્યાંકન પરિણામો:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("વાસ્તવિક AQI")
plt.ylabel("અનુમાનિત AQI")
plt.title("વાસ્તવિક વિરુદ્ધ અનુમાનિત AQI")
plt.grid(True)
plt.show()

feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('લક્ષણનું મહત્વ (Feature Importance) - AQI Prediction')
plt.xlabel('મહત્વ')
plt.ylabel('લક્ષણ')
plt.show()

# --------------------------------------------------------------------------------
# પગલું 7: નવા ડેટા માટે ભવિષ્યવાણી (મેન્યુઅલ ડેટા સાથે)
# --------------------------------------------------------------------------------
print("\nપગલું 7: નવા ડેટા માટે ભવિષ્યવાણી...")

# નવા ડેટા માટે DataFrame બનાવો.
# મહત્વપૂર્ણ: આ DataFrame માં ફક્ત તે જ કૉલમ્સ હોવા જોઈએ જે X_train માં હતા, અને તે જ ક્રમમાં.
# city_day.csv માં Temperature, Humidity, Wind_Speed કૉલમ નથી, તેથી તેમને અહીંથી દૂર કરવામાં આવ્યા છે.
# અને city_day.csv માં NO, NOx, NH3, Benzene, Toluene, Xylene કૉલમ છે, તેથી તેમને અહીં શામેલ કરવામાં આવ્યા છે.

# અહીં આપણે તાલીમ ડેટામાંથી સરેરાશ મૂલ્યોનો ઉપયોગ નવા ડેટા માટે એક ઉદાહરણ તરીકે કરી રહ્યા છીએ.
# વાસ્તવિક એપ્લિકેશનમાં, તમે આ મૂલ્યોને ક્યાંકથી મેળવશો (દા.ત., સેન્સર્સ, મેન્યુઅલ ઇનપુટ).
example_new_data = {
    'PM2.5': [45.0, 120.0, 25.0],
    'PM10': [70.0, 180.0, 40.0],
    'NO': [X['NO'].mean(), 50.0, 10.0], # ઉદાહરણ તરીકે સરેરાશ અને કેટલાક કસ્ટમ મૂલ્યો
    'NO2': [30.0, 90.0, 18.0],
    'NOx': [X['NOx'].mean(), 80.0, 25.0],
    'NH3': [X['NH3'].mean(), 15.0, 5.0],
    'CO': [2.5, 7.0, 1.2],
    'SO2': [15.0, 60.0, 8.0],
    'O3': [50.0, 150.0, 35.0],
    'Benzene': [X['Benzene'].mean(), 10.0, 1.5],
    'Toluene': [X['Toluene'].mean(), 20.0, 3.0],
    'Xylene': [X['Xylene'].mean(), 5.0, 0.5]
}

# ખાતરી કરો કે કૉલમ્સનો ક્રમ X_train ના કૉલમ્સના ક્રમ સાથે મેળ ખાય છે.
new_air_data = pd.DataFrame(example_new_data, columns=X_feature_names_for_prediction)

print("\nનવા ડેટા:")
print(new_air_data)

# નવા ડેટાને પણ તાલીમ ડેટાની જેમ જ સ્કેલ કરવો પડશે
# હવે આ સ્ટેપ ભૂલ આપશે નહીં કારણ કે ફીચરના નામો મેચ થાય છે.
new_air_data_scaled = scaler.transform(new_air_data)

# નવા ડેટા પર ભવિષ્યવાણી કરો
new_aqi_predictions = model.predict(new_air_data_scaled)

print("\nનવા ડેટા માટે અનુમાનિત Air Quality Index (AQI):")
for i, pred_aqi in enumerate(new_aqi_predictions):
    print(f"નમૂના {i+1}: AQI = {pred_aqi:.2f}")

# કયા AQI કેટેગરીમાં આવે છે તે બતાવવા માટે (વૈકલ્પિક)
def get_aqi_category(aqi_value):
    if 0 <= aqi_value <= 50:
        return "Good"
    elif 51 <= aqi_value <= 100:
        return "Satisfactory"
    elif 101 <= aqi_value <= 200:
        return "Moderately Polluted"
    elif 201 <= aqi_value <= 300:
        return "Poor"
    elif 301 <= aqi_value <= 400:
        return "Very Poor"
    elif 401 <= aqi_value <= 500:
        return "Severe"
    else:
        return "Beyond Index"

print("\nઅનુમાનિત AQI કેટેગરીઝ:")
for i, pred_aqi in enumerate(new_aqi_predictions):
    category = get_aqi_category(pred_aqi)
    print(f"નમૂના {i+1}: AQI = {pred_aqi:.2f} ({category})")

print("\nપ્રોજેક્ટ પૂર્ણ!")