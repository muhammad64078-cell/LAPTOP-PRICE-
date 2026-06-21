import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import time

# 1. Dummy Dataset Create Karna (Agar aapke paas asli CSV na ho toh chal sake)
# Asli project mein aap likhenge: df = pd.read_csv('laptop_data.csv')
np.random.seed(42)
data = {
    'Ram': np.random.choice([4, 8, 16, 32], size=600),
    'Storage': np.random.choice([128, 256, 512, 1024], size=600),
    'Screen_Size': np.random.uniform(13.3, 17.3, size=600),
    'Price': []
}

# Price generate karne ka formula (taake models seekh sakein)
for i in range(600):
    base_price = (data['Ram'][i] * 3000) + (data['Storage'][i] * 50) + (data['Screen_Size'][i] * 1000)
    data['Price'].append(base_price + np.random.normal(0, 5000))

df = pd.DataFrame(data)

# 2. X (Features) aur Y (Target) Split
X = df[['Ram', 'Storage', 'Screen_Size']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model 1: Linear Regression Train Karna
start_time = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_time = time.time() - start_time

lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

# 4. Model 2: Random Forest Train Karna
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# 5. Saara Metrics Data Save Karna Dashboard ke liye
metrics = {
    'lr_mae': lr_mae, 'lr_r2': lr_r2, 'lr_time': lr_time,
    'rf_mae': rf_mae, 'rf_r2': rf_r2, 'rf_time': rf_time
}

# 6. Files ko Save (Pickle) Karna
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("Models aur Metrics kamyabi se save ho chuke hain!")