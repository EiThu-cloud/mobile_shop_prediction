import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load features and model
features = joblib.load("unit_model_features.pkl")
model = joblib.load("unit_sold_model.pkl")

st.set_page_config(page_title="Mobile Shop Sales Predictor", layout="centered")

# Title
st.title("📱 Mobile Shop - Predict Units Sold")

# Top 10 mobile models bar chart
st.header("Top 10 Mobile Models by Units Sold")

top10_data = {
    "Mobile_Model": [
        "OnePlus Nord 4", "OnePlus 12 Pro", "OnePlus 11R", "Pixel 9 Pro", "Pixel 9",
        "Galaxy S25 Ultra", "iPhone 14", "iPhone 15", "Z Fold 6", "Redmi Note 13"
    ],
    "Units_Sold": [1485, 1480, 1430, 1203, 1191, 1030, 989, 951, 912, 896]
}
df_top10 = pd.DataFrame(top10_data)

plt.figure(figsize=(10, 6))
sns.barplot(x="Units_Sold", y="Mobile_Model", data=df_top10, palette="viridis")
plt.xlabel("Units Sold")
plt.ylabel("Mobile Model")
plt.title("Top 10 Mobile Models by Units Sold")
plt.tight_layout()
st.pyplot(plt)
plt.clf()

# Prediction section
st.header("Predict Units Sold")

# Inputs
storage_size = st.selectbox("Storage Size", ["128GB", "256GB"])
price = st.number_input("Price ($)", min_value=0, max_value=3000, value=500)

mobile_models = [
    "Galaxy M15", "Galaxy S25 Ultra", "Mi 13T Pro", "OnePlus 11R", "OnePlus 12 Pro",
    "OnePlus Nord 4", "Pixel 8a", "Pixel 9", "Pixel 9 Pro", "Poco X6 Pro",
    "Redmi Note 13", "Xiaomi 14 Ultra", "Z Fold 6"
]
selected_model = st.selectbox("Mobile Model", mobile_models)

cities = ["Lahore", "Multan", "Chittagong", "Bursa", "Rajshahi", "Antalya"]
selected_city = st.selectbox("City", cities)

# Convert Storage Size to numeric
storage_map = {"128GB": 128, "256GB": 256}
storage_num = storage_map[storage_size]

# One-hot encode mobile models
mobile_model_features = {f"Mobile_Model_{model}": 0 for model in mobile_models}
mobile_model_features[f"Mobile_Model_{selected_model}"] = 1

# One-hot encode cities
city_features = {f"City_{city}": 0 for city in cities}
city_features[f"City_{selected_city}"] = 1

# Prepare input dictionary
input_data = {
    "Storage_Size": storage_num,
    "Price": price,
}
input_data.update(mobile_model_features)
input_data.update(city_features)

# Create DataFrame in correct feature order
input_df = pd.DataFrame([input_data], columns=features)

# Predict button
if st.button("🔍 Predict Units Sold"):
    prediction = model.predict(input_df)[0]
    st.success(f"📦 Predicted Units Sold: **{int(round(prediction))}**")