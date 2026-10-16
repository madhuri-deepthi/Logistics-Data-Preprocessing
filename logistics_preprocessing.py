import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create a sample logistics dataset
data = {
    "Shipment_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Distance_km": [120, 250, 180, 300, np.nan, 150, 220, 5000, 280, 200],
    "Delivery_Time_hrs": [4, 7, 5, np.nan, 6, 4, 8, 100, 7, 5],
    "Shipping_Cost": [500, 800, 650, 900, 700, np.nan, 750, 12000, 850, 600]
}

df = pd.DataFrame(data)

print("ORIGINAL LOGISTICS DATA")
print(df)

# Step 2: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Handle missing values using column medians
df["Distance_km"] = df["Distance_km"].fillna(df["Distance_km"].median())
df["Delivery_Time_hrs"] = df["Delivery_Time_hrs"].fillna(
    df["Delivery_Time_hrs"].median()
)
df["Shipping_Cost"] = df["Shipping_Cost"].fillna(
    df["Shipping_Cost"].median()
)

# Step 4: Detect potential outliers using the IQR method
for column in ["Distance_km", "Delivery_Time_hrs", "Shipping_Cost"]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) | (df[column] > upper_limit)
    ]

    print(f"\nPotential outliers in {column}:")
    print(outliers[["Shipment_ID", column]])

# Step 5: Normalize numerical columns using Min-Max scaling
scaler = MinMaxScaler()

columns_to_scale = [
    "Distance_km",
    "Delivery_Time_hrs",
    "Shipping_Cost"
]

df[["Distance_Normalized",
    "Delivery_Time_Normalized",
    "Shipping_Cost_Normalized"]] = scaler.fit_transform(
        df[columns_to_scale]
    )

# Step 6: Save the processed dataset
df.to_csv("logistics_preprocessed.csv", index=False)

print("\nCLEANED AND PREPROCESSED DATA")
print(df.round(3))

print("\nProcessed dataset saved as logistics_preprocessed.csv")