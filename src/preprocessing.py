import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """Handles encoding and creates frequency-based features."""
    data = df.copy()
    
    # 1. Map flight days to numerical
    day_mapping = {
        "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, 
        "Fri": 5, "Sat": 6, "Sun": 7
    }
    data["flight_day"] = data["flight_day"].map(day_mapping)
    
    # 2. Frequency Encoding for high cardinality
    for col in ["route", "booking_origin"]:
        freq_encoding = data[col].value_counts(normalize=True)
        data[col + "_freq"] = data[col].map(freq_encoding)
    
    # 3. Drop original high cardinality columns
    data = data.drop(["route", "booking_origin"], axis=1)
    
    # 4. One-Hot Encoding for low cardinality
    categorical_cols = ["sales_channel", "trip_type"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data

def prepare_xy(data, target_col="booking_complete"):
    """Splits target from features and scales the data."""
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler