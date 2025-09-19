import pandas as pd
import numpy as np
import sys
import os
import joblib

# import some libraries for training

def add_weather_features(df):
    t = df["Dry_Bulb_Temperature_C"]
    rad = df["Global_Horizontal_Radiation_W/m2"].clip(lower=0)
    df["HDD18"] = (18 - t).clip(lower=0)
    df["CDD22"] = (t - 22).clip(lower=0)
    df["TempC2"] = t**2
    df["rad_sqrt"] = np.sqrt(rad)
    df["rad_log1p"] = np.log1p(rad)
    df["is_daylight"] = (rad > 20).astype(int)
    # simple interactions
    df["TempC_x_daylight"] = t * df["is_daylight"]
    df["CDD22_x_daylight"] = df["CDD22"] * df["is_daylight"]
    df["TempC_x_hour_sin"] = t * df["Hour_sin"]
    df["TempC_x_hour_cos"] = t * df["Hour_cos"]
    return df


def phase1_preprocess_data(df_in):
    df = df_in.copy()
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp_Local'])
    # Extract datetime features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DOW'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Minute'] = df['Timestamp'].dt.minute
    # sin and cos transformation for cyclical features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DOW_sin'] = np.sin(2 * np.pi * df['DOW'] / 7)
    df['DOW_cos'] = np.cos(2 * np.pi * df['DOW'] / 7)

    # weather features
    df = add_weather_features(df)

    # Build seasonal features
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['Is_Summer'] = df['Month'].isin([5, 6, 7, 8]).astype(int)
    df['Is_Winter'] = df['Month'].isin([12, 1, 2, 3]).astype(int)
    # Create hour of day categories
    df['Is_Afternoon'] = df['Hour'].isin(range(12, 18)).astype(int)
    df['Is_Evening'] = df['Hour'].isin(range(18, 24)).astype(int)
    # drop unused columns
    df.drop(columns=['Timestamp_Local','Timestamp','Site','Demand_Response_Capacity_kW'], inplace=True)
    # Fix target variable (instead of -1 make it 2)
    df['Demand_Response_Flag'] = df['Demand_Response_Flag'].replace(-1, 2)
    return df


def phase2_preprocess_data(df_in):
    df = df_in.copy()
    # First, need to remove rows with Demand_Response_Flag = 0
    # df = df[df['Demand_Response_Flag'] != 0].copy()
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp_Local'])
    # Extract datetime features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DOW'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Minute'] = df['Timestamp'].dt.minute
    # # sin and cos transformation for cyclical features
    # df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    # df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    # df['DOW_sin'] = np.sin(2 * np.pi * df['DOW'] / 7)
    # df['DOW_cos'] = np.cos(2 * np.pi * df['DOW'] / 7)

    # # weather features
    # df = add_weather_features(df)

    # Build seasonal features
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['Is_Summer'] = df['Month'].isin([5, 6, 7, 8]).astype(int)
    df['Is_Winter'] = df['Month'].isin([12, 1, 2, 3]).astype(int)
    # # Create hour of day categories
    # df['Is_Afternoon'] = df['Hour'].isin(range(12, 18)).astype(int)
    # df['Is_Evening'] = df['Hour'].isin(range(18, 24)).astype(int)
    # drop unused columns
    # Note: since this is phase-2 training, we keep 'Demand_Response_Flag' column
    df.drop(columns=['Timestamp_Local','Timestamp','Site'], inplace=True)
    return df