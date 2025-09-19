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
    return df

def add_time_features(
    df,
    ts_col,
    tz="America/Kentucky/Louisville",
    id_col=None,                 # set to an ID column if you want per-entity interarrival
    fourier_orders={"day": 4, "week": 3, "year": 4},  # tune as needed
):
    out = df.copy()
    # 1) Parse timestamp → UTC → local tz
    ts = pd.to_datetime(out[ts_col], errors="coerce", utc=True).dt.tz_convert(tz)
    out["_ts_local"] = ts  # keep a tz-aware working column

    # 2) Calendar basics
    out["year"] = ts.dt.year
    out["quarter"] = ts.dt.quarter
    out["month"] = ts.dt.month
    out["week"] = ts.dt.isocalendar().week.astype(int)
    out["day"] = ts.dt.day
    out["dow"] = ts.dt.weekday            # Monday=0
    out["doy"] = ts.dt.dayofyear
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["second"] = ts.dt.second
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_month_start"] = ts.dt.is_month_start.astype(int)
    out["is_month_end"]   = ts.dt.is_month_end.astype(int)
    out["is_quarter_end"] = ts.dt.is_quarter_end.astype(int)
    out["is_year_end"]    = ts.dt.is_year_end.astype(int)

    # # 5) Cyclical encodings (sin/cos) for hour/day-of-week/day-of-year
    # # Fractions
    # sec_of_day = ts.dt.hour*3600 + ts.dt.minute*60 + ts.dt.second + ts.dt.microsecond/1e6
    # day_frac = sec_of_day / 86400.0
    # week_frac = ((out["dow"]*86400.0) + sec_of_day) / 604800.0
    # year_frac = (out["doy"] - 1 + day_frac) / 365.2425

    # for name, frac in [("day", day_frac), ("week", week_frac), ("year", year_frac)]:
    #     out[f"sin_{name}"] = np.sin(2*np.pi*frac)
    #     out[f"cos_{name}"] = np.cos(2*np.pi*frac)

    # # 6) Fourier series (richer seasonality)
    # def add_fourier(prefix, period_seconds, order, time_seconds):
    #     for k in range(1, order+1):
    #         angle = 2*np.pi*k * (time_seconds / period_seconds)
    #         out[f"{prefix}_sin_{k}"] = np.sin(angle)
    #         out[f"{prefix}_cos_{k}"] = np.cos(angle)

    # # Use epoch seconds (local wall-clock mapped onto a circle by period)
    # t_sec = (ts.view("int64") / 1e9).to_numpy()  # ns → s
    # if "day" in fourier_orders and fourier_orders["day"] > 0:
    #     add_fourier("fourier_day", 86400.0, fourier_orders["day"], t_sec)
    # if "week" in fourier_orders and fourier_orders["week"] > 0:
    #     add_fourier("fourier_week", 604800.0, fourier_orders["week"], t_sec)
    # if "year" in fourier_orders and fourier_orders["year"] > 0:
    #     add_fourier("fourier_year", 365.2425*86400.0, fourier_orders["year"], t_sec)
    out.drop(columns=["_ts_local"], inplace=True)
    return out

def add_leak_safe_power_features(
    df,
    ts_col="Timestamp_Local",
    pow_col="Building_Power_kW",
    freeze_minutes=15                   # don’t use info closer than this to T
):
    g = df.copy()
    g["ts"] = pd.to_datetime(g[ts_col], errors="coerce")
    g = g.sort_values("ts").reset_index(drop=True)

    # infer sampling step (seconds)
    step_sec = g["ts"].diff().dt.total_seconds().median()
    if pd.isna(step_sec) or step_sec <= 0: step_sec = 900.0  # fallback 15min
    to_steps = lambda minutes: max(1, int(round((minutes*60)/step_sec)))
    steps_1h  = to_steps(60)
    steps_3h  = to_steps(180)
    steps_6h  = to_steps(360)
    steps_24h = to_steps(1440)
    freeze_steps = to_steps(freeze_minutes)

    s = g[pow_col].astype(float)

    # Lags (at least freeze_steps into the past)
    g["P_lag_1"]   = s.shift(1 + (freeze_steps-1))
    g["P_lag_1h"]  = s.shift(steps_1h + (freeze_steps-1))
    g["P_lag_3h"]  = s.shift(steps_3h + (freeze_steps-1))
    g["P_lag_6h"]  = s.shift(steps_6h + (freeze_steps-1))
    g["P_lag_24h"] = s.shift(steps_24h + (freeze_steps-1))
    g.drop(columns=["ts"], inplace=True)
    return g


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
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp_Local'])
    # time features
    df = add_time_features(
        df,
        ts_col="Timestamp",
        tz="Australia/Sydney",        # ← Wollongong / NSW
        fourier_orders={"day": 6, "week": 4, "year": 6},
    )
    df = add_leak_safe_power_features(
        df,
        ts_col="Timestamp",
        pow_col="Building_Power_kW",
        freeze_minutes=60                   # don’t use info closer than this to T
    )
    # weather features
    # df = add_weather_features(df)
    # drop unused columns
    # Note: since this is phase-2 training, we keep 'Demand_Response_Flag' column
    df.drop(columns=['Timestamp_Local','Timestamp','Site','Building_Power_kW'], inplace=True)
    return df

def perform_one_hot_encode(df, col_name='Demand_Response_Flag'):
    # One-hot encode using 0/1 for each unique value in Demand_Response_Flag
    df[col_name] = df[col_name].astype(int)
    dummies = pd.get_dummies(df[col_name], prefix=col_name, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df