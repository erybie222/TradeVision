import pandas as pd
import os
from ta import add_all_ta_features

def prepare_features(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            ticker = filename.split('.')[0]
            df = pd.read_csv(os.path.join(input_path, filename), index_col=0)
            
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Dodaj wskaźniki TA
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )

            # --- Cechy klasyczne ---
            df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
            df["Volatility_10"] = df["Close"].rolling(10).std()
            df["ROC_10"] = df["Close"].pct_change(periods=10)

            sma20 = df["Close"].rolling(20).mean()
            std20 = df["Close"].rolling(20).std()
            df["BB_Width"] = (2 * std20) / sma20

            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            df["ATR_14"] = tr.rolling(14).mean()

            # --- Cechy niestandardowe ---
            df["Cumulative_Return_5"] = df["Close"].pct_change().rolling(5).sum()
            df["Above_SMA20"] = (df["Close"] > sma20).astype(int)
            df["Volume_Spike"] = (df["Volume"] > df["Volume"].rolling(20).mean() * 1.5).astype(int)

            # --- Targety ---
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            df["Target_3day"] = (
                df["Close"].shift(-1) + df["Close"].shift(-2) + df["Close"].shift(-3)
            ) / 3 > df["Close"]
            df["Target_3day"] = df["Target_3day"].astype(int)

            # --- Czyszczenie ---
            df.dropna(inplace=True)
            df = df.iloc[:-3]  # aby target_3day miał sens

            # Zapis
            df.to_csv(os.path.join(output_path, f"{ticker}_features.csv"))
            print(f"✅ Zapisano cechy dla {ticker}")