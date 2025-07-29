import pandas as pd
import os
from ta import add_all_ta_features

def prepare_features(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            ticker = filename.split('.')[0]
            df = pd.read_csv(os.path.join(input_path, filename), index_col=0)
            df = df.dropna()
            
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()

            df.to_csv(os.path.join(output_path, f"{ticker}_features.csv"))
            print(f"Zapisano cechy dla {ticker}")