import os
import shutil
from src.features.prepare_features import prepare_features
from scripts.download_data import download_stock_data
# from src.features.prepare_features import prepare_features
# from src.models.train_model import train_model
# from src.models.evaluate_model import evaluate_model

def main():
    print("📥 [1] Pobieranie danych...")
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
    raw_path = "data/raw"

    if os.path.exists(raw_path):
        shutil.rmtree(raw_path)
    os.makedirs(raw_path, exist_ok=True)

    start_date = "2018-01-01"
    end_date = "2024-12-31"
    download_stock_data(tickers, start_date, end_date, raw_path)

    print("🧹 [2] Przygotowywanie cech...")
    processed_path = "data/processed"
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.makedirs(processed_path, exist_ok=True)
    prepare_features(raw_path, processed_path)

    print("✅ Gotowe!")


if __name__ == "__main__":
    main()