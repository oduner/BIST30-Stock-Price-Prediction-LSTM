import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

bist30_symbols = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
    "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
    "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SOKM.IS",
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS",
    "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "BRSAN.IS", "ALARK.IS"
]

today = datetime.now().date()

def updater():
    raw_folder = "yDatas/Raw"
    output_folder = "yDatas/RawUpdates"
    os.makedirs(output_folder, exist_ok=True)
    
    downloaded = []
    failed = []
    end_date = today + timedelta(days=1)

    for symbol in bist30_symbols:
        try:
            raw_file_path = os.path.join(raw_folder, f"{symbol.replace('.IS', '')}.csv")

            if os.path.exists(raw_file_path):
                df_raw = pd.read_csv(raw_file_path)
                if not df_raw.empty:
                    last_date_str = str(df_raw.iloc[-1]["Price"])
                    start_date = pd.to_datetime(last_date_str).date()
                else:
                    start_date = today - timedelta(days=30)
            else:
                start_date = today - timedelta(days=30)
            
            print(f"{symbol} indiriliyor...")
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty:
                downloaded.append(symbol)
                file_path = os.path.join(output_folder, f"{symbol.replace('.IS', '')}.csv")
                df.to_csv(file_path, index=True)
                print(f"{symbol} kaydedildi.")
            else:
                failed.append(symbol)
                print(f"{symbol} için veri bulunamadı.")
        except Exception as e:
            failed.append(symbol)
            print(f"{symbol} için hata oluştu:", e)
    
    print("\nİşlem özeti:")
    if downloaded:
        print(f"İndirilenler: {', '.join(downloaded)}")
    if failed:
        print(f"Hata oluşanlar: {', '.join(failed)}")

def merger():
    raw_folder = "yDatas/Raw"
    update_folder = "yDatas/RawUpdates"

    for filename in os.listdir(update_folder):
        if filename.endswith(".csv"):
            symbol_name = filename.replace(".csv", "")
            raw_file = os.path.join(raw_folder, filename)
            update_file = os.path.join(update_folder, filename)

            if os.path.exists(raw_file):
                try:
                    df_raw = pd.read_csv(raw_file)
                    df_update = pd.read_csv(update_file)

                    merged_df = pd.concat([df_raw, df_update]).drop_duplicates(subset=["Price"])
                    merged_df = merged_df.sort_values(by="Price").reset_index(drop=True)
                    merged_df = merged_df[:-2]

                    merged_df.to_csv(raw_file, index=False)
                    print(f"{symbol_name} güncellendi ({len(df_update)} satır).")

                except Exception as e:
                    print(f"{symbol_name} birleştirilirken hata oluştu:", e)
            else:
                os.makedirs(raw_folder, exist_ok=True)
                os.replace(update_file, raw_file)
                print(f"{symbol_name} için yeni dosya eklendi.")