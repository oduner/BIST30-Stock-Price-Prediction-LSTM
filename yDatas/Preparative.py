import os
import pandas as pd
from glob import glob

def prebist():
    csv_path = "Raw/*.csv"
    csv_files = glob(csv_path)

    for file in csv_files:
        df = pd.read_csv(file)
        df["Price"] = pd.to_datetime(df["Price"])
        df.rename(columns={"Price": "Date"})
        df.columns = ["Date","Close", "High", "Low", "Open", "Volume"]
        
        df.reset_index(drop=True, inplace=True)
        df[["Close", "High", "Low", "Open", "Volume"]] = df[["Close", "High", "Low", "Open", "Volume"]].astype(float)
        df["Change"] = df["Close"].pct_change() * 100
        df["Change"] = df["Change"].astype(float)
        df["OpenChange"] = df["Open"].pct_change() * 100 
        df["OpenChange"] = df["OpenChange"].astype(float)
        df = df.dropna()
        df = df[:-64]
        df.reset_index(drop=True, inplace=True)
        code = os.path.basename(file)
        df.index.name = code[:-4]
        
        bist_file_path = "Bist"
        os.makedirs(bist_file_path, exist_ok=True)
        bist_file_path = os.path.join("Bist", f"{code[:-4]}.csv")
        df.to_csv(bist_file_path, index=False)
        print(f"{code[:-4]} işlenip kaydedildi.")

def pretest():
    csv_path = "Raw/*.csv"
    csv_files = glob(csv_path)

    for file in csv_files:
        df = pd.read_csv(file)
        df["Price"] = pd.to_datetime(df["Price"])
        df.rename(columns={"Price": "Date"})
        df.columns = ["Date","Close", "High", "Low", "Open", "Volume"]
        
        df.reset_index(drop=True, inplace=True)
        df[["Close", "High", "Low", "Open", "Volume"]] = df[["Close", "High", "Low", "Open", "Volume"]].astype(float)
        df["Change"] = df["Close"].pct_change() * 100
        df["Change"] = df["Change"].astype(float)
        df["OpenChange"] = df["Open"].pct_change() * 100 
        df["OpenChange"] = df["OpenChange"].astype(float)
        df = df.dropna()
        df = df[-64:]
        df.reset_index(drop=True, inplace=True)
        code = os.path.basename(file)
        df.index.name = code[:-4]
        
        test_file_path = "Test"
        os.makedirs(test_file_path, exist_ok=True)
        test_file_path = os.path.join("Test", f"{code[:-4]}.csv")
        df.to_csv(test_file_path, index=False)
        print(f"{code[:-4]} işlenip kaydedildi.")