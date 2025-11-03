import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
from data_module import _preprocess_dataframe
from pandas.tseries.offsets import BDay

bist30_symbols = [
        "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
        "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
        "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SOKM.IS",
        "SISE.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS",
        "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "BRSAN.IS", "ALARK.IS"
]

today = datetime.now().date()
print(f"Güncel tarih: {today}")
weekday = datetime.now().weekday()

test_path = "yDatas/Test"
model_path = "Models/OpenChange"
result_path = "Results"

os.makedirs(result_path, exist_ok=True)

for symbol in bist30_symbols:
    try:
        test_file = os.path.join(test_path, f"{symbol.replace('.IS','')}.csv")
        if not os.path.exists(test_file):
            print(f"[{symbol}] Test dosyası bulunamadı, geçiliyor.")
            continue

        df = pd.read_csv(test_file)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', format='mixed')
        df = _preprocess_dataframe(df, symbol.replace('.IS',''))

        last_date = df['Date'].iloc[-1].date()
        last_date_obj = df['Date'].iloc[-1]

        if last_date == today or weekday in [5,6]:
            print(f"[{symbol}] Uygun tarih bulundu ({last_date}). Fine-tuning ve tahmin başlatılıyor...")

            model_file = os.path.join(model_path, f"{symbol.replace('.IS','')}_model.keras")
            scaler_X_file = os.path.join(model_path, f"{symbol.replace('.IS','')}_scaler_X.pkl")
            scaler_y_file = os.path.join(model_path, f"{symbol.replace('.IS','')}_scaler_y.pkl")

            if not all(os.path.exists(f) for f in [model_file, scaler_X_file, scaler_y_file]):
                print(f"[{symbol}] Model veya scaler dosyası eksik, geçiliyor.")
                continue

            model = load_model(model_file)
            scaler_X = joblib.load(scaler_X_file)
            scaler_y = joblib.load(scaler_y_file)

            X = df.drop(columns=["Date","Close","High","Low","Open","Volume","Change"]).values
            y_price = df['OpenChange'].values.reshape(-1, 1)
            y_dir = df['ODirection'].values.reshape(-1, 1).astype('float32')
            
            features_count = X.shape[1]

            X_scaled = scaler_X.transform(X)
            y_price_scaled = scaler_y.transform(y_price)

            def create_sequences_dual(X, y_price, y_dir, time_steps=64):
                Xs, ys_price, ys_dir = [], [], []
                for i in range(len(X) - time_steps): 
                    Xs.append(X[i:(i + time_steps)])
                    ys_price.append(y_price[i + time_steps])
                    ys_dir.append(y_dir[i + time_steps])
                return np.array(Xs), np.array(ys_price), np.array(ys_dir)

            X_seq, y_price_seq, y_dir_seq = create_sequences_dual(X_scaled, y_price_scaled, y_dir, time_steps=64)

            print(f"Fine-tuning için X_seq shape: {X_seq.shape}")
            print(f"Fine-tuning için Y_Price_seq shape: {y_price_seq.shape}")
            print(f"Fine-tuning için Y_Dir_seq shape: {y_dir_seq.shape}")
            
            if len(X_seq) == 0:
                raise ValueError(f"Boş X_seq: Veri uzunluğu ({len(X_scaled)}) 'time_steps' ({64}) değerinden büyük olmalı!")
            
            model.fit(
                X_seq, 
                [y_price_seq, y_dir_seq],
                epochs=1, 
                batch_size=1, 
                verbose=0
            )
            print(f"[{symbol}] Fine-tuning tamamlandı.")

            last_64_rows_scaled = X_scaled[1:] 
            X_to_predict = np.reshape(last_64_rows_scaled, (1, 64, features_count))
            
            price_preds_scaled, dir_preds_prob = model.predict(X_to_predict, verbose=0)
            price_preds = scaler_y.inverse_transform(price_preds_scaled)
            prediction_value = price_preds.flatten()[0] 
            
            prediction_dir_prob = dir_preds_prob.flatten()[0]
            prediction_direction = 1 if prediction_dir_prob >= 0.5 else 0

            prediction_date = last_date_obj + BDay(1)
            
            result_df = pd.DataFrame({
                'Date': [prediction_date],
                'Predicted_OpenChange': [prediction_value],
                'Predicted_ODirection_Prob': [prediction_dir_prob], # Olasılık
                'Predicted_ODirection': [prediction_direction]       # Sınıflandırma (0 veya 1)
            })
            
            result_file_path = os.path.join(result_path, f"{symbol.replace('.IS','')}_forecast.csv")
            
            result_df.to_csv(result_file_path, index=False)
            print(f"[{symbol}] Tahmin kaydedildi: {result_file_path}")
            print(f"[{symbol}] Tahmin (OpenChange): {prediction_value:.4f}, Yön: {prediction_direction} (P={prediction_dir_prob:.4f})")

        else:
            print(f"[{symbol}] Son tarih {last_date}, bugünden farklı. Atlanıyor.")

    except Exception as e:
        print(f"[{symbol}] Hata: {e}")