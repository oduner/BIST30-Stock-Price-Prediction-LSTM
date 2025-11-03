import os
import logging
import traceback
from glob import glob
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from config_module import logger, CONFIG
from data_module import load_and_preprocess_all, create_sequences
from training_module import train_model_enhanced
from prediction_module import walk_forward_preds_enhanced


def run_all_companies(start_index: int = None, batch_size: int = None):
    
    os.makedirs("Models/OpenChange", exist_ok=True)
    os.makedirs("Results/OpenChange", exist_ok=True)

    start = start_index if start_index is not None else CONFIG.get("company_index", 0)
    size = batch_size if batch_size is not None else 1
    end = start + size

    company_files = sorted(glob("yDatas/Bist/*.csv"))
    if not company_files:
        logging.error("Could not find .csv files at yDatas/Bist !")
        return

    for idx in range(start, end):
        try:
            if idx < 0 or idx >= len(company_files):
                logger.warning(f"Index {idx} could not find .csv file, skipping.")
                continue

            company_file = company_files[idx]
            company_code = os.path.splitext(os.path.basename(company_file))[0]
            CONFIG["company_index"] = idx
            logger.info(f"\n{'='*50}\n{company_code} processing (Index: {idx})\n{'='*50}")

            df_train, df_test = load_and_preprocess_all()

            if df_train.empty or df_test.empty:
                logging.warning(f"{company_code} could not find enough data, skipping.")
                continue

            logger.info(f"Training data shape: {df_train.shape}")
            logger.info(f"Test data shape: {df_test.shape}")
            logger.info(f"Full data length: {len(df_train)+len(df_test)}")

            scaler_X = RobustScaler(quantile_range=(5, 95))
            scaler_X.fit(df_train[CONFIG["features"]])

            scaler_y = RobustScaler(quantile_range=(5, 95))
            scaler_y.fit(df_train[["OpenChange"]])

            X_train_scaled = scaler_X.transform(df_train[CONFIG["features"]])
            X_seq_train, y_full_features_train = create_sequences(X_train_scaled, CONFIG["window_size"])

            open_change_idx = CONFIG["features"].index("OpenChange")
            y_price_train_scaled = y_full_features_train[:, open_change_idx].reshape(-1, 1)

            y_dir_train = df_train["ODirection"].iloc[CONFIG["window_size"]:].values.astype(int)

            if X_seq_train.shape[0] == 0 or y_price_train_scaled.shape[0] == 0 or y_dir_train.shape[0] == 0:
                logger.error("Training sequences or targets are empty after preparation. Check window_size and data length.")
                return

            if X_seq_train.shape[0] != y_price_train_scaled.shape[0] or X_seq_train.shape[0] != y_dir_train.shape[0]:
                logger.error(f"Mismatch in training sequence/target lengths: X_seq_train={X_seq_train.shape[0]}, "
                            f"y_price_train_scaled={y_price_train_scaled.shape[0]}, y_dir_train={y_dir_train.shape[0]}")
                return

            logger.info("Training multiple models and selecting the best performer...")
            best_model, best_fold_idx, calibration_params = train_model_enhanced(
                X_seq_train, y_price_train_scaled, y_dir_train, company_code, CONFIG
            )

            if best_model is None:
                logger.error("No models were trained successfully. Exiting main flow.")
                return

            logger.info(f"Performing walk-forward prediction with best model (Fold {best_fold_idx})...")
            window_w = CONFIG["window_size"]

            if len(X_train_scaled) < window_w:
                logger.error(f"Scaled training data length ({len(X_train_scaled)}) is less than window_size ({window_w}). "
                            f"Cannot initialize walk-forward window from training data. Adjust logic or data.")
                return


            result_df = walk_forward_preds_enhanced(
                best_model, calibration_params, df_test, scaler_y, 
                X_train_scaled, scaler_X, CONFIG
            )

            logger.info("\n--- Enhanced ML Pipeline with Best Model Selection Completed ---")
            logger.info("Main execution flow completed successfully.")

            model_dir = f"Models/OpenChange"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/{company_code}_model.keras"
            best_model.save(model_path)

            scaler_path = f"{model_dir}/{company_code}_scaler_X.pkl"
            joblib.dump(scaler_X, scaler_path)
            scaler_y_path = f"{model_dir}/{company_code}_scaler_y.pkl"
            joblib.dump(scaler_y, scaler_y_path)

            result_dir = f"Results/OpenChange"
            os.makedirs(result_dir, exist_ok=True)
            result_path = f"{result_dir}/{company_code}_results.csv"
            result_df.to_csv(result_path, index=False)

            logger.info(f"Artifacts saved: Model, scalers, and results for {company_code}")

            logging.info(f"{company_code} processing is done successfully.")

        except Exception as e:
            logging.error(f"{company_code} processing fault !: {str(e)}")
            logging.debug(traceback.format_exc())

    logging.info("ALL PROCESSES COMPLETED SUCCESSFULLY")