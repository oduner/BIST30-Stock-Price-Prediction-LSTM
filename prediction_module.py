import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config_module import logger

def _calculate_classification_metrics(y_true, y_pred, is_single=False):
    
    if is_single:
        match = int(y_true[-1] == y_pred[-1])
        return {
            'accuracy': match,
            'precision': match,
            'recall': match,
            'f1': match
        }
    
    if len(set(y_true)) == 1:
        accuracy = accuracy_score(y_true, y_pred)
        return {
            'accuracy': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'f1': accuracy
        }
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0, average='binary'),
        'recall': recall_score(y_true, y_pred, zero_division=0, average='binary'),
        'f1': f1_score(y_true, y_pred, zero_division=0, average='binary')
    }


def walk_forward_preds_enhanced(best_model: tf.keras.Model,
                               calibration_params: dict,
                               df_test: pd.DataFrame,
                               scaler_y: RobustScaler,
                               X_train_scaled: np.ndarray,
                               scaler_X: RobustScaler,
                               CONFIG: dict):
    
    optimal_threshold = calibration_params.get('fold_optimal_threshold', 0.5)
    calibrator = calibration_params.get('global_calibrator', None)
    
    window_size = CONFIG["window_size"]
    features = CONFIG["features"]
    
    current_window = np.copy(X_train_scaled[-window_size:])
    
    result = []
    predicted_opens = []
    predicted_dirs = []
    predicted_dirs_calibrated = []
    dir_probabilities = []
    dir_probabilities_calibrated = []
    true_opens = []
    true_dirs = []
    
    logger.info(f"Enhanced walk-forward prediction starting with threshold: {optimal_threshold:.3f}")
    
    df_slice = df_test.reset_index(drop=True)
    
    for step in range(len(df_slice)):
        price_scaled, dir_prob = best_model.predict(
            current_window.reshape(1, window_size, len(features)),
            verbose=0
        )
        
        price_unscaled = scaler_y.inverse_transform(price_scaled)[0][0]
        raw_prob = dir_prob[0, 0]
        
        if calibrator is not None:
            try:
                if hasattr(calibrator, 'predict_proba'):
                    calibrated_prob = calibrator.predict_proba([[raw_prob]])[0, 1]
                else:
                    calibrated_prob = calibrator.transform([raw_prob])[0]
                calibrated_prob = np.clip(calibrated_prob, 0.001, 0.999)
            except:
                calibrated_prob = raw_prob
        else:
            calibrated_prob = raw_prob
        
        y_true_price = float(df_slice["OpenChange"].iloc[step])
        y_true_dir = int(df_slice["ODirection"].iloc[step])
        
        y_pred_dir_raw = int(raw_prob >= 0.5)
        y_pred_dir_optimal = int(raw_prob >= optimal_threshold)
        y_pred_dir_calibrated = int(calibrated_prob >= optimal_threshold)
        
        true_values = df_slice.iloc[step][features].values.reshape(1, -1)
        true_df = pd.DataFrame(true_values, columns=features)
        true_scaled = scaler_X.transform(true_df)[0]
        
        current_window = np.vstack([current_window[1:], true_scaled])
        
        predicted_opens.append(price_unscaled)
        predicted_dirs.append(y_pred_dir_optimal)
        predicted_dirs_calibrated.append(y_pred_dir_calibrated)
        dir_probabilities.append(raw_prob)
        dir_probabilities_calibrated.append(calibrated_prob)
        true_opens.append(y_true_price)
        true_dirs.append(y_true_dir)
        
        price_errors = np.array(predicted_opens) - np.array(true_opens)
        mae_price = np.mean(np.abs(price_errors))
        mse_price = np.mean(price_errors**2)
        rmse_price = np.sqrt(mse_price)
        
        pred_abs = np.abs(predicted_opens)
        true_abs = np.abs(true_opens)
        msle_price = np.mean((np.log1p(pred_abs) - np.log1p(true_abs))**2)
        
        accuracy_raw = accuracy_score(true_dirs, [int(p >= 0.5) for p in dir_probabilities])
        
        metrics_opt = _calculate_classification_metrics(
            true_dirs, predicted_dirs, is_single=(len(predicted_dirs) == 1)
        )
        metrics_cal = _calculate_classification_metrics(
            true_dirs, predicted_dirs_calibrated, is_single=(len(predicted_dirs_calibrated) == 1)
        )
        
        
        result.append({
            "Date": df_slice["Date"].iloc[step],
            "Step": step + 1,
            "Real_OpenChange": y_true_price,
            "Predicted_OpenChange": price_unscaled,
            "Real_ODirection": y_true_dir,
            "Predicted_ODirection_Raw": y_pred_dir_raw,
            "Predicted_ODirection_Optimal": y_pred_dir_optimal,
            "Predicted_ODirection_Calibrated": y_pred_dir_calibrated,
            "Direction_Probability_Raw": raw_prob,
            "Direction_Probability_Calibrated": calibrated_prob,
            "Optimal_Threshold": optimal_threshold,
            "Price_Error": price_errors[-1],
            "Cumulative_MSE": mse_price,
            "Cumulative_RMSE": rmse_price,
            "Cumulative_MAE": mae_price,
            "Cumulative_MSLE": msle_price,
            "Cumulative_Accuracy_Raw": accuracy_raw,
            "Cumulative_Accuracy_Optimal": metrics_opt['accuracy'],
            "Cumulative_Accuracy_Calibrated": metrics_cal['accuracy'],
            "Cumulative_Precision_Optimal": metrics_opt['precision'],
            "Cumulative_Recall_Optimal": metrics_opt['recall'],
            "Cumulative_F1_Optimal": metrics_opt['f1'],
            "Cumulative_Precision_Calibrated": metrics_cal['precision'],
            "Cumulative_Recall_Calibrated": metrics_cal['recall'],
        })
    
    result_df = pd.DataFrame(result)
    
    logger.info(f"Walk-forward predictions done. Steps: {step}")
    logger.info(f"Final RMSE: {result_df['Cumulative_RMSE'].iloc[-1]:.6f}")
    logger.info(f"Final MAE: {result_df['Cumulative_MAE'].iloc[-1]:.6f}")
    logger.info(f"Final Accuracy: {result_df['Cumulative_Accuracy_Calibrated'].iloc[-1]:.4f}")
    logger.info(f"Final F1 Score: {result_df['Cumulative_F1_Optimal'].iloc[-1]:.4f}")
    
    return result_df