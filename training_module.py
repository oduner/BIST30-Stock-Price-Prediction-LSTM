import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.metrics import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError, BinaryAccuracy, Precision, Recall

from config_module import logger
from model_module import create_enhanced_model
from calibration_module import calibrate_probabilities, find_optimal_threshold
from callbacks_module import LrLogger
from visualization_module import plot_training_history


def train_model_enhanced(X_train: np.ndarray, y_price: np.ndarray, y_dir: np.ndarray, company_code, CONFIG: dict):
    tscv = TimeSeriesSplit(n_splits=CONFIG["tscv_splits"])
    fold_results = []
    calibration_data = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        logger.info(f"Training Enhanced Fold {fold+1}/{CONFIG['tscv_splits']}")
        
        y_dir_fold_train = y_dir[train_idx].flatten()
        y_dir_fold_val = y_dir[val_idx].flatten()
        
        train_counts = np.bincount(y_dir_fold_train, minlength=2)
        train_total = train_counts.sum()
        
        class_weights = {i: train_total / (2 * train_counts[i]) for i in range(2)}
        sample_weights = np.array([class_weights[cls] for cls in y_dir_fold_train])
        
        model = create_enhanced_model((CONFIG["window_size"], len(CONFIG["features"])))
        
        model.compile(
            optimizer=RMSprop(learning_rate=CONFIG["learning_rate"], clipnorm=0.5),
            loss={
                "price_output": "huber",
                "direction_output": "binary_crossentropy"
            },
            loss_weights={"price_output": 0.75, "direction_output": 0.25},
            metrics={
                "price_output": [
                    MeanAbsoluteError(name="mae_price"),
                    MeanSquaredError(name='mean_squared_error'),
                    MeanSquaredLogarithmicError(name='mean_squared_logarithmic_error'),
                ],
                "direction_output": [
                    BinaryAccuracy(name="binary_accuracy"),
                    Precision(name="precision"),
                    Recall(name="recall"),
                ]
            }
        )
        
        lr_logger = LrLogger()
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], restore_best_weights=True, verbose=1),
            ModelCheckpoint(f"temp_model_fold{fold+1}.keras", monitor="val_loss", save_best_only=True, verbose=1),
            TensorBoard(log_dir=os.path.join("logs", f"fold{fold+1}")),
            ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=8, verbose=1, cooldown=5, min_lr=1e-8),
            lr_logger
        ]
        
        history_obj = model.fit(
            X_train[train_idx],
            [y_price[train_idx], y_dir[train_idx].reshape(-1, 1)],
            validation_data=(
                X_train[val_idx],
                [y_price[val_idx], y_dir[val_idx].reshape(-1, 1)]
            ),
            sample_weight=[np.ones(len(train_idx)), sample_weights],
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        history_obj.history['lr'] = lr_logger.lrs
        
        val_predictions = model.predict(X_train[val_idx], verbose=0)
        val_dir_prob = val_predictions[1].flatten()
        
        optimal_threshold = find_optimal_threshold(y_dir_fold_val, val_dir_prob, metric='f1')
        logger.info(f"Fold {fold+1} optimal threshold: {optimal_threshold:.3f}")
        
        calibration_data.append({
            'y_true': y_dir_fold_val,
            'y_prob': val_dir_prob,
            'fold': fold
        })
        
        val_dir_pred_optimal = (val_dir_prob >= optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_dir_fold_val, val_dir_pred_optimal)
        optimal_f1 = f1_score(y_dir_fold_val, val_dir_pred_optimal, zero_division=0)
        
        
        if "val_loss" in history_obj.history:
            best_val_loss = min(history_obj.history["val_loss"])
            best_epoch = np.argmin(history_obj.history["val_loss"])
        else:
            combined_val_loss = [
                p + d for p, d in zip(
                    history_obj.history.get("val_price_output_loss", [0]),
                    history_obj.history.get("val_direction_output_loss", [0])
                )
            ]
            best_val_loss = min(combined_val_loss) if combined_val_loss else float('inf')
            best_epoch = np.argmin(combined_val_loss) if combined_val_loss else 0
        
        try:
            best_price_mae = history_obj.history.get('val_price_output_mae_price', [float('inf')])[best_epoch]
            best_dir_accuracy = history_obj.history.get('val_direction_output_binary_accuracy', [0.0])[best_epoch]
        except IndexError:
            best_price_mae = min(history_obj.history.get('val_price_output_mae_price', [float('inf')]))
            best_dir_accuracy = max(history_obj.history.get('val_direction_output_binary_accuracy', [0.0]))
        
        fold_results.append({
            'fold': fold + 1,
            'model': model,
            'history': history_obj.history,
            'val_loss': best_val_loss,
            'price_mae': best_price_mae,
            'direction_accuracy': best_dir_accuracy,
            'best_epoch': best_epoch,
            'model_path': f"temp_model_fold{fold+1}.keras",
            'optimal_threshold': optimal_threshold,
            'optimal_accuracy': optimal_accuracy,
            'optimal_f1': optimal_f1,
        })
        
        logger.info(f"Fold {fold+1} Enhanced Metrics:")
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        logger.info(f"Standard accuracy (0.5): {best_dir_accuracy:.4f}")
        logger.info(f"Optimized accuracy ({optimal_threshold:.3f}): {optimal_accuracy:.4f}")
        logger.info(f"Optimized F1: {optimal_f1:.4f}")
        
        if CONFIG.get("plot_history", False):
            plot_training_history(history_obj.history, fold+1, company_code)
    
    if not fold_results:
        logger.error("No models were trained successfully.")
        return None, None, None, None, None
    
    all_y_true = np.concatenate([data['y_true'] for data in calibration_data])
    all_y_prob = np.concatenate([data['y_prob'] for data in calibration_data])
    
    global_calibrator = calibrate_probabilities(all_y_true, all_y_prob, method='isotonic')
    global_optimal_threshold = find_optimal_threshold(all_y_true, all_y_prob, metric='f1')
    
    logger.info(f"Global optimal threshold: {global_optimal_threshold:.3f}")
    
    best_fold_result, reason = select_best_model_enhanced(fold_results)
    best_fold_idx = best_fold_result['fold']
    
    logger.info(f"FINAL ENHANCED MODEL SELECTION: Fold {best_fold_idx}")
    logger.info(f"Selection reason: {reason}")
    logger.info(f"Validation loss: {best_fold_result['val_loss']:.4f}")
    logger.info(f"Optimized accuracy: {best_fold_result['optimal_accuracy']:.4f}")
    logger.info(f"Optimized F1: {best_fold_result['optimal_f1']:.4f}")
    
    calibration_params = {
        'global_optimal_threshold': global_optimal_threshold,
        'fold_optimal_threshold': best_fold_result['optimal_threshold'],
        'global_calibrator': global_calibrator,
        'best_fold': best_fold_idx
    }
    
    for result in fold_results:
        temp_path = result['model_path']
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return best_fold_result['model'], best_fold_idx, calibration_params


def select_best_model_enhanced(fold_results):
    sorted_by_val_loss = sorted(fold_results, key=lambda x: x['val_loss'])
    top_candidates = sorted_by_val_loss[:min(3, len(sorted_by_val_loss))]
    
    logger.info("Top candidates by validation loss:")
    for i, candidate in enumerate(top_candidates):
        logger.info(f"{i+1}. Fold {candidate['fold']} - Val Loss: {candidate['val_loss']:.4f}, "
                   f"Opt F1: {candidate['optimal_f1']:.4f}")
    
    val_loss_threshold = 0.02
    min_val_loss = top_candidates[0]['val_loss']
    
    close_candidates = [
        c for c in top_candidates
        if abs(c['val_loss'] - min_val_loss) / min_val_loss <= val_loss_threshold
    ]
    
    if len(close_candidates) > 1:
        logger.info("Selecting based on combined score (F1 + calibration quality):")
        
        for candidate in close_candidates:
            calibration_score = max(0, 1 - candidate['calibration_error'])
            combined_score = (candidate['optimal_f1'] + calibration_score) / 2
            candidate['combined_score'] = combined_score
            logger.info(f"  Fold {candidate['fold']}: F1={candidate['optimal_f1']:.4f}, "
                       f"Combined={combined_score:.4f}")
        
        best_candidate = max(close_candidates, key=lambda x: x['combined_score'])
        selection_reason = "Best combined F1 and calibration score"
    else:
        best_candidate = top_candidates[0]
        selection_reason = "Lowest validation loss"
    
    return best_candidate, selection_reason