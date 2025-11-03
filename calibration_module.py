from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def calibrate_probabilities(y_true, y_prob, method='isotonic'):
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_prob, y_true)
        return calibrator
    elif method == 'platt':
        calibrator = LogisticRegression()
        calibrator.fit(y_prob.reshape(-1, 1), y_true)
        return calibrator
    return None


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    def objective(threshold):
        y_pred = (y_prob >= threshold).astype(int)
        if metric == 'f1':
            return -f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            return -accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return -precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            return -recall_score(y_true, y_pred, zero_division=0)
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            return -(prec + rec) / 2

    result = minimize_scalar(objective, bounds=(0.1, 0.9), method='bounded')
    return result.x