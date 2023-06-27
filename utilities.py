# MSE e R^2 sono incluse in scikit-learn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# definisco l'errore relativo
def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def rmspe(y_real, y_pred):
    return np.sqrt(np.mean((y_pred /y_real - 1) ** 2))

def print_eval(X, y, model, tree=False):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    re = relative_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"   Mean squared error: {mse:>12.5}")
    print(f"       Relative error: {re:>12.5%}" if not tree else f"    RMSPE: {rmspe(y, preds):>12.5}")
    print(f"R-squared coefficient: {r2:>12.5}")

