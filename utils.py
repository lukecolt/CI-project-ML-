import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def export_model_string(model, col_names, nr_decimals=3):
    coefs = np.round(model.coef_, nr_decimals)
    intercept = np.round(model.intercept_, nr_decimals)
    coef_pairs = [f"{c}*{col}" for c, col in zip(coefs, col_names) if c != 0]
    # ["1.5*CRIM", "0.4*NX", ...]
    return " + ".join(coef_pairs) + f" + {intercept}"

def print_model_statistics(y_train, y_train_est, y_test, y_test_est):
    mse_train = mean_squared_error(y_train, y_train_est)
    mse_test = mean_squared_error(y_test, y_test_est)    
    r2_train = r2_score(y_train, y_train_est)
    r2_test = r2_score(y_test, y_test_est)
    mae_train = mean_absolute_error(y_train, y_train_est)
    mae_test = mean_absolute_error(y_test, y_test_est)
    
    print(f"Train MSE: {mse_train}")
    print(f"Test MSE: {mse_test}")
    print()
    print(f"Train R²: {r2_train}")
    print(f"Test R²: {r2_test}")
    print()
    print(f"Train MAE: {mae_train}")
    print(f"Test MAE: {mae_test}")
    print()