import numpy as np
from sklearn.metrics import mean_squared_error

# load predictions
mlp_reg = np.load("mlp_reg_preds.npy")
swin_reg = np.load("swin_reg_preds.npy")
y_true = np.load("y_reg_true.npy")

# combine
final_reg = 0.6 * swin_reg + 0.4 * mlp_reg

rmse = np.sqrt(mean_squared_error(y_true, final_reg))

print("\n🔥 FINAL REGRESSION RESULTS")
print("RMSE:", round(rmse, 3))