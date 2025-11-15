import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_th_from_test(predictions, target_th=0.4):
  probs = predictions['pred_prob']
  quantile = np.quantile(probs, target_th)
  return quantile