import numpy as np
import pandas as pd
from scipy.stats import mstats
from river import metrics

def prequential_recalls(predictions, fading_factor):
    recalls = []
    counts = np.zeros(2)
    hits = np.zeros(2)

    targets = predictions['true_labels']
    predictions = predictions['pred_labels']
    print('len(targets) = ', len(targets))
    print('len(predictions) = ', len(predictions))
    n_samples = len(targets)
    for i in range(n_samples):
        label = int(targets[i])
        counts[label] = 1 + fading_factor * counts[label]
        hits[label] = int(label == predictions[i]) + \
            fading_factor * hits[label]
        recalls.append(hits / (counts + 1e-12))
    columns = ['r{}'.format(i) for i in range(2)]
    recalls = pd.DataFrame(recalls, columns=columns)
    return recalls


def prequential_f1s(predictions, fading_factor):
    f1s = []
    recalls = []
    precisions = []
    counts = np.zeros(2)
    hits = np.zeros(2)
    preds = np.zeros(2)
    targets = predictions['true_labels']
    predictions = predictions['pred_labels']
    n_samples = len(targets)

    for i in range(n_samples):
        label = int(targets[i])
        pred = predictions[i]
        counts[label] = 1 + fading_factor * counts[label]
        hits[label] = int(label == predictions[i]) + \
            fading_factor * hits[label]
        preds[label] = 1 + fading_factor * preds[label]
        recalls.append(hits[1] / (counts[1] + 1e-12))
        precisions.append(hits[1] / (preds[1] + 1e-12))
        f1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1] + 1e-12))

    metrics = pd.DataFrame({'f1': f1s, 'precision': precisions, 'recall': recalls})
    return metrics


def prequential_recalls_difference(recalls):
    recalls_difference = recalls.copy()
    recalls_difference['r0-r1'] = (recalls['r0'] - recalls['r1']).abs()
    return recalls_difference


def prequential_gmean(recalls):
    gmean = mstats.gmean(recalls[['r0', 'r1']], axis=1)
    gmean = pd.DataFrame(gmean, columns=['g-mean'])
    return pd.concat([recalls, gmean], axis='columns')


def prequential_metrics(predictions, fading_factor):
    metrics = prequential_recalls(predictions, fading_factor)
    metrics_f1 = prequential_f1s(predictions, fading_factor)
    metrics = pd.concat([metrics, metrics_f1], axis='columns')
    metrics = prequential_recalls_difference(metrics)
    metrics = prequential_gmean(metrics)
    return metrics


def rolling_roc_auc(predictions):
  metric = metrics.RollingROCAUC()
  true_label = predictions['true_labels']
  pred_prob = predictions['pred_probs']

  for yt, yp in zip(true_label, pred_prob):
    if isinstance(yp, int):
      metric.update(yt, yp)
    elif isinstance(yp, list) or isinstance(yp, np.ndarray):
      metric.update(yt, yp[0])

  return metric.get()


def calculate_prequential_mean_and_std(predictions, decay_factor=0.99):
  metrics = prequential_metrics(predictions, decay_factor)
  g_mean = metrics['g-mean'].mean()
  std_g_mean = metrics['g-mean'].std()
  f1 = metrics['f1'].mean()

  print(f"G-Mean: Mean = {g_mean:.4f}, Standard Deviation = {std_g_mean:.4f}")

  std_f1 = metrics['f1'].std()
  precision = metrics['precision'].mean()
  std_precision = metrics['precision'].std()
  recall = metrics['recall'].mean()
  std_recall = metrics['recall'].std()
  r0 = metrics['r0'].mean()
  std_r0 = metrics['r0'].std()
  r1 = metrics['r1'].mean()
  std_r1 = metrics['r1'].std()
  r_diff = metrics['r0-r1'].mean()
  std_r_diff = metrics['r0-r1'].std()

  roc_auc = rolling_roc_auc(predictions)
  print('roc_auc = ', roc_auc)

  print(f"r_diff: Mean = {r_diff:.4f}, Standard Deviation = {std_r_diff:.4f}")
  print(f"r0: Mean = {r0:.4f}, Standard Deviation = {std_r0:.4f}")
  print(f"r1: Mean = {r1:.4f}, Standard Deviation = {std_r1:.4f}")

  return g_mean, std_g_mean, r_diff, std_r_diff, f1, std_f1, precision, std_precision, recall, std_recall, r0, std_r0, r1, std_r1, roc_auc, metrics