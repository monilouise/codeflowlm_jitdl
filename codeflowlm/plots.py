import matplotlib.pyplot as plt
import numpy as np


def plot(results, model_path):
  g_means = results['g-mean']
  r_diffs = results['r0-r1']
  r1s = results['r1']
  r0s = results['r0']

  plt.figure(figsize=(10, 5))

  # Plot the original metrics
  plt.plot(g_means, label="G-Mean")
  plt.plot(r_diffs, label="|R0-R1|")
  plt.plot(r0s, label="R0")
  plt.plot(r1s, label="R1")

  # Add horizontal lines for averages
  plt.axhline(y=np.mean(g_means), color='C0', linestyle='--', alpha=0.7, label=f"G-Mean Avg = {np.mean(g_means):.2f}")
  plt.axhline(y=np.mean(r_diffs), color='C2', linestyle='--', alpha=0.7, label=f"|R0-R1| Avg = {np.mean(r_diffs):.2f}")
  plt.axhline(y=np.mean(r0s), color='C3', linestyle='--', alpha=0.7, label=f"R0 Avg = {np.mean(r0s):.2f}")
  plt.axhline(y=np.mean(r1s), color='C4', linestyle='--', alpha=0.7, label=f"R1 Avg = {np.mean(r1s):.2f}")

  plt.xlabel("Iteration")
  plt.ylabel("Value")
  plt.title("G-Mean, F1 |R0-R1| over Iterations")
  plt.legend()

  plt.savefig(f'{model_path}/plot.png')  # Save the plot to a file