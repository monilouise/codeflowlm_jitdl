import os
import pickle
import pandas as pd

# Path to the directory containing the result.pickle files
base_dir = 'mlruns'

# To store the consolidated results
consolidated_results = []

# Walk through all subdirectories to find result.pickle files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == 'results.pickle':
            file_path = os.path.join(root, file)
            # Load each result.pickle file and append its content to consolidated_results
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                consolidated_results.append(result)

# Save the consolidated results to a new file called Results.pickle
consolidated_results_path = 'mlruns/Results.pickle'
with open(consolidated_results_path, 'wb') as f:
    pickle.dump(consolidated_results, f)

print(f"Consolidated file saved at {consolidated_results_path}")
