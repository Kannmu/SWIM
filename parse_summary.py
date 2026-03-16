import json
import numpy as np
from scipy.stats import norm

def get_stats(file):
    with open(file, 'r') as f:
        data = json.load(f)
    print(f"Table for {file}")
    fe = data['model_details']['fixed_effect_params']
    cov = data['model_details']['cov_fixed_effects']
    
    methods = data['model_details']['method_columns']
    for i, m in enumerate(methods):
        coef = fe[m]
        se = np.sqrt(cov[i][i])
        z = coef / se
        p = 2 * norm.sf(abs(z))
        print(f"{m}: Coef={coef:.3f}, SE={se:.3f}, Z={z:.3f}, p={p:.3e}")

get_stats("Codes/Experiment 1/Analysis/Intensity_detailed_results.json")
get_stats("Codes/Experiment 1/Analysis/Clarity_detailed_results.json")

