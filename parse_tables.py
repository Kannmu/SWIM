import json

def parse_and_print(filepath, name):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"--- {name} ---")
    print("Fixed Effects:")
    fe = data['model_details']['fixed_effect_params']
    # Wait, the json also has p-values for pairs, but we need p-values for the fixed effects against the reference (ULM_L).
    # Actually, the user asked for a complete statistical table including fixed effect coefficients, SEs, Z-values, p-values, and random effect variance.
    # We can reconstruct it or read it from the JSON.
    print(fe)
    print("SE by method:")
    print(data['se_by_method'])
    print("Scores by method (centered):")
    print(data['score_by_method'])
    print("Model details:")
    for k, v in data['model_details'].items():
        if k not in ['cov_fixed_effects', 'cov_scores']:
            print(f"{k}: {v}")
    
    # Actually, statsmodels BinomialBayesMixedGLM returns vcp (variance of random effects). Let's check `result` or optimizer_converged.
    print("VCP/Variance:")
    try:
        opt = data['optimizer_converged']
        # the parameters include fixed effects and vcp
        print(len(opt['jac']))
    except:
        pass

parse_and_print("Codes/Experiment 1/Analysis/Intensity_detailed_results.json", "Intensity")
parse_and_print("Codes/Experiment 1/Analysis/Clarity_detailed_results.json", "Clarity")
