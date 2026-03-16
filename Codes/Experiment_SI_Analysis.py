import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

METHOD_COLORS = {
    'ULM_L': '#443983',
    'DLM_2': '#31688E',
    'DLM_3': '#21918C',
    'LM_C': '#35B779',
    'LM_L': '#90D743',
}

# 1. Individual Differences Plot (Experiment 1)
def plot_individual_differences():
    base_dir = r"Codes/Experiment 1/Data"
    all_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    preferences = []
    
    for f in all_files:
        df = pd.read_csv(f)
        pid = df['ParticipantID'].iloc[0] if 'ParticipantID' in df.columns else os.path.basename(f).split('.')[0]
        
        # We look at ULM_L wins vs total ULM_L appearances
        for block in ['Intensity', 'Spatial']:
            winner_col = f'Chosen_{block}'
            if winner_col not in df.columns:
                continue
                
            subset = df[df['BlockType'] == block].copy()
            if subset.empty:
                continue
                
            # Count how many times ULM_L appeared
            ulm_trials = subset[(subset['StimulusA'] == 'ULM_L') | (subset['StimulusB'] == 'ULM_L')]
            total_ulm = len(ulm_trials)
            
            if total_ulm > 0:
                ulm_wins = len(ulm_trials[ulm_trials[winner_col] == 'ULM_L'])
                preferences.append({
                    'ParticipantID': pid,
                    'Block': block,
                    'WinRatio': ulm_wins / total_ulm
                })

    pref_df = pd.DataFrame(preferences)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, block in enumerate(['Intensity', 'Spatial']):
        block_df = pref_df[pref_df['Block'] == block].copy()
        # Sort by win ratio
        block_df = block_df.sort_values('WinRatio', ascending=False).reset_index(drop=True)
        
        ax = axes[i]
        sns.barplot(data=block_df, x=block_df.index, y='WinRatio', ax=ax, color=METHOD_COLORS['ULM_L'])
        ax.axhline(0.5, color='r', linestyle='--', label='Chance Level (50%)')
        ax.set_title(f'ULM_L Preference Ratio ({block})', fontweight='bold', fontsize=14)
        ax.set_ylabel('Win Ratio when ULM_L is present', fontweight='bold', fontsize=12)
        ax.set_xlabel('Participant (Sorted)', fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    os.makedirs('Image/SI', exist_ok=True)
    plt.savefig('Image/SI/Figure_Individual_Preferences.pdf', dpi=300)
    plt.savefig('Image/SI/Figure_Individual_Preferences.png', dpi=300)
    print("Saved Figure_Individual_Preferences")

# 2. Staircase Trajectory Plot (Experiment 2)
def plot_staircase_trajectory():
    base_dir = r"Codes/Experiment 2 & Supp-1/Data/Experiment_2"
    # Choose a representative participant, e.g., CHY
    file_path = os.path.join(base_dir, "CHY_experiment2.csv")
    if not os.path.exists(file_path):
        all_files = glob.glob(os.path.join(base_dir, "*.csv"))
        if len(all_files) > 0:
            file_path = all_files[0]
        else:
            print("No experiment 2 data found.")
            return

    df = pd.read_csv(file_path)
    
    # We want TrialInCondition and StrengthBefore (or IntensityLevel)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['Condition'].unique()
    for method in methods:
        method_df = df[df['Condition'] == method].sort_values('TrialInCondition').copy()
        
        # Plot physical intensity (IntensityLevel) or firmware Strength (StrengthBefore)
        ax.plot(method_df['TrialInCondition'], method_df['IntensityLevel'], 
                marker='o', markersize=5, label=method, 
                color=METHOD_COLORS.get(method, 'k'), linewidth=1.5)
        
        # Mark reversals
        reversals = method_df[method_df['ReversalHappened'] == 1]
        ax.scatter(reversals['TrialInCondition'], reversals['IntensityLevel'],
                   color='red', s=60, zorder=5, marker='x')
            
    ax.set_title('Adaptive Staircase Convergence (Participant: CHY)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Trial Number within Condition', fontweight='bold', fontsize=14)
    ax.set_ylabel('Stimulus Intensity ($I_{norm}$)', fontweight='bold', fontsize=14)
    ax.legend(title='Modulation Method')
    
    # custom legend for reversal
    import matplotlib.lines as mlines
    reversal_marker = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                              markersize=8, label='Reversal')
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates from legend (because scatter might add something weird)
    by_label = dict(zip(labels, handles))
    by_label['Reversal'] = reversal_marker
    ax.legend(by_label.values(), by_label.keys(), title='Modulation Method', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Image/SI/Figure_Staircase_Trajectory.pdf', dpi=300)
    plt.savefig('Image/SI/Figure_Staircase_Trajectory.png', dpi=300)
    print("Saved Figure_Staircase_Trajectory")


if __name__ == '__main__':
    plot_individual_differences()
    plot_staircase_trajectory()

