import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# Nature style formatting
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

sns.set_theme(style="ticks", rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
})

METHOD_COLORS = {
    'ULM_L': '#443983',
    'DLM_2': '#31688E',
    'DLM_3': '#21918C',
    'LM_C': '#35B779',
    'LM_L': '#90D743',
}

# 1. Individual Differences Plot (Experiment 1)
def plot_individual_differences(axes):
    base_dir = r"Codes/Experiment 1/Data"
    all_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    preferences = []
    
    for f in all_files:
        df = pd.read_csv(f)
        pid = df['ParticipantID'].iloc[0] if 'ParticipantID' in df.columns else os.path.basename(f).split('.')[0]
        
        # We look at ULM_L wins vs total ULM_L appearances
        for block in ['Intensity', 'Spatial']:
            if block == 'Intensity':
                winner_col = 'Chosen_Intensity'
            elif block == 'Spatial':
                winner_col = 'Chosen_Clarity'
            else:
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
    
    for i, block in enumerate(['Intensity', 'Spatial']):
        block_df = pref_df[pref_df['Block'] == block].copy()
        # Sort by win ratio
        block_df = block_df.sort_values('WinRatio', ascending=False).reset_index(drop=True)
        
        ax = axes[i]
        sns.barplot(data=block_df, x=block_df.index, y='WinRatio', ax=ax, color=METHOD_COLORS['ULM_L'], edgecolor='none')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance Level (50%)')
        
        ax.set_title(f'ULM$_L$ Preference Ratio ({block})')
        ax.set_ylabel('Win Ratio when ULM$_L$ is present')
        ax.set_xlabel('Participant (Sorted)')
        ax.set_xticks([])
        ax.set_ylim(0, 1.05)
        
        sns.despine(ax=ax)
        
        if i == 0:
            ax.legend(frameon=False, loc='upper right')

# 2. Staircase Trajectory Plot (Experiment 2)
def plot_staircase_trajectory(ax):
    base_dir = r"Codes/Experiment 2 & Supp-1/Data/Experiment_2"
    # Choose a representative participant, e.g., FLH
    file_path = os.path.join(base_dir, "FLH_experiment2.csv")
    if not os.path.exists(file_path):
        all_files = glob.glob(os.path.join(base_dir, "*.csv"))
        if len(all_files) > 0:
            file_path = all_files[0]
        else:
            print("No experiment 2 data found.")
            return

    df = pd.read_csv(file_path)
    
    methods = df['Condition'].unique()
    for method in methods:
        method_df = df[df['Condition'] == method].sort_values('TrialInCondition').copy()
        
        # Plot physical intensity (IntensityLevel) or firmware Strength (StrengthBefore)
        ax.plot(method_df['TrialInCondition'], method_df['IntensityLevel'], 
                marker='o', markersize=3, label=method, 
                color=METHOD_COLORS.get(method, 'k'), linewidth=1.0, alpha=0.8)
        
        # Mark reversals
        reversals = method_df[method_df['ReversalHappened'] == 1]
        ax.scatter(reversals['TrialInCondition'], reversals['IntensityLevel'],
                   color='red', s=20, zorder=5, marker='x')
            
    ax.set_title('Adaptive Staircase Convergence (Participant: No 5)')
    ax.set_xlabel('Trial Number within Condition')
    ax.set_ylabel('Stimulus Intensity ($I_{norm}$)')
    
    sns.despine(ax=ax)
    
    # custom legend for reversal
    reversal_marker = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                              markersize=5, label='Reversal')
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates from legend (because scatter might add something weird)
    by_label = dict(zip(labels, handles))
    by_label['Reversal'] = reversal_marker
    ax.legend(by_label.values(), by_label.keys(), title='Modulation Method', loc='upper right', frameon=False)

def create_combined_figure():
    # Nature double column width is ~183 mm (7.2 inches)
    fig = plt.figure(figsize=(7.2, 6))
    
    # Create GridSpec
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25, 
                          left=0.08, right=0.98, top=0.92, bottom=0.08)
    
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    
    # Plot data
    plot_individual_differences([ax_a1, ax_a2])
    plot_staircase_trajectory(ax_b)
    
    # Add panel labels
    fig.text(0.01, 0.98, 'a', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.01, 0.48, 'b', fontsize=20, fontweight='bold', fontfamily='Arial')
    
    os.makedirs('Image/SI', exist_ok=True)
    plt.savefig('Image/SI/Figure_Supplementary_Behavioral_Consistency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Image/SI/Figure_Supplementary_Behavioral_Consistency.png', dpi=300, bbox_inches='tight')
    print("Saved Figure_Supplementary_Behavioral_Consistency")

if __name__ == '__main__':
    create_combined_figure()
