import os
import json
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from scipy.stats import pearsonr, spearmanr

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

METHOD_COLORS = {
    'ULM_L': '#443983',
    'DLM_2': '#31688E',
    'DLM_3': '#21918C',
    'LM_C': '#35B779',
    'LM_L': '#90D743',
}

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
INTENSITY_JSON_PATH = OUTPUT_DIR / 'Intensity_detailed_results.json'
SIMULATION_SCRIPT_PATH = BASE_DIR.parent / 'Simulation 1' / 'visualize_experiment1.py'
SIMULATION_MAT_PATH = Path(r'k:\Work\SWIM\Sim\Outputs_Experiment1\experiment1_data.mat')
METHOD_ORDER = ['ULM_L', 'DLM_2', 'DLM_3', 'LM_C', 'LM_L']


def format_method_name(name):
    if '_' in name:
        parts = name.split('_', 1)
        base = parts[0]
        sub = parts[1]
        return rf"$\mathregular{{{base}}}_{{\mathregular{{{sub}}}}}$"
    return name


def min_max_normalize(series, invert=False):
    values = pd.to_numeric(series, errors='coerce').astype(float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        normalized = pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def load_intensity_scores(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    scores = payload.get('score_by_method', {})
    if not scores:
        raise ValueError(f'No score_by_method found in {json_path}')
    return pd.Series(scores, name='Intensity_Score', dtype=float)


class Mat73Struct:
    def __repr__(self):
        return f"Mat73Struct({list(self.__dict__.keys())})"


def load_mat_h5py(path):
    f = h5py.File(path, 'r')

    def _convert(item):
        if isinstance(item, h5py.Group):
            d = Mat73Struct()
            for k in item.keys():
                setattr(d, k, _convert(item[k]))
            return d
        if isinstance(item, h5py.Dataset):
            val = item[()]
            if item.dtype == np.dtype('O'):
                flat_val = val.flatten()
                converted_list = []
                for ref in flat_val:
                    if ref:
                        converted_list.append(_convert(f[ref]))
                return np.array(converted_list)
            if isinstance(val, np.ndarray) and val.dtype == np.uint16:
                try:
                    chars = val.flatten()
                    return ''.join([chr(c) for c in chars if c != 0])
                except Exception:
                    pass
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                val = val.T
            val = np.squeeze(val)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            return val
        return item

    mat = {}
    for k in f.keys():
        if k == '#refs#':
            continue
        mat[k] = _convert(f[k])
    return mat


def load_simulation_module(module_path):
    spec = importlib.util.spec_from_file_location('visualize_experiment1_module', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_simulation_data(mat_path):
    if not mat_path.exists():
        raise FileNotFoundError(f'Simulation data file not found: {mat_path}')
    try:
        return scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return load_mat_h5py(str(mat_path))
    except Exception as e:
        if 'v7.3' in str(e) or 'HDF reader' in str(e):
            return load_mat_h5py(str(mat_path))
        raise


def extract_simulation_metrics(module, mat_data):
    results = module.extract_results(mat_data)
    dt = float(mat_data['dt'])

    rows = []
    for item in results:
        method = str(item['name'])
        rows.append({
            'Method': method,
            'Peak_Stress_raw': float(np.max(np.asarray(item['tau_peak'], dtype=float))),
            'Max_Gradient_raw': float(np.max(np.asarray(item['grad_mag'], dtype=float))),
            'Max_Jerk_raw': float(np.max(np.asarray(item['tau_dt_peak'], dtype=float))),
            'FFI_raw': float(module.compute_frequency_fidelity_index(item, dt)),
            'DWCI_raw': float(module.compute_directional_wavefront_concentration_index(item, dt)),
        })

    sim_df = pd.DataFrame(rows)
    if sim_df.empty:
        raise ValueError('No simulation results extracted from MAT data.')

    sim_df = sim_df.set_index('Method').reindex(METHOD_ORDER)
    if sim_df.isnull().any().any():
        missing = sim_df[sim_df.isnull().any(axis=1)].index.tolist()
        raise ValueError(f'Missing simulation metrics for methods: {missing}')

    # Physical quantities: Use raw values but compute log10 for correlation analysis
    # FFI and DWCI are ratios/indices (0-1), use raw values directly
    sim_df['Peak_Stress'] = sim_df['Peak_Stress_raw']
    sim_df['Max_Gradient'] = sim_df['Max_Gradient_raw']
    sim_df['Max_Jerk'] = sim_df['Max_Jerk_raw']
    sim_df['FFI'] = sim_df['FFI_raw']
    sim_df['DWCI'] = sim_df['DWCI_raw']

    # Compute Log10 for physical metrics to handle large ranges/singularities
    # Adding a small epsilon just in case, though these are magnitudes > 0
    epsilon = 1e-10
    sim_df['Log_Peak_Stress'] = np.log10(sim_df['Peak_Stress_raw'] + epsilon)
    sim_df['Log_Max_Gradient'] = np.log10(sim_df['Max_Gradient_raw'] + epsilon)
    sim_df['Log_Max_Jerk'] = np.log10(sim_df['Max_Jerk_raw'] + epsilon)

    sim_df = sim_df.reset_index()
    return sim_df


def build_analysis_dataframe():
    intensity_scores = load_intensity_scores(INTENSITY_JSON_PATH)
    sim_module = load_simulation_module(SIMULATION_SCRIPT_PATH)
    mat_data = load_simulation_data(SIMULATION_MAT_PATH)
    sim_df = extract_simulation_metrics(sim_module, mat_data)

    intensity_df = intensity_scores.reindex(METHOD_ORDER).rename_axis('Method').reset_index()
    if intensity_df['Intensity_Score'].isna().any():
        missing = intensity_df.loc[intensity_df['Intensity_Score'].isna(), 'Method'].tolist()
        raise ValueError(f'Missing intensity scores for methods: {missing}')

    df = sim_df.merge(intensity_df, on='Method', how='inner')
    df['Formatted_Method'] = df['Method'].apply(format_method_name)
    return df


def save_metric_table(df, output_dir):
    export_cols = [
        'Method',
        'Intensity_Score',
        'Peak_Stress_raw',
        'Max_Gradient_raw',
        'Max_Jerk_raw',
        'FFI_raw',
        'DWCI_raw',
        'Peak_Stress',
        'Max_Gradient',
        'Max_Jerk',
        'FFI',
        'DWCI',
    ]
    df[export_cols].to_csv(output_dir / 'Correlation_Source_Data.csv', index=False)


def plot_correlation_matrix(df, output_dir, metrics):
    corr_matrix_pearson = df[metrics].corr(method='pearson')
    corr_matrix_spearman = df[metrics].corr(method='spearman')

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), gridspec_kw={'wspace': 0.12, 'width_ratios': [1, 1, 0.05]})

    # Pearson
    mask = np.triu(np.ones_like(corr_matrix_pearson, dtype=bool))
    sns.heatmap(
        corr_matrix_pearson,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        mask=mask,
        cbar=False,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
    )
    axes[0].set_title('Pearson Correlation Matrix', fontweight='bold', fontsize=16)
    axes[0].tick_params(axis='x', rotation=45, labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)

    # Spearman
    mask = np.triu(np.ones_like(corr_matrix_spearman, dtype=bool))
    sns.heatmap(
        corr_matrix_spearman,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        mask=mask,
        cbar=True,
        cbar_ax=axes[2],
        cbar_kws={'label': 'Correlation Coefficient'},
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[1]
    )
    axes[1].set_yticklabels([])
    axes[1].set_title('Spearman Correlation Matrix', fontweight='bold', fontsize=16)
    axes[1].tick_params(axis='x', rotation=45, labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].set_ylabel('') # Remove Y label for the second plot to save space if shared

    # Adjust colorbar font size
    cbar = axes[1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation Coefficient', size=16, weight='bold')

    # plt.tight_layout() # tight_layout can interfere with cbar_ax
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(output_dir / 'Correlation_Matrix_Combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Correlation_Matrix_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_panels(df, output_dir):
    import matplotlib.lines as mlines
    
    # Use 2x3 layout with shared Y axis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True,gridspec_kw={'wspace': 0.2})
    # Adjust layout to make room for super titles
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)

    # Metric Groups
    physical_metrics = [
        ('Peak_Stress', 'Peak Stress', 'Pa'),
        ('Max_Gradient', 'Max Gradient', 'Pa/m'),
        ('Max_Jerk', 'Max Jerk', 'Pa/s')
    ]
    coherence_metrics = [
        ('FFI', 'Frequency Fidelity Index', ''),
        ('DWCI', 'Directional Wavefront Concentration', '')
    ]

    # Row 1: Physical Extrema (Log Scale)
    for i, (metric, title, unit) in enumerate(physical_metrics):
        ax = axes[0, i]
        
        # Regression Line (Log-Linear: y ~ log(x))
        sns.regplot(
            data=df,
            x=metric,
            y='Intensity_Score',
            ax=ax,
            color='gray',
            scatter=False,
            logx=True,  # Fits y = a + b * ln(x)
            ci=None,
            line_kws={'linestyle': '--', 'alpha': 0.6}
        )
        
        # Scatter Points
        for _, row in df.iterrows():
            ax.scatter(
                row[metric],
                row['Intensity_Score'],
                color=METHOD_COLORS.get(row['Method'], '#333333'),
                s=200,
                zorder=5,
                edgecolors='white',
                linewidth=1.5
            )

        # Set Log Scale for X-axis
        ax.set_xscale('log')

        if metric == 'Max_Jerk':
            ax.xaxis.set_minor_locator(ticker.FixedLocator([1.1e6, 1.2e6, 1.3e6, 1.4e6]))
            # ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f"$\\mathdefault{{{x/1e6:g}\\times10^{{6}}}}$"))
        
        # Labels and Title
        ax.set_title(title, fontsize=14, fontweight='bold')
        label_text = f"{title} ({unit})" if unit else title
        ax.set_xlabel(label_text, fontsize=12)
        ax.tick_params(axis='x', which='both', rotation=0)
        plt.setp(ax.get_xticklabels(which='both'), ha='center')
        ax.set_ylabel("")
        # ax.legend(loc="upper right")
        if i == 0:
            ax.set_ylabel('Intensity Score (Log-odds)', fontsize=12, fontweight='bold')
        
        # Statistics (Pearson/Spearman on Log-Transformed Data)
        log_col = f"Log_{metric}"
        p_corr, p_pval = pearsonr(df[log_col], df['Intensity_Score'])
        s_corr, s_pval = spearmanr(df[log_col], df['Intensity_Score'])
        
        stats_text = (
            f"$r_P$ = {p_corr:.2f}, $p$ = {p_pval:.3f}\n"
            f"$\\rho_S$ = {s_corr:.2f}, $p$ = {s_pval:.3f}"
        )
        
        # Inset Stats
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=14,ha="right",
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        ax.grid(True, linestyle=':', alpha=0.6, which='both')

    # Row 2: Spatiotemporal Coherence (Linear Scale)
    for i, (metric, title, unit) in enumerate(coherence_metrics):
        ax = axes[1, i]
        
        # Regression Line (Linear)
        sns.regplot(
            data=df,
            x=metric,
            y='Intensity_Score',
            ax=ax,
            color='gray',
            scatter=False,
            ci=None,
            line_kws={'linestyle': '--', 'alpha': 0.6}
        )
        
        # Scatter Points
        for _, row in df.iterrows():
            ax.scatter(
                row[metric],
                row['Intensity_Score'],
                color=METHOD_COLORS.get(row['Method'], '#333333'),
                s=200,
                zorder=5,
                edgecolors='white',
                linewidth=1.5
            )
            
        # Labels and Title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel("")
        if i == 0:
            ax.set_ylabel('Intensity Score (Log-odds)', fontsize=12, fontweight='bold')
            
        # Statistics (Pearson/Spearman on Linear Data)
        p_corr, p_pval = pearsonr(df[metric], df['Intensity_Score'])
        s_corr, s_pval = spearmanr(df[metric], df['Intensity_Score'])
        
        stats_text = (
            f"$r_P$ = {p_corr:.2f}, $p$ = {p_pval:.3f}\n"
            f"$\\rho_S$ = {s_corr:.2f}, $p$ = {s_pval:.3f}"
        )
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        ax.grid(True, linestyle=':', alpha=0.6)

    # Legend Panel (Bottom Right)
    ax_legend = axes[1, 2]
    ax_legend.axis('off')
    handles = []
    for method in METHOD_ORDER:
        color = METHOD_COLORS.get(method, 'black')
        label = format_method_name(method)
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                     markersize=16, label=label))
    
    ax_legend.legend(handles=handles, title='Modulation Methods', loc='center', 
                     fontsize=16, title_fontsize=18, frameon=False)

    # Super Titles
    # fig.text(0.5, 0.96, "Classical Hypothesis: Local Physical Extrema", ha='center', fontsize=18, fontweight='bold')
    # fig.text(0.5, 0.48, "SWIM Theory: Spatiotemporal Coherence", ha='center', fontsize=18, fontweight='bold')

    # Save
    plt.savefig(output_dir / 'Paradox_Scatter_Plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Paradox_Scatter_Plots.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = build_analysis_dataframe()
    save_metric_table(df, OUTPUT_DIR)
    metrics = ['Intensity_Score', 'Log_Peak_Stress', 'Log_Max_Gradient', 'Log_Max_Jerk', 'FFI', 'DWCI']
    plot_correlation_matrix(df, OUTPUT_DIR, metrics)
    plot_scatter_panels(df, OUTPUT_DIR)
    print(f'Analysis complete. Results saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
