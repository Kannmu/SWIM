import os
import glob
import json
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

METHOD_ORDER = ['ULM_L', 'DLM_2', 'DLM_3', 'LM_C', 'LM_L']
METHOD_COLORS = {
    'ULM_L': '#443983',
    'DLM_2': '#31688E',
    'DLM_3': '#21918C',
    'LM_C': '#35B779',
    'LM_L': '#90D743',
}


def format_method_name(name):
    if '_' in name:
        parts = name.split('_', 1)
        base = parts[0]
        sub = parts[1]
        return rf"$\mathregular{{{base}}}_{{\mathregular{{{sub}}}}}$"
    return name


def strength_to_intensity(strength: float) -> float:
    normalized = max(0.0, min(100.0, float(strength))) / 100.0
    return math.sin((math.pi / 2.0) * normalized) ** 2


def intensity_to_strength(intensity_level: float) -> float:
    intensity = max(0.0, min(1.0, float(intensity_level)))
    if intensity <= 0:
        return 0.0
    return (200.0 / math.pi) * math.asin(math.sqrt(intensity))


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient='records')
    return obj


def save_structured_results(output_dir, filename, payload):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_to_serializable)
    print(f"Saved structured analysis results: {path}")


def load_data(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame(), []

    load_logs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
            load_logs.append({
                'file': filename,
                'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
                'status': 'ok'
            })
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            load_logs.append({'file': filename, 'status': 'error', 'error': str(e)})

    if not df_list:
        return pd.DataFrame(), load_logs

    return pd.concat(df_list, ignore_index=True), load_logs


def prepare_threshold_data(df):
    df = df.copy()
    numeric_cols = [
        'GlobalTrial', 'TrialInCondition', 'Correct', 'StrengthBefore', 'StrengthAfter',
        'IntensityLevel', 'ReversalCount', 'ReversalHappened', 'ThresholdEstimate', 'ReactionTime'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    df['IntensityBefore'] = df['StrengthBefore'].apply(strength_to_intensity)
    df['IntensityAfter'] = df['StrengthAfter'].apply(strength_to_intensity)
    df['IntensityThreshold'] = df['ThresholdEstimate'].apply(strength_to_intensity)
    df['ParticipantID'] = df['ParticipantID'].astype(str)
    df['Condition'] = df['Condition'].astype(str)
    return df


def get_final_trials(df):
    ordered = df.sort_values(by=['ParticipantID', 'Condition', 'GlobalTrial', 'TrialInCondition'])
    final_trials = ordered.groupby(['ParticipantID', 'Condition']).last().reset_index()
    return final_trials


def build_threshold_summary(final_trials, conditions):
    summary_rows = []
    for condition in conditions:
        subset = final_trials[final_trials['Condition'] == condition].copy()
        if subset.empty:
            continue
        intensity_vals = subset['IntensityThreshold'].dropna()
        strength_vals = subset['ThresholdEstimate'].dropna()
        reversal_vals = subset['ReversalCount'].dropna()
        n = len(subset)
        summary_rows.append({
            'condition': condition,
            'n': int(n),
            'threshold_intensity_mean': float(intensity_vals.mean()),
            'threshold_intensity_std': float(intensity_vals.std(ddof=1)) if len(intensity_vals) > 1 else 0.0,
            'threshold_intensity_sem': float(intensity_vals.sem()) if len(intensity_vals) > 1 else 0.0,
            'threshold_intensity_median': float(intensity_vals.median()),
            'threshold_intensity_iqr': float(intensity_vals.quantile(0.75) - intensity_vals.quantile(0.25)),
            'threshold_intensity_min': float(intensity_vals.min()),
            'threshold_intensity_max': float(intensity_vals.max()),
            'threshold_strength_mean': float(strength_vals.mean()),
            'threshold_strength_std': float(strength_vals.std(ddof=1)) if len(strength_vals) > 1 else 0.0,
            'threshold_strength_sem': float(strength_vals.sem()) if len(strength_vals) > 1 else 0.0,
            'threshold_strength_median': float(strength_vals.median()),
            'threshold_strength_iqr': float(strength_vals.quantile(0.75) - strength_vals.quantile(0.25)),
            'threshold_strength_min': float(strength_vals.min()),
            'threshold_strength_max': float(strength_vals.max()),
            'final_reversal_count_mean': float(reversal_vals.mean()) if len(reversal_vals) else None,
            'final_reversal_count_min': float(reversal_vals.min()) if len(reversal_vals) else None,
            'final_reversal_count_max': float(reversal_vals.max()) if len(reversal_vals) else None,
        })
    return pd.DataFrame(summary_rows)


def compute_pairwise_tests(wide_df, conditions):
    pairwise_results = []
    p_values = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            c1 = conditions[i]
            c2 = conditions[j]
            paired = wide_df[[c1, c2]].dropna()
            if paired.empty:
                continue
            diffs = paired[c1] - paired[c2]
            try:
                stat_w, p_val_w = stats.wilcoxon(paired[c1], paired[c2], zero_method='wilcox', alternative='two-sided', mode='auto')
            except ValueError:
                stat_w, p_val_w = np.nan, 1.0
            p_values.append(p_val_w)
            mean_diff = diffs.mean()
            sd_diff = diffs.std(ddof=1)
            rank_biserial = np.nan
            if not np.isnan(stat_w) and len(paired) > 0:
                rank_biserial = 1.0 - (2.0 * float(stat_w)) / (len(paired) * (len(paired) + 1) / 2.0)
            effect_dz = np.nan
            if sd_diff and not np.isnan(sd_diff) and sd_diff > 0:
                effect_dz = float(mean_diff / sd_diff)
            pairwise_results.append({
                'method_1': c1,
                'method_2': c2,
                'n': int(len(paired)),
                'statistic': None if np.isnan(stat_w) else float(stat_w),
                'p_value': float(p_val_w),
                'mean_diff': float(mean_diff),
                'median_diff': float(diffs.median()),
                'rank_biserial': None if np.isnan(rank_biserial) else float(rank_biserial),
                'cohens_dz': None if np.isnan(effect_dz) else float(effect_dz),
                'significance': significance_label(float(p_val_w))
            })

    if p_values:
        _, p_adj, _, _ = stats.multitest.multipletests(p_values, method='holm') if hasattr(stats, 'multitest') else (None, p_values, None, None)
    else:
        p_adj = []

    if len(p_adj) == len(pairwise_results):
        for item, adjusted in zip(pairwise_results, p_adj):
            item['p_value_holm'] = float(adjusted)
            item['significance_holm'] = significance_label(float(adjusted))
    else:
        p_sorted = [item['p_value'] for item in pairwise_results]
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_adj2, _, _ = multipletests(p_sorted, method='holm')
            for item, adjusted in zip(pairwise_results, p_adj2):
                item['p_value_holm'] = float(adjusted)
                item['significance_holm'] = significance_label(float(adjusted))
        except Exception:
            for item in pairwise_results:
                item['p_value_holm'] = item['p_value']
                item['significance_holm'] = item['significance']

    return pairwise_results


def summarize_staircase_dynamics(df, conditions):
    rows = []
    for condition in conditions:
        subset = df[df['Condition'] == condition].copy().sort_values(['ParticipantID', 'GlobalTrial'])
        if subset.empty:
            continue
        rows.append({
            'condition': condition,
            'n_trials': int(len(subset)),
            'n_participants': int(subset['ParticipantID'].nunique()),
            'accuracy_mean': float(subset['Correct'].mean()),
            'reaction_time_mean': float(subset['ReactionTime'].dropna().mean()),
            'reaction_time_median': float(subset['ReactionTime'].dropna().median()),
            'reversal_rate': float(subset['ReversalHappened'].fillna(0).mean()),
            'mean_strength_before': float(subset['StrengthBefore'].mean()),
            'mean_strength_after': float(subset['StrengthAfter'].mean()),
            'mean_intensity_before': float(subset['IntensityBefore'].mean()),
            'mean_intensity_after': float(subset['IntensityAfter'].mean()),
            'mean_trial_index_to_finish': float(subset.groupby('ParticipantID')['TrialInCondition'].max().mean()),
        })
    return pd.DataFrame(rows)


def compute_psychometric_summary(df, conditions, intensity_bins=None):
    if intensity_bins is None:
        intensity_bins = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 1.0]

    psychometric_rows = []
    for condition in conditions:
        subset = df[df['Condition'] == condition].copy()
        if subset.empty:
            continue
        subset['IntensityBin'] = pd.cut(
            subset['IntensityBefore'],
            bins=intensity_bins,
            include_lowest=True,
            right=True,
            duplicates='drop'
        )
        grouped = subset.groupby('IntensityBin', observed=False).agg(
            n=('Correct', 'size'),
            accuracy=('Correct', 'mean'),
            mean_intensity=('IntensityBefore', 'mean'),
            mean_strength=('StrengthBefore', 'mean'),
            mean_rt=('ReactionTime', 'mean')
        ).reset_index()
        grouped = grouped[grouped['n'] > 0]
        for _, row in grouped.iterrows():
            psychometric_rows.append({
                'condition': condition,
                'intensity_bin': str(row['IntensityBin']),
                'n': int(row['n']),
                'accuracy': float(row['accuracy']),
                'mean_intensity': float(row['mean_intensity']),
                'mean_strength': float(row['mean_strength']),
                'mean_reaction_time': float(row['mean_rt']) if not pd.isna(row['mean_rt']) else None,
            })
    return pd.DataFrame(psychometric_rows)


def compute_rt_summary(df, conditions, min_rt=0.2, max_rt=10.0):
    rt_df = df.copy()
    rt_df['ReactionTime'] = pd.to_numeric(rt_df['ReactionTime'], errors='coerce')
    rt_df['RTValid'] = rt_df['ReactionTime'].between(min_rt, max_rt)
    rt_df['LogRT'] = np.where(rt_df['RTValid'], np.log(rt_df['ReactionTime']), np.nan)
    subject_stats = rt_df.groupby('ParticipantID')['LogRT'].agg(rt_mean='mean', rt_std='std')
    rt_df = rt_df.join(subject_stats, on='ParticipantID')
    rt_df['ZLogRT'] = (rt_df['LogRT'] - rt_df['rt_mean']) / rt_df['rt_std']
    rt_df.loc[rt_df['rt_std'].isna() | (rt_df['rt_std'] == 0), 'ZLogRT'] = 0.0
    rt_df.loc[rt_df['LogRT'].isna(), 'ZLogRT'] = np.nan

    summary_rows = []
    for condition in conditions:
        subset = rt_df[(rt_df['Condition'] == condition) & rt_df['RTValid']].copy()
        if subset.empty:
            continue
        correct_rt = subset[subset['Correct'] == 1]['ReactionTime'].dropna()
        error_rt = subset[subset['Correct'] == 0]['ReactionTime'].dropna()
        summary_rows.append({
            'condition': condition,
            'n_valid_rt': int(len(subset)),
            'mean_rt': float(subset['ReactionTime'].mean()),
            'median_rt': float(subset['ReactionTime'].median()),
            'std_rt': float(subset['ReactionTime'].std(ddof=1)) if len(subset) > 1 else 0.0,
            'mean_log_rt': float(subset['LogRT'].mean()),
            'mean_zlogrt': float(subset['ZLogRT'].mean()),
            'mean_rt_correct': float(correct_rt.mean()) if len(correct_rt) else None,
            'mean_rt_error': float(error_rt.mean()) if len(error_rt) else None,
            'rt_correct_minus_error': float(correct_rt.mean() - error_rt.mean()) if len(correct_rt) and len(error_rt) else None,
        })

    return rt_df, pd.DataFrame(summary_rows), {
        'rt_column': 'ReactionTime',
        'min_rt': float(min_rt),
        'max_rt': float(max_rt),
        'n_total': int(len(rt_df)),
        'n_valid_range': int(rt_df['RTValid'].sum()),
        'n_invalid_range': int((~rt_df['RTValid']).sum()),
        'n_zlogrt_non_nan': int(rt_df['ZLogRT'].notna().sum()),
        'participant_stats': subject_stats.reset_index().replace({np.nan: None}).to_dict(orient='records')
    }


def summarize_participant_quality(final_trials, conditions):
    wide = final_trials.pivot(index='ParticipantID', columns='Condition', values='ThresholdEstimate')
    wide = wide.reindex(columns=conditions)
    participant_rows = []
    for participant, row in wide.iterrows():
        values = row.dropna()
        if values.empty:
            continue
        participant_rows.append({
            'ParticipantID': participant,
            'n_conditions_completed': int(values.size),
            'mean_threshold_strength': float(values.mean()),
            'std_threshold_strength': float(values.std(ddof=1)) if values.size > 1 else 0.0,
            'range_threshold_strength': float(values.max() - values.min()),
            'mean_threshold_intensity': float(values.apply(strength_to_intensity).mean()),
        })
    return pd.DataFrame(participant_rows)


def plot_threshold_bars(final_trials, summary_df, pairwise_results, conditions, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    plot_data = final_trials[final_trials['Condition'].isin(conditions)].copy()
    plot_data['Condition'] = pd.Categorical(plot_data['Condition'], categories=conditions, ordered=True)

    sns.barplot(
        data=plot_data,
        x='Condition',
        y='IntensityThreshold',
        hue='Condition',
        palette=METHOD_COLORS,
        errorbar=('ci', 95),
        capsize=0.08,
        err_kws={'linewidth': 1.4},
        ax=ax,
        alpha=0.85,
        zorder=2,
        legend=False,
        dodge=False,
        width=0.8
    )

    sns.stripplot(
        data=plot_data,
        x='Condition',
        y='IntensityThreshold',
        color='black',
        alpha=0.45,
        jitter=0.18,
        size=4.5,
        ax=ax,
        zorder=3
    )

    ax.set_ylabel("Absolute Detection Threshold\n(Normalized Intensity)", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)
    ax.set_xlabel("Modulation Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([format_method_name(c) for c in conditions])
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    ax.set_axisbelow(True)

    sig_pairs = [p for p in pairwise_results if p.get('p_value_holm', p['p_value']) < 0.05]
    sig_pairs.sort(key=lambda x: abs(conditions.index(x['method_1']) - conditions.index(x['method_2'])))

    y_max = float(plot_data['IntensityThreshold'].max()) if not plot_data.empty else 1.0
    y_min = float(plot_data['IntensityThreshold'].min()) if not plot_data.empty else 0.0
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = 0.1
    y_start = y_max + y_range * 0.08
    step = y_range * 0.1

    for pair in sig_pairs:
        idx1 = conditions.index(pair['method_1'])
        idx2 = conditions.index(pair['method_2'])
        label = pair.get('significance_holm', pair['significance'])
        ax.plot([idx1, idx1, idx2, idx2], [y_start, y_start + step * 0.2, y_start + step * 0.2, y_start], lw=1.4, c='black')
        ax.text((idx1 + idx2) / 2, y_start + step * 0.22, label, ha='center', va='bottom', color='black', fontsize=13)
        y_start += step

    ax.set_ylim(0, y_start + step * 0.7 if sig_pairs else y_max + y_range * 0.18)
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_Thresholds")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_boxplot(final_trials, conditions, output_dir):
    fig, ax = plt.subplots(figsize=(7, 8))
    plot_data = final_trials[final_trials['Condition'].isin(conditions)].copy()
    plot_data['Condition'] = pd.Categorical(plot_data['Condition'], categories=conditions, ordered=True)

    sns.boxplot(
        data=plot_data,
        x='Condition',
        y='ThresholdEstimate',
        hue='Condition',
        palette=METHOD_COLORS,
        legend=False,
        width=0.55,
        linewidth=1.3,
        fliersize=0,
        ax=ax
    )
    sns.stripplot(
        data=plot_data,
        x='Condition',
        y='ThresholdEstimate',
        color='black',
        alpha=0.4,
        jitter=0.15,
        size=4,
        ax=ax
    )
    ax.set_xlabel("Modulation Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_ylabel("Threshold Estimate (Strength)", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([format_method_name(c) for c in conditions])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_Thresholds_Strength_Boxplot")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_psychometric_curves(psychometric_df, conditions, output_dir):
    from scipy.optimize import minimize
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(6, 6))
    plotted_any = False

    # Psychometric function: Cumulative Gaussian with lapse rate
    # P(x) = gamma + (1 - gamma - lambda) * CDF((x - mu) / sigma)
    # gamma is fixed at 0.5 for 2AFC
    def psychometric_model(params, x):
        mu, sigma, lamb = params
        gamma = 0.5
        return gamma + (1 - gamma - lamb) * norm.cdf(x, loc=mu, scale=sigma)

    # Negative Log-Likelihood for MLE
    def neg_log_likelihood(params, x, n, k):
        mu, sigma, lamb = params
        # Constraints: sigma > 0, 0 <= lambda <= 0.1
        if sigma <= 0.001 or lamb < 0 or lamb > 0.1:
            return 1e9
        
        p = psychometric_model(params, x)
        # Avoid log(0)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        
        # Binomial log-likelihood
        log_lik = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
        return -log_lik

    x_fit = np.linspace(-0.1, 1.0, 300)

    for condition in conditions:
        subset = psychometric_df[psychometric_df['condition'] == condition].sort_values('mean_intensity')
        if subset.empty:
            continue
        
        plotted_any = True
        
        # Prepare data for fitting
        x_data = subset['mean_intensity'].values
        n_data = subset['n'].values
        p_data = subset['accuracy'].values
        k_data = np.round(n_data * p_data).astype(int)

        # Initial guess: mu=0.4, sigma=0.2, lambda=0.02
        initial_guess = [0.4, 0.2, 0.01]
        
        try:
            # Minimize negative log-likelihood
            result = minimize(
                neg_log_likelihood, 
                initial_guess, 
                args=(x_data, n_data, k_data),
                method='Nelder-Mead'
            )
            
            if result.success:
                best_params = result.x
                y_fit = psychometric_model(best_params, x_fit)
                
                # Plot fitted curve
                ax.plot(
                    x_fit, 
                    y_fit, 
                    linewidth=2.5, 
                    color=METHOD_COLORS[condition],
                    label=format_method_name(condition),
                    alpha=0.9
                )
                
                # Plot raw data points with size indicating weight (sample size)
                # Normalize sizes for display: min 20, max 120
                max_n = n_data.max() if n_data.max() > 0 else 1
                sizes = 20 + (n_data / max_n) * 100
                
                ax.scatter(
                    x_data,
                    p_data,
                    s=sizes,
                    color=METHOD_COLORS[condition],
                    alpha=0.5,
                    edgecolor='white',
                    linewidth=0.5,
                    zorder=2
                )
            else:
                raise RuntimeError("Optimization failed")
                
        except Exception as e:
            print(f"Curve fitting failed for {condition}: {e}. Falling back to raw data plot.")
            ax.plot(
                x_data,
                p_data,
                marker='o',
                linewidth=2.3,
                markersize=6,
                color=METHOD_COLORS[condition],
                label=format_method_name(condition)
            )

    ax.axhline(0.794, linestyle='--', linewidth=1.4, color='black', alpha=0.8)
    ax.text(0.995, 0.786, '79.4%', ha='right', va='top', fontsize=14)
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Normalized Intensity Before Response", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_ylabel("Proportion Correct", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    if plotted_any:
        ax.legend(frameon=True, ncol=2, fontsize=14, loc='lower right')
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_Psychometric_Curves")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_staircase_trajectories(df, conditions, output_dir, max_participants=6):
    sampled_participants = sorted(df['ParticipantID'].dropna().unique())[:max_participants]
    plot_df = df[df['ParticipantID'].isin(sampled_participants)].copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 2.25 * len(conditions)), sharex=False, gridspec_kw={'hspace': 0.5})
    if len(conditions) == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        subset = plot_df[plot_df['Condition'] == condition].copy().sort_values(['ParticipantID', 'TrialInCondition'])
        for participant in sampled_participants:
            psub = subset[subset['ParticipantID'] == participant]
            if psub.empty:
                continue
            ax.plot(
                psub['TrialInCondition'],
                psub['ThresholdEstimate'],
                marker='o',
                markersize=3.5,
                linewidth=1.2,
                alpha=0.65,
                color=METHOD_COLORS[condition]
            )
        reversals = subset[subset['ReversalHappened'] == 1]
        if not reversals.empty:
            ax.scatter(
                reversals['TrialInCondition'],
                reversals['ThresholdEstimate'],
                s=24,
                color='black',
                alpha=0.85,
                zorder=4
            )
        ax.set_title(format_method_name(condition), fontsize=14, fontweight='bold', loc='left')
        ax.set_ylabel("Strength", fontweight='bold')
        ax.set_ylim(0,110)
        ax.grid(True, linestyle='--', alpha=0.55)

    axes[-1].set_xlabel("Trial in Condition", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_Staircase_Trajectories")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_rt_distributions(rt_df, conditions, output_dir):
    plot_df = rt_df[(rt_df['Condition'].isin(conditions)) & (rt_df['RTValid'])].copy()
    if plot_df.empty:
        return
    plot_df['Condition'] = pd.Categorical(plot_df['Condition'], categories=conditions, ordered=True)

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    sns.violinplot(
        data=plot_df,
        x='Condition',
        y='ReactionTime',
        hue='Condition',
        palette=METHOD_COLORS,
        legend=False,
        inner='box',
        cut=0,
        linewidth=1.1,
        ax=ax,
        dodge=False,
        width=0.5
    )
    ax.set_xlabel("Modulation Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_ylabel("Reaction Time (s)", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([format_method_name(c) for c in conditions])
    ax.grid(axis='y', linestyle='--', alpha=0.65)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_ReactionTime_Distribution")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_reversal_heatmap(final_trials, conditions, output_dir):
    pivot = final_trials.pivot(index='ParticipantID', columns='Condition', values='ReversalCount')
    pivot = pivot.reindex(columns=conditions)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, max(6.2, 0.28 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Final Reversal Count'},
        ax=ax
    )
    ax.set_xlabel("Modulation Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_ylabel("Participant", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_xticklabels([format_method_name(c) for c in conditions], rotation=0)
    plt.tight_layout()

    plot_prefix = os.path.join(output_dir, "Experiment2_Reversal_Heatmap")
    plt.savefig(f"{plot_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_thresholds(df, output_dir):
    df = prepare_threshold_data(df)
    final_trials = get_final_trials(df)

    conditions = [c for c in METHOD_ORDER if c in final_trials['Condition'].unique()]
    wide_df = final_trials.pivot(index='ParticipantID', columns='Condition', values='IntensityThreshold')
    wide_df = wide_df.reindex(columns=conditions).dropna()

    if not conditions or wide_df.empty:
        raise ValueError("No complete threshold data available for analysis.")

    summary_df = build_threshold_summary(final_trials, conditions)
    staircase_summary_df = summarize_staircase_dynamics(df, conditions)
    psychometric_df = compute_psychometric_summary(df, conditions)
    rt_df, rt_summary_df, rt_meta = compute_rt_summary(df, conditions)
    participant_quality_df = summarize_participant_quality(final_trials, conditions)

    stat, p_value = stats.friedmanchisquare(*[wide_df[c] for c in conditions])
    pairwise_results = compute_pairwise_tests(wide_df, conditions)

    plot_threshold_bars(final_trials, summary_df, pairwise_results, conditions, output_dir)
    plot_threshold_boxplot(final_trials, conditions, output_dir)
    plot_psychometric_curves(psychometric_df, conditions, output_dir)
    plot_staircase_trajectories(df, conditions, output_dir)
    plot_rt_distributions(rt_df, conditions, output_dir)
    plot_reversal_heatmap(final_trials, conditions, output_dir)

    payload = {
        'analysis_overview': {
            'n_total_rows': int(len(df)),
            'n_final_trials': int(len(final_trials)),
            'n_participants_complete_cases': int(len(wide_df)),
            'conditions': conditions,
        },
        'threshold_summary': summary_df.to_dict(orient='records'),
        'friedman_test_intensity_threshold': {
            'statistic': float(stat),
            'p_value': float(p_value)
        },
        'pairwise_wilcoxon_intensity_threshold': pairwise_results,
        'staircase_dynamics_summary': staircase_summary_df.to_dict(orient='records'),
        'psychometric_summary': psychometric_df.to_dict(orient='records'),
        'reaction_time_analysis': {
            'summary': rt_meta,
            'by_condition': rt_summary_df.to_dict(orient='records')
        },
        'participant_quality': participant_quality_df.to_dict(orient='records'),
        'final_trial_records': final_trials[[
            'ParticipantID', 'Condition', 'GlobalTrial', 'TrialInCondition', 'ThresholdEstimate',
            'IntensityThreshold', 'ReversalCount', 'ReactionTime'
        ]].to_dict(orient='records'),
        'output_files': {
            'threshold_bar_pdf': os.path.join(output_dir, 'Experiment2_Thresholds.pdf'),
            'threshold_bar_svg': os.path.join(output_dir, 'Experiment2_Thresholds.svg'),
            'threshold_bar_png': os.path.join(output_dir, 'Experiment2_Thresholds.png'),
            'threshold_box_pdf': os.path.join(output_dir, 'Experiment2_Thresholds_Strength_Boxplot.pdf'),
            'psychometric_pdf': os.path.join(output_dir, 'Experiment2_Psychometric_Curves.pdf'),
            'staircase_pdf': os.path.join(output_dir, 'Experiment2_Staircase_Trajectories.pdf'),
            'rt_pdf': os.path.join(output_dir, 'Experiment2_ReactionTime_Distribution.pdf'),
            'reversal_heatmap_pdf': os.path.join(output_dir, 'Experiment2_Reversal_Heatmap.pdf')
        }
    }

    save_structured_results(output_dir, "Experiment2_detailed_results.json", payload)

    # Generate simplified summary for paper
    simplified_payload = {
        'threshold_descriptive': {
            row['condition']: {
                'mean': row['threshold_intensity_mean'],
                'std': row['threshold_intensity_std'],
                'sem': row['threshold_intensity_sem']
            } for row in summary_df.to_dict(orient='records')
        },
        'statistics': {
            'friedman_test': {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significance': significance_label(float(p_value))
            },
            'pairwise_comparisons': [
                {
                    'comparison': f"{item['method_1']} vs {item['method_2']}",
                    'p_value': item['p_value'],
                    'p_value_holm': item.get('p_value_holm'),
                    'significance': item.get('significance_holm', item['significance']),
                    'cohens_dz': item.get('cohens_dz')
                }
                for item in pairwise_results
            ]
        }
    }
    save_structured_results(output_dir, "Experiment2_summary.json", simplified_payload)

    return payload


def main():
    base_dir = r"d:\Data\OneDrive\Papers\SWIM\Codes\Experiment 2 & Supp-1"
    data_dir = os.path.join(base_dir, "Data", "Experiment_2")
    output_dir = os.path.join(base_dir, "Analysis", "Experiment_2")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df, load_logs = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    payload = analyze_thresholds(df, output_dir)
    overall_payload = {
        'data_loading': {
            'data_dir': data_dir,
            'n_rows': int(len(df)),
            'n_cols': int(df.shape[1]),
            'columns': list(df.columns),
            'load_logs': load_logs
        },
        'analysis': payload
    }
    save_structured_results(output_dir, "Experiment2_overall_results.json", overall_payload)


if __name__ == "__main__":
    main()
