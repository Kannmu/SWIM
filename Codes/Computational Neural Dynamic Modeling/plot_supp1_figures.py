from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, get_window, sosfilt
from scipy.interpolate import griddata

# =============================================================================
# Configuration
# =============================================================================

METHOD_ORDER = ['ULM_L_M0p2', 'ULM_L_M0p5', 'ULM_L_M1p0', 'ULM_L_M1p2', 'ULM_L_M2p0']
METHOD_LABELS = {
    'ULM_L_M0p2': 'Mach 0.2',
    'ULM_L_M0p5': 'Mach 0.5',
    'ULM_L_M1p0': 'Mach 1.0',
    'ULM_L_M1p2': 'Mach 1.2',
    'ULM_L_M2p0': 'Mach 2.0'
}

METHOD_COLORS = {
    'ULM_L_M0p2': '#440154',
    'ULM_L_M0p5': '#3b528b',
    'ULM_L_M1p0': '#21918c',
    'ULM_L_M1p2': '#5ec962',
    'ULM_L_M2p0': '#fde725',
}

ORTHO_COMPONENTS = ['xy', 'xz', 'yz']
COMPONENT_LABELS = {'xy': 'XY', 'xz': 'XZ', 'yz': 'YZ'}
COMPONENT_COLORS = {'xy': '#d55e00', 'xz': '#009e73', 'yz': '#0072b2'}

BASE_DIR = Path(r'd:/Data/OneDrive/Papers/SWIM')
MAT_FILE_PATH = BASE_DIR / 'Reference/Supplementary Sim 1/Outputs_Supplementary_Experiment1/supplementary_experiment1_data.mat'
MODEL_RESULTS_DIR = BASE_DIR / 'Reference/Neural Dynamics Model V2/data/results_supp1'
OUTPUT_DIR = BASE_DIR / 'Codes/Computational Neural Dynamic Modeling/Supp1'

FONT_NAME = 'Arial'
SNS_CONTEXT = 'talk'
SNS_STYLE = 'white'
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 14
LINE_WIDTH = 3.2
DPI = 300

BANDPASS_LOW = 150.0
BANDPASS_HIGH = 300.0
BANDPASS_ORDER = 4
CARRIER_FREQ = 200.0
SHEAR_SPEED = 5.0
LAMBDA_SPACE = 0.004
RECEPTOR_SPACING = 0.002
N_RASTER_NEURONS = 50
RASTER_DURATION_MS = 15.0
WATERFALL_RECEPTORS = 11
FREQ_MAX = 800.0

# =============================================================================
# Styling
# =============================================================================

def setup_style() -> None:
    sns.set_theme(context=SNS_CONTEXT, style=SNS_STYLE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME, 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    plt.rcParams['legend.fontsize'] = LEGEND_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Data loading
# =============================================================================

def ensure_time_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        return arr
    time_axis = int(np.argmax(arr.shape))
    if time_axis != 2:
        arr = np.moveaxis(arr, time_axis, 2)
    return arr

class KWaveMatLoader:
    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def _decode_utf16(dataset) -> str:
        arr = np.asarray(dataset, dtype=np.uint16).ravel()
        return ''.join(chr(v) for v in arr if v != 0)

    @staticmethod
    def _read_array(dataset):
        arr = np.asarray(dataset)
        return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))

    def load(self) -> Dict:
        methods: Dict[str, Dict] = {}
        with h5py.File(self.path, 'r') as f:
            results = f['results']
            dt = float(np.asarray(f['dt']).squeeze())
            for idx in range(results.shape[0]):
                group = f[results[idx, 0]]
                name = self._decode_utf16(group['name'])
                methods[name] = {
                    'tau_xy': ensure_time_last(self._read_array(group['tau_roi_steady_xy'])),
                    'tau_xz': ensure_time_last(self._read_array(group['tau_roi_steady_xz'])),
                    'tau_yz': ensure_time_last(self._read_array(group['tau_roi_steady_yz'])),
                    'tau_eq': ensure_time_last(self._read_array(group['tau_roi_steady'])),
                    'roi_x': np.asarray(group['roi_x_vec']).reshape(-1),
                    'roi_y': np.asarray(group['roi_y_vec']).reshape(-1),
                    't': np.asarray(group['t_vec_steady']).reshape(-1),
                }
        return {'dt': dt, 'methods': methods}

def load_model_summary() -> Dict:
    summary_path = MODEL_RESULTS_DIR / 'summary.json'
    with summary_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def load_population_outputs() -> Dict[str, Dict]:
    outputs = {}
    for method in METHOD_ORDER:
        data_path = MODEL_RESULTS_DIR / f'{method}_population_outputs.npz'
        spikes_path = MODEL_RESULTS_DIR / f'{method}_spikes.npy'
        if not data_path.exists():
            print(f"Warning: {data_path} not found.")
            continue
        data = np.load(data_path, allow_pickle=True)
        spikes = np.load(spikes_path, allow_pickle=True) if spikes_path.exists() else None
        outputs[method] = {
            'weights': data['weights'],
            'rates': data['rates'],
            'vector_strength': data['vector_strength'],
            'population_map': data['population_map'],
            'receptor_coords_m': data['receptor_coords_m'],
            'spikes': spikes,
        }
    return outputs

def load_all_data() -> Dict:
    loader = KWaveMatLoader(MAT_FILE_PATH)
    kwave = loader.load()
    summary = load_model_summary()
    population = load_population_outputs()
    return {'kwave': kwave, 'summary': summary, 'population': population}

# =============================================================================
# Mechanics / neural helpers
# =============================================================================

def compute_dynamic_components(method_data: Dict) -> Dict[str, np.ndarray]:
    out = {}
    for key in ORTHO_COMPONENTS:
        tau = np.asarray(method_data[f'tau_{key}'], dtype=np.float64)
        out[key] = tau - tau.mean(axis=2, keepdims=True)
    return out

def build_receptor_lattice(roi_x: np.ndarray, roi_y: np.ndarray, spacing_m: float = RECEPTOR_SPACING) -> Dict:
    xs = np.arange(float(np.min(roi_x)), float(np.max(roi_x)) + spacing_m * 0.5, spacing_m)
    ys = np.arange(float(np.min(roi_y)), float(np.max(roi_y)) + spacing_m * 0.5, spacing_m)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    return {'coords_m': coords, 'x_m': xs, 'y_m': ys, 'shape': gx.shape}

class CoherentIntegrator:
    def __init__(
        self,
        roi_x: np.ndarray,
        roi_y: np.ndarray,
        receptor_coords: np.ndarray,
        conduction_velocity_m_s: float,
        spatial_decay_lambda_m: float,
        dt: float,
    ):
        self.roi_x = np.asarray(roi_x, dtype=np.float64)
        self.roi_y = np.asarray(roi_y, dtype=np.float64)
        self.receptor_coords = np.asarray(receptor_coords, dtype=np.float64)
        gx, gy = np.meshgrid(self.roi_x, self.roi_y, indexing='xy')
        self.source_coords = np.column_stack([gx.ravel(), gy.ravel()])
        self.distance_m = np.linalg.norm(self.receptor_coords[:, None, :] - self.source_coords[None, :, :], axis=-1)
        self.weight = np.exp(-self.distance_m / max(float(spatial_decay_lambda_m), 1e-12))
        self.delay_steps = np.rint(self.distance_m / max(float(conduction_velocity_m_s) * float(dt), 1e-12)).astype(np.int32)

    def integrate(self, tau_dyn: np.ndarray) -> np.ndarray:
        source_signal = np.reshape(np.asarray(tau_dyn, dtype=np.float64), (-1, tau_dyn.shape[-1]))
        n_receptors = self.receptor_coords.shape[0]
        n_time = source_signal.shape[1]
        integrated = np.zeros((n_receptors, n_time), dtype=np.float64)
        time_index = np.arange(n_time, dtype=np.int32)
        for ridx in range(n_receptors):
            acc = np.zeros(n_time, dtype=np.float64)
            for sidx in range(source_signal.shape[0]):
                shifted_idx = time_index - self.delay_steps[ridx, sidx]
                valid = shifted_idx >= 0
                clipped = np.clip(shifted_idx, 0, n_time - 1)
                acc += source_signal[sidx, clipped] * valid * self.weight[ridx, sidx]
            integrated[ridx] = acc
        return integrated

def apply_pacinian_filter(signal: np.ndarray, dt: float) -> np.ndarray:
    fs = 1.0 / dt
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, signal, axis=-1)

def compute_single_sided_spectrum(signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    n = len(signal)
    win = get_window('hann', n)
    y = np.fft.rfft(signal * win, n=2 ** int(np.ceil(np.log2(n * 8))))
    f = np.fft.rfftfreq((len(y) - 1) * 2, d=dt)
    amp = np.abs(y) / n
    if amp.size > 2:
        amp[1:-1] *= 2
    return f, amp

def compute_vector_strength_from_spike_times(spike_times_s: np.ndarray, f0: float = CARRIER_FREQ) -> float:
    if spike_times_s.size == 0:
        return 0.0
    phases = np.exp(1j * 2.0 * np.pi * f0 * spike_times_s)
    return float(np.abs(np.mean(phases)))

def compute_vector_strength_from_spike_train(spike_train: np.ndarray, dt: float, f0: float = CARRIER_FREQ) -> float:
    spike_train = np.asarray(spike_train)
    if spike_train.size == 0:
        return 0.0
    spike_idx = np.flatnonzero(spike_train > 0)
    if spike_idx.size == 0:
        return 0.0
    spike_times_s = spike_idx.astype(np.float64) * float(dt)
    return compute_vector_strength_from_spike_times(spike_times_s, f0)

def choose_centerline_receptors(coords_m: np.ndarray, n_select: int = WATERFALL_RECEPTORS) -> np.ndarray:
    coords = np.asarray(coords_m, dtype=np.float64)
    y_abs = np.abs(coords[:, 1])
    center_y = np.min(y_abs)
    candidates = np.where(np.isclose(y_abs, center_y))[0]
    xs = coords[candidates, 0]
    order = np.argsort(xs)
    candidates = candidates[order]
    if len(candidates) <= n_select:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, n_select).round().astype(int)
    return candidates[positions]

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_figure1_coherent_integration(data: Dict) -> None:
    kwave = data['kwave']['methods']
    fig = plt.figure(figsize=(25, 8))
    outer = gridspec.GridSpec(1, len(METHOD_ORDER), width_ratios=[1]*len(METHOD_ORDER), wspace=0.18)

    global_max_tau = 0.0
    global_max_u = 0.0
    plot_data_cache = {}

    for method in METHOD_ORDER:
        method_data = kwave[method]
        dyn = compute_dynamic_components(method_data)['xy'] 
        lattice = build_receptor_lattice(method_data['roi_x'], method_data['roi_y'])
        receptor_idx = choose_centerline_receptors(lattice['coords_m'], WATERFALL_RECEPTORS)
        
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], 
                                        lattice['coords_m'][receptor_idx], 
                                        SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        m_drive = integrator.integrate(dyn)
        u_drive = apply_pacinian_filter(m_drive, data['kwave']['dt'])

        y_idx = int(np.argmin(np.abs(method_data['roi_y'])))
        x_indices = [int(np.argmin(np.abs(method_data['roi_x'] - x0))) for x0 in lattice['coords_m'][receptor_idx, 0]]
        tau_traces = np.stack([dyn[y_idx, xi, :] for xi in x_indices], axis=0)

        t_ms = method_data['t'] * 1000.0
        window_mask = t_ms >= (t_ms.max() - RASTER_DURATION_MS)
        
        tau_win = tau_traces[:, window_mask]
        u_win = u_drive[:, window_mask]
        
        global_max_tau = max(global_max_tau, np.max(np.abs(tau_win)))
        global_max_u = max(global_max_u, np.max(np.abs(u_win)))
        
        plot_data_cache[method] = (t_ms[window_mask], tau_win, u_win)

    offset_tau = global_max_tau * 1.5 + 1e-6
    offset_u = global_max_u * 1.2 + 1e-6

    for col, method in enumerate(METHOD_ORDER):
        t_win, tau_win, u_win = plot_data_cache[method]
        
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[col], height_ratios=[2.0, 1.0], hspace=0.15)
        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        for ridx in range(tau_win.shape[0]):
            base = ridx * offset_tau
            smooth = gaussian_filter1d(tau_win[ridx], sigma=1.0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, 0, None), color='#b40426', alpha=0.45, linewidth=0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, None, 0), color='#3b4cc0', alpha=0.45, linewidth=0)
            ax_top.plot(t_win, base + smooth, color='0.35', lw=0.9, alpha=0.9)

        for ridx in range(u_win.shape[0]):
            ax_bottom.plot(t_win, u_win[ridx] + ridx * offset_u, color='black', lw=1.6, alpha=0.9)

        method_max_tau = np.max(np.abs(tau_win))
        method_max_u = np.max(np.abs(u_win))
        gain = method_max_u / max(method_max_tau, 1e-12)

        ax_top.set_title(f'{METHOD_LABELS[method]}\nRaw shear wavefronts', fontweight='bold')
        if col == 0:
            ax_bottom.set_title(f'Effective neural drive u(t)', fontweight='bold', pad=6)
        else:
            ax_bottom.set_title(f'', fontweight='bold', pad=6)
            
        ax_top.text(0.02, 0.95, f'Effective Gain ≈ {gain:.1f}×', transform=ax_top.transAxes, 
                    ha='left', va='top', fontsize=LEGEND_SIZE, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        if col == 0:
            ax_top.set_ylabel('Virtual receptors', fontweight='bold')
            ax_bottom.set_ylabel('Filtered traces', fontweight='bold')
        
        ax_bottom.set_xlabel('Time [ms]', fontweight='bold')
        
        ax_top.set_ylim(-offset_tau, tau_win.shape[0] * offset_tau)
        ax_bottom.set_ylim(-offset_u, u_win.shape[0] * offset_u)
        
        ax_top.set_yticks(np.arange(tau_win.shape[0]) * offset_tau)
        ax_top.set_yticklabels([str(i + 1) for i in range(tau_win.shape[0])])
        ax_bottom.set_yticks([])
        ax_top.grid(False)
        ax_bottom.grid(True, linestyle='--', alpha=0.35)
        ax_top.tick_params(labelbottom=False)
        
        if col > 0:
            ax_top.tick_params(labelleft=False)

    save_figure(fig, OUTPUT_DIR / 'Supp1_Figure_Neural_1_Coherent_Integration_Dynamics')

def plot_figure2_frequency_fidelity(data: Dict) -> None:
    kwave = data['kwave']['methods']
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axvspan(BANDPASS_LOW, BANDPASS_HIGH, color='0.85', alpha=0.7, zorder=0)
    ax.text((BANDPASS_LOW + BANDPASS_HIGH) / 2.0, -180, 'Pacinian band-pass', 
            ha='center', va='bottom', fontsize=LEGEND_SIZE, fontweight='bold')

    for method in METHOD_ORDER:
        method_data = kwave[method]
        lattice = build_receptor_lattice(method_data['roi_x'], method_data['roi_y'])
        
        distances = np.linalg.norm(lattice['coords_m'], axis=1)
        center_idx = [np.argmin(distances)]
        
        dyn_all = compute_dynamic_components(method_data)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], 
                                        lattice['coords_m'][center_idx], 
                                        SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        
        best_m_drive = None
        max_energy = -1
        for comp in ORTHO_COMPONENTS:
            m_comp = integrator.integrate(dyn_all[comp])[0]
            if np.max(np.abs(m_comp)) > max_energy:
                max_energy = np.max(np.abs(m_comp))
                best_m_drive = m_comp

        u_drive = apply_pacinian_filter(best_m_drive[None, :], data['kwave']['dt'])[0]
        freqs_m, spec_m = compute_single_sided_spectrum(best_m_drive, data['kwave']['dt'])
        freqs_u, spec_u = compute_single_sided_spectrum(u_drive, data['kwave']['dt'])
        
        spec_m_db = 20.0 * np.log10(spec_m + 1e-12)
        spec_u_db = 20.0 * np.log10(spec_u + 1e-12)
        color = METHOD_COLORS[method]
        
        ax.plot(freqs_m, spec_m_db, color=color, lw=LINE_WIDTH, label=f'{METHOD_LABELS[method]} | m(t)', zorder=100)
        ax.plot(freqs_u, spec_u_db, color=color, lw=2.2, ls='--', alpha=0.8, zorder=50)

    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(-200, 80)
    ax.set_xlabel('Frequency [Hz]', fontweight='bold')
    ax.set_ylabel('PSD [dB]', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=True, ncol=2, loc='lower right')
    save_figure(fig, OUTPUT_DIR / 'Supp1_Figure_Neural_2_Frequency_Fidelity')

def plot_figure3_phase_folded(data: Dict) -> None:
    kwave = data['kwave']['methods']
    population = data['population']
    dt = float(data['kwave']['dt'])
    dt_ms = dt * 1000.0

    cycle_ms = 1000.0 / CARRIER_FREQ
    n_phase_bins = 240

    def _centers_to_edges(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size == 1:
            dv = 1.0
            return np.array([v[0] - 0.5 * dv, v[0] + 0.5 * dv], dtype=np.float64)
        mids = 0.5 * (v[:-1] + v[1:])
        first = v[0] - 0.5 * (v[1] - v[0])
        last = v[-1] + 0.5 * (v[-1] - v[-2])
        return np.concatenate([[first], mids, [last]])

    def _choose_symmetric_centerline_indices(coords_m: np.ndarray, y_round_decimals: int = 6) -> Tuple[np.ndarray, Dict]:
        coords = np.asarray(coords_m, dtype=np.float64)
        y_round = np.round(coords[:, 1], y_round_decimals)
        unique_y = np.unique(y_round)
        center_y = unique_y[np.argmin(np.abs(unique_y))]

        row_idx = np.where(np.isclose(y_round, center_y, atol=10 ** (-y_round_decimals)))[0]
        row_idx = row_idx[np.argsort(coords[row_idx, 0])]
        x_row_mm = coords[row_idx, 0] * 1000.0

        dx_mm = np.diff(x_row_mm)
        median_dx_mm = float(np.median(np.abs(dx_mm))) if dx_mm.size > 0 else 2.0
        zero_tol_mm = 0.55 * median_dx_mm

        left_local = np.where(x_row_mm < -0.5 * zero_tol_mm)[0]
        right_local = np.where(x_row_mm > 0.5 * zero_tol_mm)[0]
        center_local = int(np.argmin(np.abs(x_row_mm)))
        has_center = abs(float(x_row_mm[center_local])) <= zero_tol_mm

        left_near = left_local[np.argsort(np.abs(x_row_mm[left_local]))]
        right_near = right_local[np.argsort(np.abs(x_row_mm[right_local]))]
        n_pairs = int(min(len(left_near), len(right_near)))

        left_keep = left_near[:n_pairs]
        right_keep = right_near[:n_pairs]

        chosen_local = list(left_keep)
        if has_center:
            chosen_local.append(center_local)
        chosen_local.extend(list(right_keep))
        chosen_local = np.array(sorted(set(chosen_local), key=lambda k: x_row_mm[k]), dtype=int)

        selected_idx = row_idx[chosen_local]
        return selected_idx, {}

    def _phase_fold_map(signal_xt: np.ndarray, t_rel_ms: np.ndarray, period_ms: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        signal_xt = np.asarray(signal_xt, dtype=np.float64)
        t_rel_ms = np.asarray(t_rel_ms, dtype=np.float64)

        phase_edges = np.linspace(0.0, period_ms, n_bins + 1)
        phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

        phase = np.mod(t_rel_ms, period_ms)
        bin_idx = np.floor(phase / period_ms * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        folded = np.full((signal_xt.shape[0], n_bins), np.nan, dtype=np.float64)

        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                folded[:, b] = np.mean(signal_xt[:, mask], axis=1)

        if np.any(~np.isfinite(folded)):
            for i in range(folded.shape[0]):
                row = folded[i]
                good = np.isfinite(row)
                if np.sum(good) >= 2:
                    folded[i] = np.interp(np.arange(n_bins), np.where(good)[0], row[good])
                elif np.sum(good) == 1:
                    folded[i] = row[good][0]
                else:
                    folded[i] = 0.0

        return folded, phase_centers

    cache = {}
    all_values = []
    all_profiles = []
    global_abs_x_mm = 0.0

    for method in METHOD_ORDER:
        method_data = kwave[method]
        coords = np.asarray(population[method]['receptor_coords_m'], dtype=np.float64)

        selected_idx, _ = _choose_symmetric_centerline_indices(coords)
        line_coords = coords[selected_idx]
        x_mm = line_coords[:, 0] * 1000.0
        global_abs_x_mm = max(global_abs_x_mm, float(np.max(np.abs(x_mm))))

        t_ms_full = np.asarray(method_data['t'], dtype=np.float64) * 1000.0
        t_end = float(t_ms_full.max())
        window_mask = (t_ms_full >= (t_end - RASTER_DURATION_MS - 0.5 * dt_ms)) & (t_ms_full < (t_end - 0.5 * dt_ms))
        t_win_ms = t_ms_full[window_mask]
        t_rel_ms = t_win_ms - t_win_ms[0]

        dyn = compute_dynamic_components(method_data)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], line_coords, SHEAR_SPEED, LAMBDA_SPACE, dt)

        spike_array = np.asarray(population[method]['spikes'])
        
        component_weight_maps = []
        component_profiles = []
        for comp_idx, comp in enumerate(ORTHO_COMPONENTS):
            m_comp = integrator.integrate(dyn[comp])
            u_comp = apply_pacinian_filter(m_comp, dt)
            u_win = u_comp[:, window_mask]
            r_comp = np.maximum(u_win, 0.0)

            spike_comp = spike_array[comp_idx]
            vs_all = np.array([compute_vector_strength_from_spike_train(spike_comp[idx], dt, CARRIER_FREQ) for idx in range(spike_comp.shape[0])], dtype=np.float64)
            vs_sel = vs_all[selected_idx]
            weight_map = r_comp * vs_sel[:, None]
            weight_profile = np.max(weight_map, axis=1)

            component_weight_maps.append(weight_map)
            component_profiles.append(weight_profile)

        component_weight_maps = np.stack(component_weight_maps, axis=0)
        component_profiles = np.stack(component_profiles, axis=0)

        final_weight_map = np.maximum.reduce(component_weight_maps, axis=0)
        final_profile = np.max(component_profiles, axis=0)

        folded_map, phase_centers_ms = _phase_fold_map(final_weight_map, t_rel_ms, cycle_ms, n_phase_bins)

        cache[method] = {
            'x_mm': x_mm,
            'phase_centers_ms': phase_centers_ms,
            'folded_map': folded_map,
            'final_profile': final_profile,
        }

        all_values.append(folded_map.ravel())
        all_profiles.append(final_profile.ravel())

    all_values = np.concatenate(all_values)
    all_profiles = np.concatenate(all_profiles)

    vmax = float(np.nanpercentile(all_values, 99.5))
    if not np.isfinite(vmax) or vmax <= 0: vmax = float(np.nanmax(all_values))
    if vmax <= 0: vmax = 1.0

    profile_max = float(np.nanpercentile(all_profiles, 99.5))
    if not np.isfinite(profile_max) or profile_max <= 0: profile_max = float(np.nanmax(all_profiles))
    if profile_max <= 0: profile_max = 1.0

    fig = plt.figure(figsize=(12.8, 14.0))
    gs = gridspec.GridSpec(len(METHOD_ORDER), 3, width_ratios=[6.8, 2.0, 0.18], hspace=0.10, wspace=0.12)

    heat_axes = []
    profile_axes = []
    mappable = None

    for row, method in enumerate(METHOD_ORDER):
        ax_h = fig.add_subplot(gs[row, 0], sharex=heat_axes[0] if len(heat_axes) > 0 else None, sharey=heat_axes[0] if len(heat_axes) > 0 else None)
        ax_p = fig.add_subplot(gs[row, 1], sharex=profile_axes[0] if len(profile_axes) > 0 else None, sharey=ax_h)

        heat_axes.append(ax_h)
        profile_axes.append(ax_p)

        x_mm = cache[method]['x_mm']
        folded_map = cache[method]['folded_map']
        phase_centers_ms = cache[method]['phase_centers_ms']
        final_profile = cache[method]['final_profile']

        x_edges = _centers_to_edges(x_mm)
        phase_edges = np.linspace(0.0, cycle_ms, len(phase_centers_ms) + 1)

        mappable = ax_h.pcolormesh(phase_edges, x_edges, folded_map, shading='auto', cmap='magma', vmin=0.0, vmax=vmax)

        ax_h.axhline(0.0, color='white', linestyle='--', lw=1.0, alpha=0.55)
        ax_h.set_ylim(-global_abs_x_mm, global_abs_x_mm)
        ax_h.grid(False)

        ax_h.text(0.015, 0.92, METHOD_LABELS[method], transform=ax_h.transAxes, ha='left', va='top', color='white', fontsize=LEGEND_SIZE, fontweight='bold', bbox=dict(facecolor='black', alpha=0.20, edgecolor='none', pad=2.5))

        if row < len(METHOD_ORDER) - 1:
            ax_h.tick_params(labelbottom=False)
        else:
            ax_h.set_xlabel('Phase within one 200 Hz cycle [ms]', fontweight='bold')

        ax_p.plot(final_profile, x_mm, color=METHOD_COLORS[method], lw=LINE_WIDTH)
        ax_p.fill_betweenx(x_mm, 0.0, final_profile, color=METHOD_COLORS[method], alpha=0.22)
        ax_p.axhline(0.0, color='0.35', linestyle='--', lw=0.9, alpha=0.60)
        ax_p.set_xlim(0.0, profile_max * 1.05)
        ax_p.grid(True, axis='x', linestyle='--', alpha=0.35)
        ax_p.tick_params(labelleft=False)

        if row < len(METHOD_ORDER) - 1:
            ax_p.tick_params(labelbottom=False)
        else:
            ax_p.set_xlabel('Final readout weight $w_i$ [a.u.]', fontweight='bold')

        if row == 0:
            ax_h.set_title(r'Phase-folded centerline weight map $r(x,\phi) \times VS_{200\,\mathrm{Hz}}$', fontweight='bold', pad=10)
            ax_p.set_title('Final readout\n'+r'$w_i = \max(w_{xy}, w_{xz}, w_{yz})$', fontweight='bold', pad=10)

    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r'Phase-folded weight $r \times VS_{200\,\mathrm{Hz}}$ [a.u.]', fontweight='bold')
    cbar.outline.set_linewidth(1.0)

    fig.text(0.02, 0.5, 'Centerline receptor position x [mm]', rotation=90, va='center', ha='center', fontweight='bold')

    save_figure(fig, OUTPUT_DIR / 'Supp1_Figure_Neural_3_PhaseFolded_Centerline_DriveMagnitude')

def plot_figure4_max_pooling(data: Dict) -> None:
    kwave = data['kwave']['methods']
    population = data['population']
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(25, 6.4), sharey=True, gridspec_kw={'wspace': 0.1})

    for ax, method in zip(axes, METHOD_ORDER):
        method_data = kwave[method]
        coords = population[method]['receptor_coords_m']
        lattice_center_idx = choose_centerline_receptors(coords, len(np.unique(np.round(coords[:, 0], 6))))
        x_coords = coords[lattice_center_idx, 0]
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]

        dyn = compute_dynamic_components(method_data)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], coords[lattice_center_idx], SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        comp_weights = []
        for key in ORTHO_COMPONENTS:
            m_drive = integrator.integrate(dyn[key])
            u_drive = apply_pacinian_filter(m_drive, data['kwave']['dt'])
            positive = np.maximum(u_drive, 0.0)
            weight_line = positive.mean(axis=1)
            comp_weights.append(weight_line[sort_idx])
            ax.plot(x_coords * 1000.0, weight_line[sort_idx], color=COMPONENT_COLORS[key], lw=5, label=COMPONENT_LABELS[key])

        envelope = np.maximum.reduce(comp_weights)
        ax.plot(x_coords * 1000.0, envelope, color='black', lw=5.0, ls='--', label='Max-pooling envelope', alpha=0.8)
        ax.fill_between(x_coords * 1000.0, 0, envelope, color='0.7', alpha=0.15)
        ax.set_title(f'{METHOD_LABELS[method]}', fontweight='bold')
        ax.set_xlabel('x [mm]', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.35)

    axes[0].set_ylabel('Neural drive weight [a.u.]', fontweight='bold')
    handles = [Line2D([0], [0], color=COMPONENT_COLORS[k], lw=LINE_WIDTH, label=COMPONENT_LABELS[k]) for k in ORTHO_COMPONENTS]
    handles.append(Line2D([0], [0], color='black', lw=3.0, ls='--', label='Max-pooling envelope'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, OUTPUT_DIR / 'Supp1_Figure_Neural_4_Directional_Max_Pooling')

def plot_population_maps(data: Dict) -> None:
    population = data['population']
    grid_res = 100 
    high_res_maps = []
    extents = None
    
    for m in METHOD_ORDER:
        method_pop = population[m]
        coords = method_pop['receptor_coords_m']
        weights = np.asarray(method_pop['weights'], dtype=np.float64)
        
        if extents is None:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            extents = [x_min * 1000, x_max * 1000, y_min * 1000, y_max * 1000]
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, grid_res),
                np.linspace(y_min, y_max, grid_res)
            )
            
        map_hr = griddata(
            points=(coords[:, 0], coords[:, 1]), 
            values=weights, 
            xi=(grid_x, grid_y), 
            method='cubic', 
            fill_value=np.min(weights)
        )
        high_res_maps.append(map_hr)
        
    vmax = max(np.nanmax(sm) for sm in high_res_maps)
    vmin = min(np.nanmin(sm) for sm in high_res_maps)
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 4), gridspec_kw={'wspace': 0.3})
    
    for ax, method, sm in zip(axes, METHOD_ORDER, high_res_maps):
        im = ax.imshow(sm, cmap='Greens', vmin=0, vmax=vmax, 
                       origin='lower', extent=extents, aspect='equal')
        
        local_min = np.nanmin(sm)
        local_max = np.nanmax(sm)
        relative_thr = local_min + 0.5 * (local_max - local_min)
        
        if (local_max - local_min) > (0.1 * vmax):
            X_hr, Y_hr = np.meshgrid(
                np.linspace(extents[0], extents[1], grid_res),
                np.linspace(extents[2], extents[3], grid_res)
            )
            ax.contour(X_hr, Y_hr, sm, levels=[relative_thr], colors='white', linewidths=2.0, alpha=0.9)
            
        ax.set_title(METHOD_LABELS[method], fontweight='bold', fontsize=TITLE_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.75, aspect=25, pad=0.05)
    cbar.set_label('Population weight [a.u.]', fontweight='bold', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(1.0)
    
    save_figure(fig, OUTPUT_DIR / 'Supp1_Figure_Neural_5_Population_Weight_Maps')

def plot_intensity_scores(data: Dict) -> None:
    summary = data['summary']
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mach_values = [0.2, 0.5, 1.0, 1.2, 2.0]
    scores = [summary['methods'][m]['intensity_score'] for m in METHOD_ORDER]
    
    ax.plot(mach_values, scores, marker='o', markersize=10, lw=3, color='#2c3e50')
    
    for i, m in enumerate(METHOD_ORDER):
        ax.plot(mach_values[i], scores[i], marker='o', markersize=12, 
                color=METHOD_COLORS[m], markeredgecolor='black', zorder=5)

    ax.set_xlabel('Mach Number (v/c)', fontweight='bold')
    ax.set_ylabel('Predicted Intensity Score [a.u.]', fontweight='bold')
    ax.set_title('Supplementary Experiment 1 | Intensity vs. Scan Speed', fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(mach_values)
    
    save_figure(fig, OUTPUT_DIR / 'Supp1_Intensity_vs_Mach')

def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=DPI)
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        data = load_all_data()
        plot_figure1_coherent_integration(data)
        plot_figure2_frequency_fidelity(data)
        plot_figure3_phase_folded(data)
        plot_figure4_max_pooling(data)
        plot_population_maps(data)
        plot_intensity_scores(data)
        
        print(f'Supplementary experiment 1 figures generated in {OUTPUT_DIR}')
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    main()
