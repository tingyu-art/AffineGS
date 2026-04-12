"""
flow_classifier.py
==================
Training-free flow type classifier from event data.

5 output classes
----------------
    linear     : uniform / constant-velocity linear flow
    solid_rot  : solid body rotation
    vortex     : vortex-type flow (cellular / Lamb-Oseen / Taylor-Green)
    shear      : shear / sinusoidal linear flow
    turbulence : isotropic turbulence

Coordinate convention
---------------------
    events[:, 0] = h  (row,    vertical)
    events[:, 1] = w  (column, horizontal)

Usage
-----
    from flow_classifier import classify_flow_from_events

    result = classify_flow_from_events(events, H=256, W=256)
    print(result.label)        # 'linear'|'solid_rot'|'vortex'|'shear'|'turbulence'
    print(result.confidence)   # float in [0, 1]
    print(result)              # full text summary

Decision tree (in order)
------------------------
    unif > 0.40                 ->  linear
    axial_asymmetry > 0.70      ->  shear
    ring < 0.55 AND rpk < 0.80  ->  turbulence
    radial_peak < 0.20          ->  solid_rot
    else                        ->  vortex
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict


# ──────────────────────────────────────────────────────────────────────────────
#  Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowTypeResult:
    """
    Attributes
    ----------
    label      : predicted class
                 'linear' | 'solid_rot' | 'vortex' | 'shear' | 'turbulence'
    confidence : softmax score of the predicted class  [0, 1]
    scores     : softmax scores for all 5 classes
    features   : the 4 discriminative feature values
    """
    label:      str
    confidence: float
    scores:     Dict[str, float]
    features:   Dict[str, float]

    def __str__(self):
        lines = [
            f"Prediction : {self.label}",
            f"Confidence : {self.confidence:.1%}",
            "-" * 56,
            "Scores:",
        ]
        for name, score in sorted(self.scores.items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
            lines.append(f"  {name:<12} {score:.3f}  {bar}")
        lines += ["-" * 56, "Key features:"]
        desc = {
            'density_uniformity': 'density CV inverse  high=uniform/linear  low=structured',
            'radial_peak':        'radial mid-peak      low=solid_rot/linear high=vortex',
            'ring_structure':     'ring uniformity      low=turbulence       high=others',
            'axial_asymmetry':    'h/w power asymmetry  high=shear           low=isotropic',
        }
        for name, val in self.features.items():
            bar  = "#" * int(val * 16) + "." * (16 - int(val * 16))
            lines.append(f"  {name:<22} {val:.4f}  {bar}  {desc.get(name, '')}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Internal utilities
# ──────────────────────────────────────────────────────────────────────────────

def _density_map(events: torch.Tensor, H: int, W: int,
                 smooth_k: int = 9) -> torch.Tensor:
    device = events.device
    d  = torch.zeros(H, W, device=device)
    hs = events[:, 0].long().clamp(0, H - 1)
    ws = events[:, 1].long().clamp(0, W - 1)
    d.index_put_((hs, ws), torch.ones(len(events), device=device), accumulate=True)
    if smooth_k > 1:
        d = F.avg_pool2d(d[None, None], smooth_k, stride=1,
                         padding=smooth_k // 2).squeeze()
    return d


def _centroid(density: torch.Tensor):
    H, W  = density.shape
    dev   = density.device
    h_idx = torch.arange(H, device=dev).float()
    w_idx = torch.arange(W, device=dev).float()
    total = density.sum() + 1e-8
    ch    = (density * h_idx[:, None]).sum() / total
    cw    = (density * w_idx[None, :]).sum() / total
    return ch, cw


# ──────────────────────────────────────────────────────────────────────────────
#  4 discriminative features
# ──────────────────────────────────────────────────────────────────────────────

def _feat_density_uniformity(density: torch.Tensor) -> float:
    """
    Spatial uniformity of the event density map.

    Uniform / constant-velocity linear flow: all particles move at the same
    speed, triggering events at equal rates everywhere -> density is nearly
    flat -> very low coefficient of variation (CV = std / mean).

    All other flows have spatial structure (rings, gradients, vortices) which
    raises the CV significantly.

    Score = exp(-k * CV), so:
        CV -> 0   (uniform flow) : score -> 1.0
        CV -> inf (structured)   : score -> 0.0

    The decay constant k=3 is calibrated so that:
        linear flow  CV ~ 0.05-0.15  ->  score ~ 0.64-0.86  (high)
        solid_rot    CV ~ 0.5-0.8    ->  score ~ 0.09-0.22  (low)
        shear        CV ~ 0.4-0.6    ->  score ~ 0.16-0.30  (low)
        turbulence   CV ~ 0.6-1.0    ->  score ~ 0.05-0.16  (low)
        vortex       CV ~ 0.4-0.7    ->  score ~ 0.12-0.25  (low)
    """
    mean = density.mean()
    if mean < 1e-6:
        return 0.5
    cv = (density.std() / (mean + 1e-8)).item()
    return float(np.exp(-3.0 * cv))


def _feat_radial_peak(density: torch.Tensor, n_bins: int = 20) -> float:
    """
    Score for an intermediate peak in the radial density profile.

    solid_rot: monotonically increasing with r -> score ~ 0.
    linear:    flat radial profile (no interior peak) -> score ~ 0.
    vortex:    clear peak at interior radius -> score high.
    turbulence: scattered peaks -> moderate score.

    Calibrated values:
        linear=0.00  solid_rot=0.00  cellular=0.86  lamb=0.95
        turbulence=0.45  shear=0.00
    """
    H, W   = density.shape
    dev    = density.device
    ch, cw = _centroid(density)

    h_idx  = torch.arange(H, device=dev).float()
    w_idx  = torch.arange(W, device=dev).float()
    hh, ww = torch.meshgrid(h_idx, w_idx, indexing='ij')
    r      = torch.sqrt((hh - ch) ** 2 + (ww - cw) ** 2).reshape(-1)
    d_flat = density.reshape(-1)

    r_max = min(H, W) * 0.45
    valid = r < r_max
    r_v   = r[valid];  d_v = d_flat[valid]

    edges   = torch.linspace(0, r_max, n_bins + 1, device=dev)
    profile = []
    for i in range(n_bins):
        m = (r_v >= edges[i]) & (r_v < edges[i + 1])
        profile.append(d_v[m].mean().item() if m.sum() > 10 else 0.0)

    p           = np.array(profile)
    p          /= (p.max() + 1e-8)
    peak_idx    = int(np.argmax(p))
    is_interior = 2 <= peak_idx <= n_bins - 3
    edge_mean   = (p[:2].mean() + p[-2:].mean()) / 2
    score       = float(p[peak_idx] - edge_mean) * float(is_interior)
    return float(np.clip(score * 2.0, 0.0, 1.0))


def _feat_ring_structure(density: torch.Tensor) -> float:
    """
    Minimum azimuthal CV over 3 radial bands x 16 sectors.

    Turbulence: no coherent ring -> high CV -> low score.
    Linear: flat density also appears azimuthally uniform, but this case
            is caught earlier by density_uniformity.
    Others (rotation, vortex, shear): moderate-to-high ring uniformity.

    Calibrated values:
        solid_rot=0.71  cellular=0.65  lamb=0.79
        turbulence=0.34  shear=0.52
    """
    H, W   = density.shape
    dev    = density.device
    ch, cw = _centroid(density)

    h_idx  = torch.arange(H, device=dev).float()
    w_idx  = torch.arange(W, device=dev).float()
    hh, ww = torch.meshgrid(h_idx, w_idx, indexing='ij')
    r      = torch.sqrt((hh - ch) ** 2 + (ww - cw) ** 2)
    theta  = torch.atan2(hh - ch, ww - cw)

    r_max   = min(H, W) * 0.45
    r_bands = [(r_max * 0.05, r_max * 0.35),
               (r_max * 0.15, r_max * 0.55),
               (r_max * 0.30, r_max * 0.70)]
    n_ang   = 16
    edges   = torch.linspace(-float(torch.pi), float(torch.pi), n_ang + 1, device=dev)

    min_cv = 1.0
    for r_lo, r_hi in r_bands:
        ring_mask    = (r >= r_lo) & (r <= r_hi)
        sector_means = []
        for i in range(n_ang):
            m = ring_mask & (theta >= edges[i]) & (theta < edges[i + 1])
            if m.sum() > 10:
                sector_means.append(density[m].mean().item())
        if len(sector_means) >= 10:
            arr    = np.array(sector_means)
            cv     = arr.std() / (arr.mean() + 1e-8)
            min_cv = min(min_cv, cv)

    return float(np.clip(1.0 - min_cv * 3.0, 0.0, 1.0))


def _feat_axial_asymmetry(density: torch.Tensor) -> float:
    """
    Absolute spectral power difference between h-projection and w-projection.
    Shear flow concentrates energy along one spatial axis -> high score.

    Calibrated values:
        linear=0.05  solid_rot=0.04  cellular=0.08  lamb=0.00
        turbulence=0.06  shear=0.89
    """
    ph    = torch.fft.rfft(density.mean(dim=1)).abs() ** 2
    pw    = torch.fft.rfft(density.mean(dim=0)).abs() ** 2
    ph[0] = 0;  pw[0] = 0
    total = ph.sum() + pw.sum() + 1e-8
    return float((torch.abs(ph.sum() - pw.sum()) / total).item())


# ──────────────────────────────────────────────────────────────────────────────
#  Decision tree
# ──────────────────────────────────────────────────────────────────────────────

_LABELS = ['linear', 'solid_rot', 'vortex', 'shear', 'turbulence']

# Calibration (12 flows, smooth_k=9):
#   Flow         unif   rpk   ring   asym   -> label
#   uniform      0.460  0.000  0.759  0.075  linear
#   solid_rot    0.223  0.000  0.619  0.135  solid_rot
#   cellular     0.298  0.983  0.755  0.132  vortex
#   lamb         0.371  1.000  0.699  0.135  vortex
#   gs_flow      0.175  0.397  0.127  0.380  turbulence
#   JHTDB_iso    0.176  0.310  0.242  0.074  turbulence
#   DNS_turb     0.147  0.521  0.333  0.252  turbulence
#   SQG          0.156  0.503  0.088  0.061  turbulence
#   channel      0.149  0.296  0.197  0.201  turbulence
#   MHD          0.125  0.000  0.000  0.286  turbulence
#   shear        0.241  0.000  0.533  0.924  shear
#   backstep     0.116  0.427  0.174  0.792  shear
#
# Decision rules and margins:
#   linear     : unif>0.40 only  (uniform=0.486, next=lamb 0.342, margin=0.058)
#                  rpk condition dropped: uniform rpk is stochastic (0.000-0.324)
#   shear      : asym>0.70  (shear=0.941, backstep=0.799; others<=0.566, margin=0.13)
#   turbulence : ring<0.55 AND rpk<0.80
#                  worst-case turb: DNS_turb ring=0.422 rpk=0.699  both pass
#                  vortex min: cellular ring=0.685 rpk=0.950  both fail
#   solid_rot  : rpk<0.20  solid=0.000, others caught earlier
#   vortex     : default

def _rule_classify(unif: float, rpk: float, ring: float, asym: float):
    """
    Returns (label, class_index).
    Rules applied in priority order; first match wins.
    """
    if unif > 0.40:                  return 'linear',     0  # uniform=0.486, others<=0.342
    if asym > 0.70:                  return 'shear',      3  # shear=0.941 backstep=0.799
    if ring < 0.55 and rpk < 0.80:  return 'turbulence', 4  # DNS rpk=0.699, vortex min=0.950
    if rpk  < 0.20:                  return 'solid_rot',  1  # solid=0.000
    return                           'vortex',    2          # cellular=0.950 lamb=0.973


# Softmax weight matrix — confidence scores only, not used for the decision.
# Column order: unif, rpk, ring, asym
_W = np.array([
    [ 1.00, -0.30,  0.10, -0.20],   # linear     : high uniformity
    [-0.40, -0.80,  0.40, -0.20],   # solid_rot  : low rpk, moderate ring
    [-0.30,  0.60,  0.20, -0.20],   # vortex     : high rpk
    [-0.20, -0.20, -0.30,  1.00],   # shear      : high asym
    [-0.20,  0.10, -0.80, -0.10],   # turbulence : low ring
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def classify_flow_from_events(
    events:   torch.Tensor,
    H:        int  = 256,
    W:        int  = 256,
    smooth_k: int  = 9,
    verbose:  bool = True,
) -> FlowTypeResult:
    """
    Classify flow type from a segment of event data.

    Parameters
    ----------
    events   : Tensor (N, 4), columns [h, w, t, p]
               h = row (0 = top), w = column (0 = left)
    H, W     : sensor resolution in pixels
    smooth_k : density-map Gaussian smoothing kernel size (recommended 7-11)
               Larger values reduce noise in the density map but blur fine
               structure.  For very short event segments, increase to 15+.
    verbose  : if True, print the result summary to stdout

    Returns
    -------
    FlowTypeResult
        .label      : 'linear' | 'solid_rot' | 'vortex' | 'shear' | 'turbulence'
        .confidence : softmax score of the predicted class  [0, 1]
        .scores     : dict of softmax scores for all 5 classes
        .features   : dict of the 4 key feature values

    Examples
    --------
    >>> result = classify_flow_from_events(events, H=256, W=256)
    >>> print(result.label, result.confidence)

    >>> result = classify_flow_from_events(events, verbose=False)
    >>> if result.label == 'turbulence':
    ...     cfg.w_div = 5.0   # relax divergence regularisation for turbulence
    """
    density = _density_map(events, H, W, smooth_k=smooth_k)

    unif = _feat_density_uniformity(density)
    rpk  = _feat_radial_peak(density)
    ring = _feat_ring_structure(density)
    asym = _feat_axial_asymmetry(density)

    label, best_idx = _rule_classify(unif, rpk, ring, asym)

    f_vec  = np.array([unif, rpk, ring, asym])
    raw    = _W @ f_vec
    raw   -= raw.max()
    exp_s  = np.exp(raw / 0.3)
    soft_s = exp_s / exp_s.sum()
    # Guarantee rule result is ranked first
    soft_s[best_idx] = max(float(soft_s[best_idx]), float(soft_s.max()) + 1e-3)
    soft_s /= soft_s.sum()

    result = FlowTypeResult(
        label      = label,
        confidence = float(soft_s[best_idx]),
        scores     = {_LABELS[i]: float(soft_s[i]) for i in range(5)},
        features   = {'density_uniformity': unif,
                      'radial_peak':        rpk,
                      'ring_structure':     ring,
                      'axial_asymmetry':    asym},
    )
    if verbose:
        print(result)
    return result

def diagnose(
    events:   torch.Tensor,
    H:        int  = 256,
    W:        int  = 256,
    smooth_k: int  = 9,
    label:    str  = "",
) -> dict:
    """
    Print raw feature values for a single event segment.
    Use this to recalibrate decision thresholds when misclassification occurs.

    Parameters
    ----------
    events   : Tensor (N, 4), columns [h, w, t, p]
    label    : optional true label string, shown in the header
    smooth_k : must match the value used in classify_flow_from_events()

    Returns
    -------
    dict of feature values

    Examples
    --------
    >>> # When turbulence is misclassified as vortex:
    >>> diagnose(turb_events, label="turbulence (misclassified as vortex)")
    >>> diagnose(vortex_events, label="vortex (correct)")
    """
    density = _density_map(events, H, W, smooth_k=smooth_k)

    unif = _feat_density_uniformity(density)
    rpk  = _feat_radial_peak(density)
    ring = _feat_ring_structure(density)
    asym = _feat_axial_asymmetry(density)

    feats = {
        "density_uniformity": unif,
        "radial_peak":        rpk,
        "ring_structure":     ring,
        "axial_asymmetry":    asym,
    }

    title = f"diagnose: {label}" if label else "diagnose"
    print(f"\n{'='*54}")
    print(f"  {title}")
    print(f"  N={len(events)}  smooth_k={smooth_k}")
    print(f"{'='*54}")

    thresholds = {
        "density_uniformity": (">", 0.70, "-> linear"),
        "axial_asymmetry":    (">", 0.70, "-> shear"),
        "ring_structure":     ("<", 0.55, "-> turb (also needs rpk<0.65)"),
        "radial_peak":        ("<", 0.65, "-> turb (also needs ring<0.55)"),
        "radial_peak":        ("<", 0.20, "-> solid_rot"),
    }

    # Print each feature with its value, bar, and whether it triggers a rule
    rules = [
        ("density_uniformity", ">",  0.70, "linear"),
        ("axial_asymmetry",    ">",  0.70, "shear"),
        ("ring_structure",     "<",  0.55, "turb-AND"),
        ("radial_peak",        "<",  0.65, "turb-AND"),
        ("radial_peak",        "<",  0.20, "solid_rot"),
    ]

    print(f"  {'feature':<22}  {'value':>7}  {'bar':<20}  rule")
    print(f"  {'-'*68}")
    printed = set()
    for fname, op, thresh, rule_tag in rules:
        val = feats[fname]
        bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
        if op == ">":
            fires = val > thresh
            cmp   = f"> {thresh:.2f}"
        else:
            fires = val < thresh
            cmp   = f"< {thresh:.2f}"
        mark = f"  FIRES -> {rule_tag}" if fires else ""
        key  = (fname, thresh)
        if key not in printed:
            print(f"  {fname:<22}  {val:>7.4f}  {bar}  {cmp}{mark}")
            printed.add(key)

    print(f"  {'-'*68}")

    # Simulate decision tree
    label_pred, _ = _rule_classify(unif, rpk, ring, asym)
    turb_cond = ring < 0.55 and rpk < 0.65
    print(f"  turbulence joint condition  ring<0.55={ring<0.55}  rpk<0.65={rpk<0.65}  -> fires={turb_cond}")
    print(f"  predicted: {label_pred}")
    print(f"{'='*54}")

    return feats


def diagnose_all(
    flow_specs: dict,
    time_length: float = 3.0,
    density:     float = 0.05,
    H:           int   = 256,
    W:           int   = 256,
    smooth_k:    int   = 9,
    device:      str   = 'cpu',
) -> dict:
    """
    Run all flows, compute features, print a calibration table.

    Parameters
    ----------
    flow_specs : dict  {label: FlowManager instance}
    time_length, density : event simulation parameters

    Returns
    -------
    dict {label: {feature: value}}

    Examples
    --------
    >>> import numpy as np
    >>> from generator.flow import FlowManager
    >>> from flow_classifier import diagnose_all
    >>>
    >>> max_v = 5
    >>> flows = {
    ...     "cellular":  FlowManager(mode="analytical", name="cellular_flow", vmax=max_v, p=256),
    ...     "lamb":      FlowManager(mode="analytical", name="lamb_oseen", x_c=128, y_c=128, Gamma=3e3, rc=60),
    ...     "shear":     FlowManager(mode="analytical", name="shear_flow", vmax=max_v, p=256),
    ...     "JHTDB":     FlowManager(mode="data", data_path="./data/JHTDB_isotropic1024_hd_00001_flow.npz"),
    ...     "solid_rot": FlowManager(mode="analytical", name="solid_rot", x_c=128, y_c=128, omega=0.005525*max_v),
    ...     "uniform":   FlowManager(mode="analytical", name="uniform", uc=max_v/np.sqrt(2), vc=max_v/np.sqrt(2)),
    ... }
    >>> all_feats = diagnose_all(flows)
    """
    from generator.images import config_events
    from generator.loader import EventManager
    import numpy as np

    cfg             = config_events()
    cfg.time_length = time_length
    cfg.d           = 0.5
    cfg.density     = density

    FEAT_NAMES = ["density_uniformity", "radial_peak", "ring_structure", "axial_asymmetry"]

    all_feats = {}
    for label, flow in flow_specs.items():
        loader  = EventManager(source_type="simulator", path=None,
                               cfg=cfg, flow_func=flow.evaluate)
        evts, _ = loader.load()
        evts    = evts.to(device)
        density_map = _density_map(evts, H, W, smooth_k=smooth_k)
        all_feats[label] = {
            "density_uniformity": _feat_density_uniformity(density_map),
            "radial_peak":        _feat_radial_peak(density_map),
            "ring_structure":     _feat_ring_structure(density_map),
            "axial_asymmetry":    _feat_axial_asymmetry(density_map),
        }
        pred, _ = _rule_classify(
            all_feats[label]["density_uniformity"],
            all_feats[label]["radial_peak"],
            all_feats[label]["ring_structure"],
            all_feats[label]["axial_asymmetry"],
        )
        all_feats[label]["_pred"] = pred
        print(f"  {label:<20} done  ->  {pred}")

    labels = list(all_feats.keys())
    col_w  = 10

    # Header
    print()
    header = f"{'feature':<24}" + "".join(f"{lb[:col_w]:>{col_w}}" for lb in labels)
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for fn in FEAT_NAMES:
        vals = [all_feats[lb][fn] for lb in labels]
        best = max(range(len(vals)), key=lambda i: vals[i])
        row  = f"{fn:<24}"
        for i, v in enumerate(vals):
            marker = "<" if i == best else " "
            row   += f"{v:>{col_w-1}.4f}{marker}"
        print(row)

    print(sep)
    pred_row = f"{'predicted':<24}"
    for lb in labels:
        p = all_feats[lb]["_pred"]
        pred_row += f"{p[:col_w]:>{col_w}}"
    print(pred_row)
    print(sep)

    # Current threshold analysis
    print()
    print("Current thresholds:")
    print("  unif>0.40                   -> linear")
    print("  asym>0.70                   -> shear")
    print("  ring<0.55 AND rpk<0.80      -> turbulence")
    print("  rpk<0.20                    -> solid_rot")
    print("  else                        -> vortex")
    print()
    print("Conflict check (cases where turbulence rule misfires or fails):")
    for lb in labels:
        f    = all_feats[lb]
        ring = f["ring_structure"]
        rpk  = f["radial_peak"]
        fires = ring < 0.55 and rpk < 0.65
        if fires and f["_pred"] != "turbulence":
            print(f"  WARNING {lb:<18} ring={ring:.3f} rpk={rpk:.3f}  rule fires but pred={f['_pred']}")
        elif not fires and "turb" in lb.lower():
            print(f"  MISS    {lb:<18} ring={ring:.3f} rpk={rpk:.3f}  turbulence rule did NOT fire")

    return all_feats