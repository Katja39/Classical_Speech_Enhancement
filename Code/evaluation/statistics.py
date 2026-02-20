import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_SUMMARY_DIR = (SCRIPT_DIR / ".." / "results_summary").resolve()
JSON_FILENAME = "all_results.json"

_DF_CACHE: dict[tuple[str, ...], pd.DataFrame] = {}
ALG_LABELS = {
    "omlsa": "Log-MMSE",
}

plt.rcParams.update({
    "font.size": 12,        # Standard
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})

def rename_alg(name: str) -> str:
    return ALG_LABELS.get(name, name)

def write_json(obj, output_json: str | None):
    if not output_json:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {path}")

def load_data(folder_filter_func):
    all_folders = [p.name for p in RESULTS_SUMMARY_DIR.iterdir() if p.is_dir()]
    test_folders = sorted([f for f in all_folders if folder_filter_func(f)])

    if not test_folders:
        print("No folders remaining after filtering.")
        return None

    key = tuple(test_folders)
    if key in _DF_CACHE:
        return _DF_CACHE[key].copy()

    frames = []
    for folder in test_folders:
        json_path = RESULTS_SUMMARY_DIR / folder / JSON_FILENAME
        if not json_path.is_file():
            print(f"Warning: {json_path} not found, skipping.")
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        frames.append(pd.DataFrame(data).assign(test_group=folder))

    if not frames:
        print("No JSON found.")
        return None

    df = pd.concat(frames, ignore_index=True)
    _DF_CACHE[key] = df
    return df.copy()


def apply_filters(df, include_algs=None, filter_metric=None, filter_max=None):
    if df is None or df.empty:
        return df

    if include_algs is not None:
        df = df[df["alg"].isin(include_algs)]
        if df.empty:
            print("No data left after filtering algorithms.")
            return df

    if filter_metric and (filter_max is not None):
        if filter_metric not in df.columns:
            print(f"Warning: filter_metric '{filter_metric}' not found, skipping filter.")
        else:
            before = len(df)
            df = df[df[filter_metric] <= filter_max]
            after = len(df)
            print(f"Filtered by {filter_metric} <= {filter_max}: {after} of {before} rows kept.")
            if df.empty:
                print("No data left after filtering.")
                return df

    return df

def get_df(folder_filter_func, include_algs=None, filter_metric=None, filter_max=None):
    df = load_data(folder_filter_func)
    if df is None:
        return None
    df = apply_filters(df, include_algs=include_algs, filter_metric=filter_metric, filter_max=filter_max)
    if df is None or df.empty:
        return None
    return df

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        return False
    return True

def _noisy_col(metric: str) -> str | None:
    base = metric.split("_", 1)[0]
    return f"{base}_noisy"

def _add_delta(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str, str]:
    noisy = _noisy_col(metric)
    if noisy not in df.columns:
        raise KeyError(f"Cannot compute delta: noisy column '{noisy}' not found.")
    plot_col = f"delta_{metric}"
    df = df.copy()
    df[plot_col] = df[metric] - df[noisy]
    return df, plot_col, noisy

def _add_scenario(df: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    df = df.copy()
    gl = df["test_group"].astype(str).str.lower()
    scenario = pd.Series(pd.NA, index=df.index, dtype="object")
    for key, label in scenarios.items():
        scenario = scenario.mask(gl.str.contains(str(key).lower(), na=False), label)
    df["scenario"] = scenario
    return df.dropna(subset=["scenario"])

def _tables_by_alg_scenario(df: pd.DataFrame, value_col: str):
    means = df.groupby(["alg", "scenario"])[value_col].mean().unstack("scenario")
    counts = df.groupby(["alg", "scenario"]).size().unstack("scenario").fillna(0).astype(int)
    return means, counts

def _reindex_tables(means, counts, alg_order, scen_order):
    means = means.reindex(index=alg_order, columns=scen_order)
    counts = counts.reindex(index=alg_order, columns=scen_order).fillna(0).astype(int)
    return means, counts

def _apply_legend_labels(ax, metric_labels: dict | None, title="Metric", loc="best"):
    if not metric_labels:
        return
    handles, labels = ax.get_legend_handles_labels()
    labels = [metric_labels.get(l, l) for l in labels]
    ax.legend(handles, labels, title=title, loc=loc)

def _plot_heatmap(means, counts, title, xlabel, ylabel,
                      figsize=(8, 4), value_format="{:.3f}", show_counts=True):
    data = means.to_numpy()
    plt.figure(figsize=figsize)
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(means.shape[1]), means.columns, rotation=30, ha="right")
    plt.yticks(range(means.shape[0]), [rename_alg(a) for a in means.index])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, alg in enumerate(means.index):
        for j, sc in enumerate(means.columns):
            val = means.loc[alg, sc]
            if pd.isna(val):
                txt = "n/a"
            else:
                txt = value_format.format(val)
                if show_counts:
                    txt += f"\n(n={int(counts.loc[alg, sc])})"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_algorithm_summary(
    folder_filter_func,
    metrics,
    title=None,
    output_json=None,
    show_values=None,
    show_noisy_lines=None,
    metric_labels=None,
    include_algs=None,
    filter_metric=None,
    filter_max=None,
    value_format="{:.3f}",
    figsize=(10, 6),
):
    df = get_df(folder_filter_func, include_algs, filter_metric, filter_max)
    if df is None:
        return

    available = [m for m in metrics if m in df.columns]
    missing = set(metrics) - set(available)
    if missing:
        print(f"Warning: metrics missing: {missing}")
    if not available:
        print("None of the requested metrics are available.")
        return

    summary = df.groupby("alg")[available].mean()
    counts = df.groupby("alg").size()

    out = {
        alg: {met: float(summary.loc[alg, met]) for met in available} | {"count": int(counts[alg])}
        for alg in summary.index
    }

    print("\nAverage values per algorithm:")
    print(summary.round(4))

    ax = summary.plot(kind="bar", figsize=figsize, width=0.7)
    ax.set_title(title or "Average metric values per algorithm")
    ax.set_ylabel("Mean value")
    ax.set_xlabel("Algorithm")
    ax.set_xticklabels([rename_alg(t.get_text()) for t in ax.get_xticklabels()],
                       rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt=value_format, label_type="edge", padding=2)
        y_max = summary.max().max()
        ax.set_ylim(0, y_max * 1.15)

    if show_noisy_lines:
        for metric in available:
            noisy_metric = _noisy_col(metric) if metric.endswith(("_stoiopt", "_pesqopt", "_balopt")) else None
            if not noisy_metric or noisy_metric not in df.columns:
                continue

            y0 = float(df[noisy_metric].mean())

            line_label = f"{noisy_metric} (avg)"
            if metric_labels is not None:
                line_label = metric_labels.get(line_label, line_label)

            ax.axhline(
                y=y0,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=line_label,
            )

            x_mid = (len(summary.index) - 1) / 2 + 0.4
            ax.text(
                x_mid,
                y0,
                value_format.format(y0),
                color="red",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    _apply_legend_labels(ax, metric_labels, title="Metric", loc="lower right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Metric", loc="lower right")

    plt.tight_layout()
    plt.show()

    write_json(out, output_json)


def plot_tradeoff_scatter(
    folder_filter_func,
    variant="stoiopt",
    title=None,
    include_algs=None,
    filter_metric=None,
    filter_max=None,
    show_means=True,
    show_origin_lines=True,
    alpha_points=0.25,
    figsize=(8, 6),
    output_json=None,
    value_format="{:.4f}",
):
    df = get_df(folder_filter_func, include_algs, filter_metric, filter_max)
    if df is None:
        return

    stoi_col = f"stoi_{variant}"
    pesq_col = f"pesq_{variant}"
    req = ["alg", "stoi_noisy", "pesq_noisy", stoi_col, pesq_col]
    if not require_cols(df, req):
        return

    d = df[req].dropna().copy()
    d["d_stoi"] = d[stoi_col] - d["stoi_noisy"]
    d["d_pesq"] = d[pesq_col] - d["pesq_noisy"]
    d = d.dropna(subset=["d_stoi", "d_pesq"])
    if d.empty:
        print("No valid rows after dropping NaNs.")
        return

    out = {
        "variant": variant,
        "stoi_col": stoi_col,
        "pesq_col": pesq_col,
        "filters": {"include_algs": include_algs, "filter_metric": filter_metric, "filter_max": filter_max},
        "per_algorithm": {},
    }

    print("\nTrade-off summary (ΔSTOI, ΔPESQ) per algorithm:")
    for alg, g in d.groupby("alg"):
        ds = g["d_stoi"].to_numpy()
        dp = g["d_pesq"].to_numpy()
        stats = {
            "count": int(len(g)),
            "mean_d_stoi": float(ds.mean()),
            "std_d_stoi": float(ds.std(ddof=1)) if len(ds) > 1 else 0.0,
            "median_d_stoi": float(np.median(ds)),
            "mean_d_pesq": float(dp.mean()),
            "std_d_pesq": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
            "median_d_pesq": float(np.median(dp)),
            "pct_d_stoi_negative": float((ds < 0).mean() * 100),
            "pct_d_pesq_negative": float((dp < 0).mean() * 100),
        }
        out["per_algorithm"][alg] = stats

        print(
            f"- {alg:18s} n={stats['count']:4d} | "
            f"mean ΔSTOI={value_format.format(stats['mean_d_stoi'])} | "
            f"mean ΔPESQ={value_format.format(stats['mean_d_pesq'])} | "
            f"%ΔSTOI<0={stats['pct_d_stoi_negative']:.1f}%"
        )

    plt.figure(figsize=figsize)

    for alg, g in d.groupby("alg"):
        plt.scatter(g["d_stoi"], g["d_pesq"], alpha=alpha_points, label=f"{rename_alg(alg)} (files)")
        if show_means and len(g) > 0:
            m_stoi = out["per_algorithm"][alg]["mean_d_stoi"]
            m_pesq = out["per_algorithm"][alg]["mean_d_pesq"]
            plt.scatter([m_stoi], [m_pesq], marker="X", s=160, edgecolors="black", linewidths=1.2, label=f"{rename_alg(alg)} mean")

    if show_origin_lines:
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel(f"ΔSTOI = {stoi_col} - stoi_noisy")
    plt.ylabel(f"ΔPESQ = {pesq_col} - pesq_noisy")
    plt.title(title or f"Trade-off: ΔSTOI vs ΔPESQ ({variant})\nX = Algorithmus-Mittelwert")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    write_json(out, output_json)


def plot_tradeoff_variants_summary(
    folder_filter_func,
    variants=("stoiopt", "balopt", "pesqopt"),
    include_algs=None,
    filter_metric=None,
    filter_max=None,
    title=None,
    figsize=(8, 6),
    output_json=None,
):
    df = get_df(folder_filter_func, include_algs, filter_metric, filter_max)
    if df is None:
        return

    out = {
        "variants": list(variants),
        "filters": {"include_algs": include_algs, "filter_metric": filter_metric, "filter_max": filter_max},
        "per_algorithm": {},
    }

    plt.figure(figsize=figsize)

    for alg, g_alg in df.groupby("alg"):
        points = []
        for v in variants:
            stoi_col = f"stoi_{v}"
            pesq_col = f"pesq_{v}"
            req = ["stoi_noisy", "pesq_noisy", stoi_col, pesq_col]
            if any(c not in g_alg.columns for c in req):
                continue

            gg = g_alg.dropna(subset=req)
            if gg.empty:
                continue

            ds = float((gg[stoi_col] - gg["stoi_noisy"]).mean())
            dp = float((gg[pesq_col] - gg["pesq_noisy"]).mean())
            points.append((v, ds, dp))

        if len(points) < 2:
            continue

        out["per_algorithm"][alg] = {v: {"mean_d_stoi": ds, "mean_d_pesq": dp} for v, ds, dp in points}

        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=2, label=rename_alg(alg))
        for v, x, y in points:
            plt.text(x, y, f" {v}", fontsize=9, va="center")

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("ΔSTOI")
    plt.ylabel("ΔPESQ")
    plt.title(title or "Trade-off: Mittelwerte pro Variante (stoiopt/balopt/pesqopt)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    write_json(out, output_json)


def plot_scenario_heatmap(
    folder_filter_func,
    scenarios,
    metric,
    include_algs=None,
    filter_metric=None,
    filter_max=None,
    title=None,
    figsize=(8, 4),
    output_json=None,
    delta_to_noisy=True,
    value_format="{:.3f}",
    show_counts=True,
):
    df = get_df(folder_filter_func, include_algs, filter_metric, filter_max)
    if df is None:
        return

    df = _add_scenario(df, scenarios)
    if df.empty:
        print("No data left after scenario assignment (check folder names / scenarios keys).")
        return
    if metric not in df.columns:
        print(f"Metric '{metric}' not found.")
        return

    noisy_col = None
    plot_col = metric
    if delta_to_noisy:
        try:
            df, plot_col, noisy_col = _add_delta(df, metric)
        except KeyError as e:
            print(e)
            return

    means, counts = _tables_by_alg_scenario(df, plot_col)

    alg_order = [a for a in include_algs if a in means.index] if include_algs else list(means.index)
    scen_order = [scenarios[k] for k in scenarios.keys() if scenarios[k] in means.columns]
    means, counts = _reindex_tables(means, counts, alg_order, scen_order)

    out = {
        "metric": metric,
        "delta_to_noisy": bool(delta_to_noisy),
        "value_column_used": plot_col,
        "noisy_column_used": noisy_col,
        "scenarios": scenarios,
        "algorithms": alg_order,
        "table_mean": {a: {s: (None if pd.isna(means.loc[a, s]) else float(means.loc[a, s])) for s in scen_order} for a in alg_order},
        "table_count": {a: {s: int(counts.loc[a, s]) for s in scen_order} for a in alg_order},
    }

    _plot_heatmap(
        means, counts,
        title=title or f"Scenario heatmap: {metric}" + (" (Δ vs noisy)" if delta_to_noisy else ""),
        xlabel="Scenario", ylabel="Algorithm",
        figsize=figsize, value_format=value_format, show_counts=show_counts
    )
    write_json(out, output_json)


def plot_noise_method_usage_grouped_side_by_side(
    folder_filter_func,
    include_algs=None,
    filter_metric=None,
    filter_max=None,
    title=None,
    figsize=(10, 5),
    output_json=None,
    show_percent=True,
    bar_width=0.35,
):
    df = get_df(folder_filter_func, include_algs, filter_metric, filter_max)
    if df is None:
        return

    def extract_counts(params_col):
        if params_col not in df.columns:
            return None, None
        tmp = df.copy()
        tmp["noise_method"] = tmp[params_col].apply(lambda d: d.get("noise_method") if isinstance(d, dict) else None)
        tmp = tmp.dropna(subset=["noise_method"])
        if tmp.empty:
            return None, None
        counts = pd.crosstab(tmp["alg"], tmp["noise_method"])
        if include_algs is not None:
            counts = counts.reindex(include_algs).fillna(0).astype(int)
        perc = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        return counts, perc

    c_stoi, p_stoi = extract_counts("best_params_stoi")
    c_pesq, p_pesq = extract_counts("best_params_pesq")
    if c_stoi is None or c_pesq is None:
        print("Missing best_params_stoi or best_params_pesq (or no noise_method entries).")
        return

    algs = include_algs if include_algs is not None else sorted(set(c_stoi.index) | set(c_pesq.index))

    t_stoi = p_stoi if show_percent else c_stoi.astype(float)
    t_pesq = p_pesq if show_percent else c_pesq.astype(float)

    methods = sorted(set(t_stoi.columns) | set(t_pesq.columns))
    t_stoi = t_stoi.reindex(index=algs, columns=methods).fillna(0.0)
    t_pesq = t_pesq.reindex(index=algs, columns=methods).fillna(0.0)

    # JSON export
    out = {
        "show_percent": bool(show_percent),
        "filters": {"include_algs": include_algs, "filter_metric": filter_metric, "filter_max": filter_max},
        "stoi_opt_counts": c_stoi.reindex(index=algs, columns=methods, fill_value=0).to_dict(),
        "pesq_opt_counts": c_pesq.reindex(index=algs, columns=methods, fill_value=0).to_dict(),
        "methods": methods,
        "algorithms": algs,
    }

    # Plot
    x = np.arange(len(algs))
    x_stoi = x - bar_width / 2
    x_pesq = x + bar_width / 2

    fig, ax = plt.subplots(figsize=figsize)
    bottom_stoi = np.zeros(len(algs))
    bottom_pesq = np.zeros(len(algs))

    # Colors: Matplotlib cycle
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C3", "C4", "C5"]
    method_colors = {m: cycle[i % len(cycle)] for i, m in enumerate(methods)}

    labeled = set()
    for m in methods:
        vals_stoi = t_stoi[m].to_numpy()
        vals_pesq = t_pesq[m].to_numpy()
        label = m if m not in labeled else None

        ax.bar(x_stoi, vals_stoi, bar_width, bottom=bottom_stoi, color=method_colors[m],
               edgecolor="black", linewidth=0.3, label=label)
        ax.bar(x_pesq, vals_pesq, bar_width, bottom=bottom_pesq, color=method_colors[m],
               edgecolor="black", linewidth=0.3)

        labeled.add(m)
        bottom_stoi += vals_stoi
        bottom_pesq += vals_pesq

    ax.set_xticks(x)
    ax.set_xticklabels([rename_alg(a) for a in algs], rotation=45, ha="right")

    minor_pos, minor_lab = [], []
    for i in range(len(algs)):
        minor_pos.extend([x_stoi[i], x_pesq[i]])
        minor_lab.extend(["STOI", "PESQ"])
    ax.set_xticks(minor_pos, minor=True)
    ax.set_xticklabels(minor_lab, minor=True)

    ax.tick_params(axis="x", which="major", pad=18)
    ax.tick_params(axis="x", which="minor", pad=2)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Anteil [%]" if show_percent else "Count")
    ax.set_title(title or "Welche Noise-Estimation wurde gewählt? (STOI-opt vs PESQ-opt)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="noise_method", loc="upper right")
    ax.set_ylim(0, 100 if show_percent else max(bottom_stoi.max(), bottom_pesq.max()) * 1.1)

    plt.tight_layout()
    plt.show()

    write_json(out, output_json)


def plot_oracle_gap_heatmap(
    folder_filter_func_true,
    folder_filter_func_est,
    scenarios,
    metric,
    include_algs=None,
    title=None,
    figsize=(8, 4),
    output_json=None,
    delta_to_noisy=True,
    value_format="{:.3f}",
    show_counts=True,
):
    df_true = get_df(folder_filter_func_true, include_algs)
    df_est = get_df(folder_filter_func_est, include_algs)
    if df_true is None or df_est is None:
        return

    def prepare(df):
        df = _add_scenario(df, scenarios)
        if df.empty or metric not in df.columns:
            return None
        noisy_col = None
        plot_col = metric
        if delta_to_noisy:
            df, plot_col, noisy_col = _add_delta(df, metric)
        means, counts = _tables_by_alg_scenario(df, plot_col)
        return means, counts, plot_col, noisy_col

    prep_true = prepare(df_true)
    prep_est = prepare(df_est)
    if prep_true is None or prep_est is None:
        print("Could not prepare tables (missing metric/noisy columns or scenario mapping).")
        return

    means_true, counts_true, plot_col_true, noisy_col_true = prep_true
    means_est, counts_est, plot_col_est, noisy_col_est = prep_est

    alg_order = include_algs if include_algs else sorted(set(means_true.index) | set(means_est.index))
    scen_order = [scenarios[k] for k in scenarios.keys()]
    means_true, counts_true = _reindex_tables(means_true, counts_true, alg_order, scen_order)
    means_est, counts_est = _reindex_tables(means_est, counts_est, alg_order, scen_order)

    diff = means_true - means_est

    out = {
        "metric": metric,
        "delta_to_noisy": bool(delta_to_noisy),
        "true_value_col": plot_col_true,
        "est_value_col": plot_col_est,
        "algorithms": alg_order,
        "scenarios": scenarios,
        "means_true": {a: {s: (None if pd.isna(means_true.loc[a, s]) else float(means_true.loc[a, s])) for s in scen_order} for a in alg_order},
        "means_est": {a: {s: (None if pd.isna(means_est.loc[a, s]) else float(means_est.loc[a, s])) for s in scen_order} for a in alg_order},
        "diff_true_minus_est": {a: {s: (None if pd.isna(diff.loc[a, s]) else float(diff.loc[a, s])) for s in scen_order} for a in alg_order},
        "counts_true": {a: {s: int(counts_true.loc[a, s]) for s in scen_order} for a in alg_order},
        "counts_est": {a: {s: int(counts_est.loc[a, s]) for s in scen_order} for a in alg_order},
    }

    # Plot diff heatmap (mit nT/nE)
    plt.figure(figsize=figsize)
    im = plt.imshow(diff.to_numpy(), aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(scen_order)), scen_order, rotation=30, ha="right")
    plt.yticks(range(len(alg_order)), [rename_alg(a) for a in alg_order])
    plt.title(title or f"Oracle-Gap: TrueNoise − Estimated ({metric})" + (" (Δ vs noisy)" if delta_to_noisy else ""))
    plt.xlabel("Scenario")
    plt.ylabel("Algorithm")

    for i, alg in enumerate(alg_order):
        for j, sc in enumerate(scen_order):
            val = diff.loc[alg, sc]
            if pd.isna(val):
                txt = "n/a"
            else:
                txt = value_format.format(val)
                if show_counts:
                    txt += f"\n(nT={counts_true.loc[alg, sc]}, nE={counts_est.loc[alg, sc]})"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    write_json(out, output_json)


if __name__ == "__main__":

   # STOI Vergleich aller Tests (ohne mmse) mit true noise
   # plot_algorithm_summary(
   #     folder_filter_func=lambda name: "mitTrueNoise" in name,
   #     output_json="meanBestSTOI_allAlgorithms_trueNoise.json",
   #     metrics=["stoi_stoiopt"],
   #     metric_labels={
   #         "stoi_stoiopt": "STOI optimiert",
   #         "stoi_noisy (avg)": "STOI vor Optimierung",
   #     },
   #     include_algs=["spectralSubtractor", "wiener", "omlsa"],
   #     title="Durchschnittlicher bester STOI – mit true noise",
   #     show_values=True,
   #     show_noisy_lines=True
   # )

    # PESQ Vergleich aller Tests (ohne mmse) mit true noise
    #plot_algorithm_summary(
    #    folder_filter_func=lambda name: "mitTrueNoise" in name,
    #    output_json="meanBestPESQ_allAlgorithms_trueNoise.json",
    #    metrics=["pesq_pesqopt"],
    #    metric_labels={
    #        "pesq_pesqopt": "PESQ optimiert",
    #        "pesq_noisy (avg)": "PESQ vor Optimierung",
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    title="Durchschnittlicher bester PESQ – mit true noise",
    #    show_values=True,
    #    show_noisy_lines=True
    #)

     #STOI Vergleich aller Tests (ohne mmse) ohne true noise
     #plot_algorithm_summary(
     #   folder_filter_func=lambda name: "ohneTrueNoise" in name,
     #   output_json="meanBestSTOI_allAlgorithms_withoutTrueNoise.json",
     #   metrics=["stoi_stoiopt"],
     #  metric_labels={
     #       "stoi_stoiopt": "STOI optimiert",
     #       "stoi_noisy (avg)": "STOI vor Optimierung",
     #   },
     #   include_algs=["spectralSubtractor", "wiener", "omlsa"],
     #   title="Durchschnittlicher bester STOI – ohne true noise",
     #   show_values=True,
     #   show_noisy_lines=True
     #)

    #PESQ Vergleich aller Tests (ohne mmse) ohne true noise
    #plot_algorithm_summary(
    #    folder_filter_func=lambda name: "ohneTrueNoise" in name,
    #    output_json="meanBestPESQ_allAlgorithms_withoutTrueNoise.json",
    #    metrics=["pesq_pesqopt"],
    #    metric_labels={
    #        "pesq_pesqopt": "PESQ optimiert",
    #        "pesq_noisy (avg)": "PESQ vor Optimierung",
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    title="Durchschnittlicher bester PESQ – ohne true noise",
    #    show_values=True,
    #    show_noisy_lines=True
    #)

#Vergleich bei schlechter Qualität, welcher Algorithmus schneidet besser ab?
    #plot_algorithm_summary(
    #    folder_filter_func=lambda name: "ohneTrueNoise" in name,
    #    output_json="meanBestSTOI_lowQuality_withoutTrueNoise.json",
    #    metrics=["stoi_stoiopt"],
    #    metric_labels={
    #        "stoi_stoiopt": "STOI optimiert",
    #        "stoi_noisy (avg)": "STOI vor Optimierung (niedrige Qualität)"
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    filter_metric="stoi_noisy",
    #    filter_max=0.7,
    #    title="Durchschnittlicher STOI (optimiert) – mit STOI ≤ 0.7",
    #    show_values=True,
    #    show_noisy_lines=True
    #)

    #plot_algorithm_summary(
    #    folder_filter_func=lambda name: "ohneTrueNoise" in name,
    #    output_json="meanBestPESQ_lowQuality_withoutTrueNoise.json",
    #    metrics=["pesq_pesqopt"],
    #    metric_labels={
    #        "pesq_pesqopt": "PESQ optimiert",
    #        "pesq_pesq (avg)": "PESQ vor Optimierung (niedrige Qualität)"
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    filter_metric="pesq_noisy",
    #    filter_max=1.1,
    #    title="Durchschnittlicher PESQ (optimiert) – mit PESQ ≤ 1.1",
    #    show_values=True,
    #    show_noisy_lines=True
    #)

    #Trade-off: STOI-Optimierung (ohne TrueNoise)
    plot_tradeoff_scatter(
        folder_filter_func=lambda name: "mitTrueNoise" in name,
        variant="stoiopt",
        include_algs=["spectralSubtractor", "wiener", "omlsa"],
        title="ΔSTOI vs ΔPESQ (STOI-Optimierung)",
        figsize=(8, 6),
        output_json="tradeoff_PESQ_vs_STOI_stoiopt_withoutTrueNoise.json"
    )

    # STOI vs. PESQ Trade-off
    # Trade-off: PESQ-Optimierung (ohne TrueNoise)
    plot_tradeoff_scatter(
        folder_filter_func=lambda name: "mitTrueNoise" in name,
        variant="pesqopt",
        include_algs=["spectralSubtractor", "wiener", "omlsa"],
        title="ΔSTOI vs ΔPESQ (PESQ-Optimierung)",
        figsize=(8, 6),
        output_json="tradeoff_PESQ_vs_STOI_pesqopt_withoutTrueNoise.json"
    )

    #Vergleich: STOI-opt vs Score-opt (0.5·STOI + 0.5·PESQ_norm) vs PESQ-opt (Mittelwerte)
    plot_tradeoff_variants_summary(
        folder_filter_func=lambda name: "mitTrueNoise" in name,
        include_algs=["spectralSubtractor", "wiener", "omlsa"],
        title="Vergleich: STOI-opt vs Score-opt vs PESQ-opt (Mittelwerte)",
        output_json="tradeoff_variants_summary_ohneTrueNoise.json"
    )

    # Testszenarien

    SCENARIOS = {
        "rauschen": "Rauschen",
        "musik": "Musik",
        "menschen": "Stimmen",
        "kombi": "Kombination",
    }

    #ΔSTOI bei STOI-Optimierung (ohne TrueNoise) über alle Szenarien
    #plot_scenario_heatmap(
    #    folder_filter_func=lambda name: ("ohneTrueNoise" in name) and any(k in name for k in SCENARIOS),
    #    scenarios=SCENARIOS,
    #    metric="stoi_stoiopt",
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    delta_to_noisy=True,
    #    title="ΔSTOI (STOI-opt) nach Szenario – ohne TrueNoise",
    #    output_json="heatmap_deltaSTOI_stoiopt_scenarios_withoutTrueNoise.json",
    #    figsize=(8, 4)
    #)

    #plot_scenario_heatmap(
    #    folder_filter_func=lambda name: ("ohneTrueNoise" in name)
    #                                    and any(k in name for k in SCENARIOS),
    #    scenarios=SCENARIOS,
    #    metric="pesq_pesqopt",
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    delta_to_noisy=True,
    #    title="ΔPESQ (PESQ-opt) nach Szenario – ohne TrueNoise",
    #    output_json="heatmap_deltaPESQ_pesqopt_scenarios_withoutTrueNoise.json",
    #    figsize=(8, 4)
    #)


    #ΔSTOI bei STOI-Optimierung (mit TrueNoise) über alle Szenarien
    #plot_scenario_heatmap(
    #    folder_filter_func=lambda name: ("mitTrueNoise" in name) and any(k in name for k in SCENARIOS),
    #    scenarios=SCENARIOS,
    #    metric="stoi_stoiopt",
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    delta_to_noisy=True,
    #    title="ΔSTOI (STOI-opt) nach Szenario – mit TrueNoise",
    #    output_json="heatmap_deltaSTOI_stoiopt_scenarios_withTrueNoise.json",
    #    figsize=(8, 4)
    #)


    #plot_scenario_heatmap(
    #    folder_filter_func=lambda name: ("mitTrueNoise" in name)
    #                                    and any(k in name for k in SCENARIOS),
    #    scenarios=SCENARIOS,
    #    metric="pesq_pesqopt",
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    delta_to_noisy=True,
    #    title="ΔPESQ (PESQ-opt) nach Szenario – mit TrueNoise",
    #    output_json="heatmap_deltaPESQ_pesqopt_scenarios_withTrueNoise.json",
    #    figsize=(8, 4)
    #)


    #Estimated noise percentile oder min tracking
    #plot_noise_method_usage_grouped_side_by_side(
    #    folder_filter_func=lambda name: "ohneTrueNoise" in name,
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    title="Wahl der Noise-Estimation Methode (STOI-opt vs PESQ-opt)",
    #    output_json="noise_method_usage_stoi_vs_pesq_withoutTrueNoise.json",
    #    show_percent=True
    #)

    #> 0: Oracle hilft - Noise-Schätzung limitiert um diesen Betrag
    #≈ 0: Noise-Schätzung ist “gut genug” oder Algorithmus wenig abhängig davon
    #nT = number TrueNoise
    #nE = number Estimated
    #plot_oracle_gap_heatmap(
    #folder_filter_func_true=lambda name: "mitTrueNoise" in name,
    #folder_filter_func_est=lambda name: "ohneTrueNoise" in name,
    #scenarios=SCENARIOS,
    #metric="stoi_stoiopt",
    #include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #title="Performance-Gap (TrueNoise − Estimated): ΔSTOI (STOI-opt) nach Szenario",
    #output_json="oracle_gap_deltaSTOI_stoiopt.json",
    #figsize=(8, 4)
    #)

    #plot_oracle_gap_heatmap(
    #folder_filter_func_true=lambda name: "mitTrueNoise" in name,
    #folder_filter_func_est=lambda name: "ohneTrueNoise" in name,
    #scenarios=SCENARIOS,
    #metric="pesq_pesqopt",
    #include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #title="Performance-Gap (TrueNoise − Estimated): ΔPESQ (PESQ-opt) nach Szenario",
    #output_json="oracle_gap_deltaPESQ_pesqopt.json",
    #figsize=(8, 4)
    #)