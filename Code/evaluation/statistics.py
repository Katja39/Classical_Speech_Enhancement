import os
import json
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_SUMMARY_DIR = os.path.join(SCRIPT_DIR, "..", "results_summary")
JSON_FILENAME = "all_results.json"

def load_data(folder_filter_func):

    all_folders = [d for d in os.listdir(RESULTS_SUMMARY_DIR)
                   if os.path.isdir(os.path.join(RESULTS_SUMMARY_DIR, d))]
    test_folders = [f for f in all_folders if folder_filter_func(f)]

    if not test_folders:
        print("No folders remaining after filtering.")
        return None

    all_data = []
    for folder in sorted(test_folders):
        json_path = os.path.join(RESULTS_SUMMARY_DIR, folder, JSON_FILENAME)
        if not os.path.isfile(json_path):
            print(f"Warning: {json_path} not found, skipping.")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_test = pd.DataFrame(data)
        df_test['test_group'] = folder
        all_data.append(df_test)

    if not all_data:
        print("Keine JSON-Dateien geladen.")
        return None

    return pd.concat(all_data, ignore_index=True)


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
        figsize=(10, 6)):

    df = load_data(folder_filter_func)
    if df is None:
        return

    #filter algorithms
    if include_algs is not None:
        df = df[df['alg'].isin(include_algs)]
        if df.empty:
            print("No data left after filtering algorithms.")
            return

    if filter_metric is not None:
        if filter_metric not in df.columns:
            print(f"Warning: filter_metric '{filter_metric}' not found, skipping filter.")
        else:
            before = len(df)
            if filter_max is not None:
                df = df[df[filter_metric] <= filter_max]
            after = len(df)
            print(f"Filtered by {filter_metric} <= {filter_max}): {after} of {before} rows kept.")
            if after == 0:
                print("No data left after filtering.")
                return

    available_metrics = [m for m in metrics if m in df.columns]
    missing = set(metrics) - set(available_metrics)
    if missing:
        print(f"Warning: The following metrics are not present in the DataFrame: {missing}")

    if not available_metrics:
        print("None of the requested metrics are available.")
        return

    # Mean per algorithm for each metric
    summary = df.groupby('alg')[available_metrics].mean()

    counts = df.groupby('alg').size()
    out_dict = {}
    for alg in summary.index:
        out_dict[alg] = {met: summary.loc[alg, met] for met in available_metrics}
        out_dict[alg]['count'] = int(counts[alg])

    # Print summary table
    print("\nAverage values per algorithm:")
    print(summary.round(4))

    # Grouped bar chart
    ax = summary.plot(kind='bar', figsize=figsize, width=0.7)
    if title is None:
        title = "Average metric values per algorithm"
    plt.title(title)
    plt.ylabel('Mean value')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt=value_format, label_type='edge', padding=2)
        y_max = summary.max().max()
        plt.ylim(0, y_max * 1.15)

    if show_noisy_lines:
        for i, metric in enumerate(available_metrics):
            if metric.endswith(('_stoiopt', '_pesqopt', '_balopt')):
                base = metric.split('_')[0]
                noisy_metric = f"{base}_noisy"
            else:
                continue

            if noisy_metric not in df.columns:
                print(f"Warning: {noisy_metric} not found, skipping noisy line.")
                continue

            mean_noisy = df[noisy_metric].mean()
            line_color = 'red'
            ax.axhline(y=mean_noisy, color=line_color, linestyle='--', linewidth=1.5,
                       label=f'{noisy_metric} (avg)')

            ax.text(0.35, mean_noisy + 0.33 , f'{mean_noisy:.3f}',
                    transform=ax.get_xaxis_transform(),
                    ha='left', va='center', fontsize=plt.rcParams['font.size'], color=line_color,
                    clip_on=False)

        handles, labels = ax.get_legend_handles_labels()
        if metric_labels:
            new_labels = [metric_labels.get(l, l) for l in labels]
        else:
            new_labels = labels
        ax.legend(handles, new_labels, title='Metric', loc='lower right')

    plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.show()

    if output_json:
        print(f"Attempting to write to: {repr(output_json)}")  # Debug
        if not isinstance(output_json, str) or not output_json.strip():
            print("Warning: output_json is not a valid filename, skipping JSON export.")
        else:
            try:
                out_dir = os.path.dirname(output_json)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(out_dict, f, indent=2, ensure_ascii=False)
                print(f"Aggregated data saved to: {output_json}")
            except Exception as e:
                print(f"Error saving JSON to {output_json}: {e}")

if __name__ == "__main__":
    # STOI Vergleich aller Tests (ohne mmse) mit true noise
    #plot_algorithm_summary(
    #    folder_filter_func=lambda name: "mitTrueNoise" in name,
    #    output_json="meanBestSTOI_allAlgorithms_trueNoise.json",
    #    metrics=["stoi_stoiopt"],
    #    metric_labels={
    #        "stoi_stoiopt": "STOI optimiert",
    #        "stoi_noisy (avg)": "STOI vor Optimierung",
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    title="Durchschnittlicher bester STOI – mit true noise",
    #    show_values=True,
    #    show_noisy_lines=True
    #)

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

    # STOI Vergleich aller Tests (ohne mmse) ohne true noise
    # plot_algorithm_summary(
    #    folder_filter_func=lambda name: "ohneTrueNoise" in name,
    #    output_json="meanBestSTOI_allAlgorithms_withoutTrueNoise.json",
    #    metrics=["stoi_stoiopt"],
    #   metric_labels={
    #        "stoi_stoiopt": "STOI optimiert",
    #        "stoi_noisy (avg)": "STOI vor Optimierung",
    #    },
    #    include_algs=["spectralSubtractor", "wiener", "omlsa"],
    #    title="Durchschnittlicher bester STOI – ohne true noise",
    #    show_values=True,
    #    show_noisy_lines=True
    # )

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

#Vergleich bei schlechter Qualität, welcher Algorithmus schneidet da besser ab?
    plot_algorithm_summary(
        folder_filter_func=lambda name: "ohneTrueNoise" in name,
        output_json="meanBestSTOI_lowQuality_withoutTrueNoise.json",
        metrics=["stoi_stoiopt"],
        metric_labels={
            "stoi_stoiopt": "STOI optimiert",
            "stoi_noisy (avg)": "STOI vor Optimierung (niedrige Qualität)"
        },
        include_algs=["spectralSubtractor", "wiener", "omlsa"],
        filter_metric="stoi_noisy",
        filter_max=0.7,
        title="Durchschnittlicher STOI (optimiert) – mit STOI ≤ 0.7",
        show_values=True,
        show_noisy_lines=True
    )

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
    #    filter_max=1.4,
    #    title="Durchschnittlicher PESQ (optimiert) – mit PESQ ≤ 1.4",
    #    show_values=True,
    #    show_noisy_lines=True
    #)




#STOI vs. PESQ Trade-off - „Algorithmus A maximiert Qualität, Algorithmus B Verständlichkeit“
#Unterschied von STOI und PESQ Optimierung

#Testszenarien
#Ergebnisse bei Rauschen (mit true noise)
#Ergebnisse bei Musik (mit true noise)
#Ergebnisse bei Stimmen(mit true noise)

#Unterschied true noise vs geschätztes noise mit percentile oder min tracking

