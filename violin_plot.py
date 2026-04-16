import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import matplotlib.patheffects as path_effects
import sys
import glob

NUM_SPLITS = 5 #change based on the number of the result csv
THRESHOLD = -8.8626 #change if needed

#9 bin boundary as a label on the X axis
bins = [-15, -13, -11, -9, -7, -5, -3, -1, 1]

midpoints = [ (bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1) ]
ordered_labels = [f'{m:.0f}' for m in midpoints]

border_labels = [str(b) for b in bins]

FILE_PATH_TEMPLATE = os.path.join('.', '{F:02d}', '{F:02d}.csv')

def load_and_average_predictions(folder_path, ordered_labels, bins):
    preds_acc = {} 
    exp_vals = {}  

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"There is no CSV file in the folder: {folder_path}")

    for file_path in csv_files:
        try:
            df_k = pd.read_csv(file_path, sep=';', header=0)
            if df_k.empty:
                continue

            for _, row in df_k.iterrows():
                name = str(row['name'])
                expected = float(row['expected'])
                predicted = float(row['predicted'])

                if name not in preds_acc:
                    preds_acc[name] = []
                    exp_vals[name] = expected
                
                preds_acc[name].append(predicted)

        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue

   
    results = []
    for name, v_list in preds_acc.items():
        results.append({
            'name': name,
            'expected': exp_vals[name],
            'predicted': np.mean(v_list) 
        })

    df_final = pd.DataFrame(results)

    if df_final.empty:
        raise ValueError("There is no data left to process!")

    # Bin-elés és hiba számítás (marad az eredeti logika szerint)
    df_final['absolute_error'] = (df_final['expected'] - df_final['predicted']).abs()
    df_final['expected_bin'] = pd.cut(df_final['expected'], bins=bins, labels=ordered_labels, right=False)
    df_final['expected_bin'] = pd.Categorical(df_final['expected_bin'], categories=ordered_labels, ordered=True)

    return df_final.dropna(subset=['expected_bin'])


def plot_absolute_error_by_fpfn(df, threshold, border_labels):

    def get_classification_status(row, threshold):
        is_positive_true = row['expected'] <= threshold
        is_positive_pred = row['predicted'] <= threshold

        if is_positive_true and not is_positive_pred:
            return 'FN' # False Negative
        elif not is_positive_true and is_positive_pred:
            return 'FP' # False Positive
        else:
            return None # True Positive or True Negative

    df['classification_status'] = df.apply(lambda row: get_classification_status(row, threshold), axis=1)

    df_original_fpfn = df.dropna(subset=['classification_status']).copy()

    hue_order_fpfn = ['FP', 'FN']
    df_original_fpfn['classification_status'] = pd.Categorical(
        df_original_fpfn['classification_status'],
        categories=hue_order_fpfn,
        ordered=True
    )

    counts_fpfn = df_original_fpfn.groupby(['expected_bin', 'classification_status']).size().unstack(fill_value=0).reset_index()
    counts_fpfn['expected_bin_index'] = counts_fpfn['expected_bin'].cat.codes

    df_filtered = df_original_fpfn.copy()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()



    sns.violinplot(
        x='expected_bin',
        y='absolute_error',
        hue='classification_status',
        data=df_filtered,
        split=True,
        inner=None,
        palette={'FP': 'tab:blue', 'FN': 'tab:green'},
        ax=ax,
        zorder=2
    )

    y_base = 10 * 0.8
    y_offset = 10 / 20

    ax.set_ylim(-0.2, 10.2)

    stroke = [path_effects.withStroke(linewidth=3, foreground='white')]

    for index, row in counts_fpfn.iterrows():
        bin_index = row['expected_bin_index']
        fp_count = row.get('FP', 0)
        fn_count = row.get('FN', 0)

        if fp_count > 0:
            ax.text(
                bin_index,
                y_base - y_offset * 1.5,
                f'{fp_count:,}',
                color='tab:blue',
                fontsize=18,
                weight='bold',
                horizontalalignment='center',
                path_effects=stroke
            )

        if fn_count > 0:
            ax.text(
                bin_index,
                y_base - y_offset * 3,
                f'{fn_count:,}',
                color='tab:green',
                fontsize=18,
                weight='bold',
                horizontalalignment='center',
                path_effects=stroke
            )

    ax.set_xticks(np.arange(len(border_labels)))
    ax.set_xticklabels(border_labels, fontsize=14)
    ax.set_xlim(-0.2, len(border_labels) - 0.8)
    ax.tick_params(axis='y', labelsize=14)
    plt.xlabel(r'DS$_{exp}$ [kcal/mol]', fontsize=14)
    plt.ylabel(r'Absolute prediction error [kcal/mol]', fontsize=14)
    leg = plt.legend(loc='upper right', fontsize=18, frameon=True)

    for handle in leg.legend_handles:
        handle.set_linewidth(0)

    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(1)

    plt.grid(linestyle='--', alpha=0.7,zorder=0)

    plt.tight_layout()
    plt.savefig('violin_plot.png')
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error! Usage: python violin_plot.py /path/directory")
        sys.exit(1)

    input_folder = sys.argv[1]

    try:
        df_averaged = load_and_average_predictions(input_folder, ordered_labels, bins)

        plot_absolute_error_by_fpfn(df_averaged.copy(), THRESHOLD, border_labels)
        print(f'Done!')

    except Exception as e:
        print(f"Error: {e}")
