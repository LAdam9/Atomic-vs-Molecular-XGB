import pandas as pd
import numpy as np
import os
import sys
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# --- KONSTANSOK ---
FIXED_THRESHOLD = -8.8626 #change if needed
MAIN_OUTPUT = 'dens_plot.png'
CBAR_H_OUTPUT = 'colorbar_horizontal.png'
CBAR_V_OUTPUT = 'colorbar_vertical.png'

def load_filter_set(filter_path):
    if os.path.exists(filter_path):
        with open(filter_path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return None

def _read_csv(file_path):
    try:
        return pd.read_csv(file_path, sep=';', header=0)
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_classification_metrics(df):
    y_pred_bin = (df['predicted'] <= FIXED_THRESHOLD).astype(int)
    y_true_bin = (df['expected'] <= FIXED_THRESHOLD).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Total_P': int(np.sum(y_true_bin == 1)),
        'Total_N': int(np.sum(y_true_bin == 0))
    }


def plot_dense_classification(df_final, metrics, file_name):
    x_orig = df_final['expected'].values
    y_orig = df_final['predicted'].values

    xy = np.vstack([x_orig, y_orig])
    z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min())
    idx = z.argsort()
    x, y, z = x_orig[idx], y_orig[idx], z[idx]

    model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    m, c = model.coef_[0][0], model.intercept_[0]
    r_sq = r2_score(y, model.predict(x.reshape(-1, 1)))

    plt.figure(figsize=(10, 10))
    sns.set_style("whitegrid")
    cmap = sns.blend_palette(["#330066", "#0000FF", "#00FF00", "#FFFF00", "#FF0000"], as_cmap=True)

    plt.scatter(x, y, c=z, s=50, cmap=cmap, alpha=0.7, edgecolor='none')

    lims = [-15, 1]
    plt.plot(lims, lims, color='black', linestyle='--', linewidth=4, alpha=0.8)
    plt.plot(lims, m * np.array(lims) + c, color='red', linestyle='--', linewidth=4)

    plt.axvline(x=FIXED_THRESHOLD, color='black', linestyle=':', linewidth=4)
    plt.axhline(y=FIXED_THRESHOLD, color='black', linestyle=':', linewidth=4)

    stroke = [path_effects.withStroke(linewidth=2, foreground='white')]
    v_offset = 0.3

    plt.text(-14.5, FIXED_THRESHOLD + v_offset, "FN", color='black', ha='left', va='bottom', fontsize=18, weight='bold', path_effects=stroke)
    plt.text(-14.5, FIXED_THRESHOLD - v_offset, "TP", color='black',  ha='left', va='top',    fontsize=18, weight='bold', path_effects=stroke)

    plt.text(0.5, FIXED_THRESHOLD + v_offset, "TN", color='black',     ha='right', va='bottom', fontsize=18, weight='bold', path_effects=stroke)
    plt.text(0.5, FIXED_THRESHOLD - v_offset, "FP", color='black',    ha='right', va='top',    fontsize=18, weight='bold', path_effects=stroke)

    tp, tn, fp, fn = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
    t_p, t_n = metrics['Total_P'], metrics['Total_N']
    tpr, tnr = (tp/t_p*100 if t_p>0 else 0), (tn/t_n*100 if t_n>0 else 0)
    fpr, fnr = (fp/t_n*100 if t_n>0 else 0), (fn/t_p*100 if t_p>0 else 0)

    f_tp, f_tn, f_fp, f_fn = [f'{val:,}' for val in [tp, tn, fp, fn]]

    stats_labels = [f'TP = {f_tp}; {tpr:.2f}%', f'TN = {f_tn}; {tnr:.2f}%',
            f'FP = {f_fp}; {fpr:.2f}%', f'FN = {f_fn}; {fnr:.2f}%']
    reg_label = f'Linear Reg:\nslope={m:.2f} $R^2$={r_sq:.2f}'

    full_labels = stats_labels + [reg_label]
    full_handles = [plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)] * 4 + \
                   [Line2D([0], [0], color='red', linestyle='--', linewidth=4)]

    leg = plt.legend(full_handles, full_labels, loc='upper left', fontsize=18,
                     frameon=True, facecolor='white', framealpha=1, edgecolor='black',
                     handlelength=1.5, handletextpad=0.5)

    for i, text in enumerate(leg.get_texts()):
        if i < 4:
            text.set_ha('left')
            text.set_position((-150, 0))

    plt.xlabel(r'DS$_{exp}$ [kcal/mol]', fontsize=18)
    plt.ylabel(r'DS$_{pred}$ [kcal/mol]', fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.xlim(lims); plt.ylim(lims)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

    # colorbar saving
    #save_colorbars(cmap)

def process_and_plot_folder(folder_path, filter_set=None):
    search_pattern = os.path.join(folder_path, "*.csv")
#    valid_files = [f for f in glob.glob(search_pattern) if re.search('split[1-5]',f)]
    valid_files = glob.glob(search_pattern)


    if not valid_files:
        print("There are no files.")
        return

    preds_acc, exp_vals = {}, {}
    for f_path in sorted(valid_files):
        df = _read_csv(f_path)
        if df is None: continue
        for _, row in df.iterrows():
            name = str(row.iloc[0])

            if name not in preds_acc:
                preds_acc[name] = []
                exp_vals[name] = float(row.iloc[1])
            preds_acc[name].append(float(row.iloc[2]))


    results = [{'name': k, 'expected': exp_vals[k], 'predicted': np.mean(v)}
               for k, v in preds_acc.items() if v]
    df_avg = pd.DataFrame(results)

    if not df_avg.empty:
        metrics = get_classification_metrics(df_avg)
        plot_dense_classification(df_avg, metrics, MAIN_OUTPUT)
        print(f"Done!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_and_plot_folder(sys.argv[1], None)
