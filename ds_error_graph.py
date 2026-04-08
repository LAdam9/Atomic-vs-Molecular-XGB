#!/usr/bin/env python3

from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot
import logging
import glob


def main():
    logging.basicConfig(level=logging.INFO, filename='analyze.log', filemode='a', force=True)

    if len(sys.argv) < 2:
        print("Usage: python3 ds_error_graph.py /path/directory/")
        sys.exit(1)

    target_dir = sys.argv[1]
    DS_type_label = "DS$_{exp}$ [kcal/mol]"

    csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))

    if not csv_files:
        print(f"Error: No .csv file found: {target_dir}")
        sys.exit(1)

    print(f"Processing: {len(csv_files)} CSV files ...")
    run(csv_files, DS_type_label)
    return 0

def run(csv_files, DS_type_label):
    mse_all, sigma_all = [], []
    intervaly_all, mse_ds_all, rel_pocet_ds_all = [], [], []
    mu_ds_all, sigma_ds_all, fraction_over_2_ds_all = [], [], []

    for f in csv_files:
        c, AD, TF = read_csv(f)

        if c.size == 0:
            continue


        error, mse_val, sigma_val = errors(c, AD, TF)


        mse_all.append(mse_val)
        sigma_all.append(sigma_val)

        intervaly, m_ds, r_ds, mu_d, s_d, f_d = DS_gaussian(np.arange(-15, 2, 2), AD, error) #change interval if needed

        intervaly_all.append(intervaly)
        mse_ds_all.append(m_ds)
        rel_pocet_ds_all.append(r_ds)
        mu_ds_all.append(mu_d)
        sigma_ds_all.append(s_d)
        fraction_over_2_ds_all.append(f_d)

    save_plots(np.asfarray(mse_all), np.asfarray(sigma_all),
               np.array(intervaly_all), np.asfarray(mse_ds_all), np.asfarray(rel_pocet_ds_all),
               np.asfarray(mu_ds_all), np.asfarray(sigma_ds_all), DS_type_label)
    return 0


def read_csv(name_csv):
    sep = ' ' if "XG" in os.path.basename(name_csv) else ';'
    df = pd.read_csv(name_csv, sep=sep)


    data = df.values.transpose()
    return data[0], data[1].astype(float), data[2].astype(float)


def errors(name, original, predicted):
    error = np.subtract(original, predicted)
    mse = round(np.square(error).mean(), 3)
    sigma = np.std(error, ddof=1)
    return error, mse, sigma

def DS_gaussian(intervaly, original, error):
    ae = np.absolute(error)
    size = intervaly.size - 1
    sigma_ds, mu_ds, mse_ds = np.zeros(size), np.zeros(size), np.zeros(size)
    rel_pocet_ds, f_over_2 = np.zeros(size), np.zeros(size)

    for i in range(size):
        mask = (original > intervaly[i]) & (original <= intervaly[i+1])
        if i == 0: mask |= (original <= intervaly[i])
        if i == size - 1: mask |= (original > intervaly[i+1])

        idx = np.where(mask)
        if idx[0].size > 0:
            rel_pocet_ds[i] = idx[0].size / original.size
            sigma_ds[i] = np.std(error[idx], ddof=0)
            mu_ds[i] = np.mean(error[idx])
            mse_ds[i] = np.square(error[idx]).mean()
            f_over_2[i] = np.count_nonzero(ae[idx] > 2.0) / idx[0].size

    return intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, f_over_2

def save_plots(mse, sigma, intervaly, mse_ds, rel_pocet_ds, mu_ds, sigma_ds, label):
    intervaly_avg = intervaly.mean(axis=0)
    posun = np.abs(np.diff(intervaly_avg).mean()/2)

    m_avg, m_std = mse_ds.mean(axis=0), mse_ds.std(axis=0, ddof=1)
    r_avg, r_std = rel_pocet_ds.mean(axis=0), rel_pocet_ds.std(axis=0, ddof=1)
    mu_avg, mu_std = mu_ds.mean(axis=0), mu_ds.std(axis=0, ddof=1)
    s_avg, s_std = sigma_ds.mean(axis=0), sigma_ds.std(axis=0, ddof=1)

    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1, figsize=(5, 7), sharex=True)
    fig.subplots_adjust(left=0.18, bottom=0.12, right=0.82, top=0.95, hspace=0)

    # Linewidth
    lw = 1.5

    # Upper plot: MSE and Rel Abundance
    ax1.set_ylabel("MSE [(kcal/mol)$^2$] --●--", color='tab:blue', fontsize=12)
    ax1.errorbar(intervaly_avg[:-1]+posun, m_avg, yerr=m_std, fmt='--o', color='tab:blue', markersize=7.5, linewidth=lw, capsize=0)
    ax1.axhline(mse.mean(), linestyle='--', color='tab:blue', linewidth=lw)
    ax1.set_ylim(0, 5) #change lim if needed
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1b = ax1.twinx()
    ax1b.set_ylabel("relative abundance ··■··", color='tab:green', fontsize=12)
    ax1b.errorbar(intervaly_avg[:-1]+posun, r_avg, yerr=r_std, fmt=':s', color='tab:green', markersize=7.5, linewidth=lw, capsize=0)
    ax1b.set_ylim(0, 1)

    # Lower plot: mu and sigma
    ax2.set_xlabel(label, fontsize=12)
    ax2.set_ylabel("$\mu$ [kcal/mol] --●--", color='tab:blue', fontsize=12)
    ax2.errorbar(intervaly_avg[:-1]+posun, mu_avg, yerr=mu_std, fmt='--o', color='tab:blue', markersize=7.5, linewidth=lw, capsize=0, zorder=10)
    ax2.axhline(0.0, linestyle='--', color='tab:blue', linewidth=lw, zorder=1)
    ax2.set_ylim(-4, 4)


    ax2.set_yticks(ax2.get_yticks()[1:-1])

    ax2b = ax2.twinx()
    ax2b.set_ylabel("$\sigma$ [kcal/mol] ··■··", color='tab:green', fontsize=12)
    ax2b.errorbar(intervaly_avg[:-1]+posun, s_avg, yerr=s_std, fmt=':s', color='tab:green', markersize=7.5, linewidth=lw, capsize=0, zorder=5)
    ax2b.axhline(sigma.mean(), linestyle=':', color='tab:green', linewidth=lw)
    ax2b.set_ylim(0, 2)
    ax2b.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax2b.set_yticks(ax2b.get_yticks()[:-1])

    ax2.set_xlim(intervaly_avg[0], intervaly_avg[-1])
    ax2.set_xticks(intervaly_avg)
    ax2.set_xticklabels([int(x) for x in intervaly_avg])

    ax2.set_zorder(ax2b.get_zorder() + 1)
    ax2.patch.set_visible(False)

    plt_name = "result_error_graph.png"
    matplotlib.pyplot.savefig(plt_name, dpi=300)
    print(f"Done! Graph saved: {plt_name}")

if __name__ == "__main__":
    main()
