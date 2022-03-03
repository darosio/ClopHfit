#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy import optimize
import matplotlib.pyplot as plt

import seaborn


def main():
    """Fit pH and cl titrations where data are spectra.
    input: spectra_table.csv; _note
    output: pK; spK; and plot (*.pdf)
    """
    description = "Fit spectra from a pH or a Cl titration"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('csvtable',
                        help="a table containing spectra arranged in columns")
    parser.add_argument('note_file',
                        help="a file to describe the titration")
    parser.add_argument("-d", '--out', dest='out',
                        help="destination directory (default: .)")
    parser.add_argument("-m", "--method-of-analysis", action="store",
                        default="svd", choices=["svd", "band"],
                        dest='analysis_method')
    parser.add_argument("-t", "--titration-of", action="store", default="pH",
                        choices=["pH", "cl"], dest='titration_type')
    parser.add_argument("-b", "--band-interval", action="store", nargs=2,
                        type=int, dest='band',
                        help='Integration interval from <1> to <2>')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    # input of spectra (csv) and titration (note) data
    csv = pd.read_csv(args.csvtable)
    note_file = pd.read_table(args.note_file)
    note_file = note_file[note_file['mutant'] != 'buffer']
    # TODO aggregation logic for some pH or cloride
    Note = namedtuple("note", "wells conc")
    # pH or Cl titration
    if args.titration_type == "cl":
        note = Note(list(note_file.well), list(note_file.Cl))

        def fz(Kd, p, x):
            return ((p[0] + p[1] * x / Kd) / (1 + x / Kd))
    elif args.titration_type == "pH":
        note = Note(list(note_file.well), list(note_file.pH))

        def fz(pK, p, x):
            return (p[1] + p[0] * 10 ** (pK - x)) / (1 + 10 ** (pK - x))
    df = csv[note.wells]
    df.index = csv['lambda']
    conc = np.array(note.conc)
    # sideeffect print input data
    if args.verbose:
        print(csv)
        print(note)
        print('conc vector\n', conc)
        print('DataFrame\n', df)
    if args.analysis_method == 'svd':
        # svd on difference spectra
        ddf = df.sub(df.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        # fitting
        result = fit_titration(fz, conc, v[0, :])
        # plotting
        seaborn.set_style('ticks')
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_axes([0.05, 0.65, 0.32, 0.31])
        plt.grid(True)
        ax2 = fig1.add_axes([0.42, 0.65, 0.32, 0.31])
        plt.grid(True)
        ax1.plot(df)
        ax2.plot(ddf.index, u[:, 0], 'k-', lw=3)
        ax2.plot(ddf.index, u[:, 1], 'b--')
        ax3 = fig1.add_axes([0.80, 0.65, 0.18, 0.31], yscale='log',
                            xticks=[1, 2, 3, 4], title='autovalues')
        ax3.bar([1, 2, 3, 4], (s**2 / sum(s**2))[:4],
                align='center', alpha=0.7, width=0.66)
        ax4 = fig1.add_axes([0.05, 0.08, 0.50, 0.50], title="fitting")
        ax5 = fig1.add_axes([0.63, 0.08, 0.35, 0.50],
                            title='SVD coefficients',
                            xlabel='1$^{st}$ autovector',
                            ylabel='2$^{nd}$ autovector')
        ax4.scatter(conc, v[0, :])
        xmin = conc.min()
        xmax = conc.max()
        xmax += (xmax - xmin) / 7
        xlin = np.linspace(xmin, xmax, 100)
        ax4.plot(xlin, fz(result.K, [result.SA, result.SB], xlin))
        result.s1 = str(round(result.K, 2)) + ' \u00B1 '
        result.s1 = result.s1 + str(round(result.sK, 2))
        plt.figtext(.26, .54, result.s1, size=20)
        ax5.plot(v[:, 1], v[:, 2], lw=0.8)
        for x, y, l in zip(v[:, 1], v[:, 2], note.wells):
            ax5.text(x, y, l)
        if 1 == 1:
            kd = []
            sa = []
            sb = []
            for i in range(100):
                boot_idxs = np.random.randint(0, len(ddf.columns) - 1,
                                                      len(ddf.columns))
                ddf2 = df.iloc[:, boot_idxs]
                conc2 = conc[boot_idxs]
                u, s, v = np.linalg.svd(ddf2)
                result2 = fit_titration(fz, conc2, v[0, :])
                kd.append(result2.K)
                sa.append(result2.SA)
                sb.append(result2.SB)
            bs = pd.DataFrame({'kd': kd, 'SA': sa, 'SB': sb})
            bs.to_csv('bs.txt')
    elif args.analysis_method == 'band':
        # fitting
        try:
            ini = args.band[0]
            fin = args.band[1]
            y = []
            for c in df.columns:
                y.append(df[c].loc[ini: fin].sum())
            result = fit_titration(fz, conc, y)
        except:
            print('''bands [{0}, {1}] not in index.
                  Try other values'''.format(ini, fin))
            raise
        # plotting
        fig1 = plt.figure(figsize=(12, 8))
        ax4 = fig1.add_axes([0.05, 0.08, 0.50, 0.50], title="fitting")
        ax4.scatter(conc, y)
        xmin = conc.min()
        xmax = conc.max()
        xmax += (xmax - xmin) / 7
        xlin = np.linspace(xmin, xmax, 100)
        ax4.plot(xlin, fz(result.K, [result.SA, result.SB], xlin))
        result.s1 = str(round(result.K, 2)) + ' \u00B1 '
        result.s1 = result.s1 + str(round(result.sK, 2)) + '[' + str(ini)
        result.s1 = result.s1 + ':' + str(fin) + ']'
        plt.figtext(.26, .54, result.s1, size=20)
        for x, y, l in zip(conc, y, note.wells):
            ax4.text(x, y, l)

    # output
    f_csv_shortname = os.path.splitext(os.path.split(args.csvtable)[1])[0]
    f_note_shortname = os.path.splitext(os.path.split(args.note_file)[1])[0]
    f_out = f_csv_shortname.join([args.analysis_method + "_", "_"])
    f_out = f_out + f_note_shortname
    if args.out:
        if not os.path.isdir(args.out):
            os.makedirs(args.out)
        f_out = os.path.join(args.out, f_out)
    fig1.savefig(f_out + ".pdf")

    print("best-fitting using: ", args.analysis_method)
    print("spectra csv file: ", f_csv_shortname)
    print("note file: ", f_note_shortname)
    print("K = ", round(result.K, 3))
    print("sK = ", round(result.sK, 3))
    print("SA = ", round(result.SA, 3))
    print("sSA = ", round(result.sSA, 3))
    print("SB = ", round(result.SB, 3))
    print("sSB = ", round(result.sSB, 3))


def fit_titration(fz, x, y):
    """Fit a dataset (x, y) using a single-site binding model provided by the
    function **fz** that defines a constant *K* and 2 plateau *SA* and *SB*.

    Parameters
    ----------

    x, y : numpy arrays

    Return
    ------

    res : namedtuple
        contains the least square results

    """
    def ssq(p, x, y1):
        return np.r_[y1 - fz(p[0], p[1:3], x)]
    # plateau calculation
    df = pd.DataFrame({'x': x, 'y': y})
    SA = df.y[df.x == min(df.x)].values[0]
    SB = df.y[df.x == max(df.x)].values[0]
    p0 = np.r_[7.1, SA, SB]
    p, cov, info, msg, success = optimize.leastsq(ssq, p0, args=(x, y),
                                                  full_output=True, xtol=1e-11)
    # ftol=1.49012e-23,
    res = namedtuple("Result", "success msg df chisqr K sK SA sSA SB sSB")
    res.msg = msg
    res.success = success
    if 1 <= success <= 4:
        chisq = sum(info['fvec'] * info['fvec'])
        res.df = len(y) - len(p)
        res.chisqr = chisq / res.df
        res.K = p[0]
        res.sK = np.sqrt(cov[0][0] * res.chisqr)
        res.SA = p[1]
        res.sSA = np.sqrt(cov[1][1] * res.chisqr)
        res.SB = p[2]
        res.sSB = np.sqrt(cov[2][2] * res.chisqr)
    return res


main()
