#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy import optimize
import matplotlib.pyplot as plt


def main():
    """titration fit of spectra
    input: spectra_table.csv; _note
    output: pK; spK; and plot (*.png)
    """
    description = "Fit x,y1,y2 file for pH or Cl titration"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file',
                        help="x,y1,y2 file without header"
                        )
    parser.add_argument("-d", '--out',
                        dest='out',
                        #default='Fit',
                        help="destination directory (default: Meas)"
                        )
    parser.add_argument( "-t", "--titration-of",
                        action="store", default="pH", 
                        choices=["pH", "cl"], dest='titration_type')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    df = pd.read_csv('A01.dat', sep=' ', names=['x', 'y1', 'y2'])


    import rpy2
    import pandas.rpy.common as com
    from rpy2.robjects import r

    from rpy2.robjects.packages import importr

    from rpy2.robjects import globalenv
    r_df = com.convert_to_r_dataframe(df)
    globalenv['r_df'] = r_df

    fit = r('nls(y1 ~ (SB + SA * 10 ** (K - x)) / (1 + 10 ** (K - x)), start = list(SA=20, SB=40000, K=7), data=r_df)')
    print(r.confint(fit))

    print(r.coef(fit))
    print(r.summary(fit))
    nlstools = importr('nlstools')
    nb = nlstools.nlsBoot(fit, niter=1000)
    r.summary(nb)
    gr = importr('grDevices')
    r.plot(nb)
    input()

    print(df)
    sys.exit(1)

    csv = pd.read_csv(args.csvtable[0])
    note_file = pd.read_table(args.note_file[0])
    note_file = note_file[note_file['mutant'] != 'buffer']

    Note = namedtuple("note", "wells conc")

    # TODO aggregation logic for some pH or cloride

    if args.titration_type == "cl":
        note = Note(list(note_file.well), list(note_file.Cl))
        def fz(Kd, p, x):
            return ((p[0] + p[1] * x / Kd) / (1 + x / Kd))

    if args.titration_type == "pH":
        note = Note(list(note_file.well), list(note_file.pH))
        def fz(pK, p, x):
                return (p[1] + p[0] * 10 ** (pK - x)) / (1 + 10 ** (pK - x)) 

    if args.verbose:
        print(csv)
        print(note)
    
    df = csv[note.wells] 
    df.index = csv['lambda']

    ddf = df.sub(df.icol(0), axis=0)
    u,s,v = np.linalg.svd(ddf)
    
    xx=np.array(note.conc)
    res = fit_titration(fz, xx, v[0,:])
    # output
    f_csv_shortname = os.path.splitext(os.path.split(f_csv)[1])[0]
    f_note_shortname = os.path.splitext(os.path.split(f_note)[1])[0]
    f_out = f_csv_shortname.join([args.analysis_method + "_", "_"]) \
            + f_note_shortname
    if args.out:
        if not os.path.isdir(args.out):
            os.makedirs(args.out)
        f_out = os.path.join(args.out, f_out)
    print("best-fitting using: ", args.analysis_method)
    print("spectra csv file: ", f_csv_shortname)
    print("note file: ", f_note_shortname)
    print("K = ", round(res.K, 3))
    print("sK = ", round(res.sK, 3))
    print("SA = ", round(res.SA, 3))
    print("sSA = ", round(res.sSA, 3))
    print("SB = ", round(res.SB, 3))
    print("sSB = ", round(res.sSB, 3))

    # Plotting
    import seaborn
    seaborn.set_style('ticks')
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_axes([0.05, 0.65, 0.32, 0.31])
    plt.grid(True)
    ax2 = fig1.add_axes([0.42, 0.65, 0.32, 0.31])
    plt.grid(True)
    ax1.plot(df.index, df)
    ax2.plot(ddf.index, u[:, 0], 'k-', lw=3)
    ax2.plot(ddf.index, u[:, 1], 'b--')
    ax3 = fig1.add_axes([0.80, 0.65, 0.18, 0.31], yscale='log',
            xticks=[1,2,3,4], title='autovalues')
    ax3.bar([1,2,3,4],(s**2 / sum(s**2))[:4], align='center', alpha=0.7, width=0.66)
    ax4 = fig1.add_axes([0.05, 0.08, 0.50, 0.50], title="fitting")
    ax5 = fig1.add_axes([0.63, 0.08, 0.35, 0.50], title='SVD coefficients',
                    xlabel='1$^{st}$ autovector', ylabel='2$^{nd}$ autovector')
    ax4.scatter(xx, v[0,:])
    xmin = xx.min()
    xmax = xx.max()
    xmax += (xmax - xmin) / 7
    xlin = np.linspace(xmin, xmax, 100)
    ax4.plot(xlin, fz(res.K, [res.SA, res.SB], xlin))
    res.s1 = str(round(res.K, 2)) + ' \u00B1 ' + str(round(res.sK, 2))
    plt.figtext(.26,.54,res.s1, size=20)
    ax5.plot(v[:, 1], v[:, 2], lw=0.8)
    for x,y,l in zip(v[:, 1], v[:, 2], note.wells):
        ax5.text(x,y,l)
    fig1.savefig(f_out + ".pdf")


def fit_titration(fz, x, y):
    ''' Fit a dataset (x, y) using a single-site binding model provided by the
    function **fz** that defines a constant *K* and 2 plateau *SA* and *SB*.
    '''

    y1 = np.array(y)
    def ssq(p, x, y1):
        return np.r_[y1 - fz(p[0], p[1:3], x)]
    p0 = np.r_[7.1, y1[0], y1[-1]]
    p, cov, info, msg, success = optimize.leastsq(ssq, p0, args=(x, y1),
            full_output=True, xtol=1e-11)  # ftol=1.49012e-23,
    res = namedtuple("Result",
        "success msg df chisqr K sK SA sSA SB sSB")
    res.msg = msg
    res.success = success
    if 1 <= success <= 4:
        chisq = sum(info['fvec'] * info['fvec'])
        res.df = len(y1) - len(p)
        res.chisqr = chisq / res.df
        res.K = p[0]
        res.sK = np.sqrt(cov[0][0] * res.chisqr)
        res.SA = p[1]
        res.sSA = np.sqrt(cov[1][1] * res.chisqr)
        res.SB = p[2]
        res.sSB = np.sqrt(cov[2][2] * res.chisqr)
    return res



main()

