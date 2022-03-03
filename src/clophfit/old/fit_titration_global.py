#!/usr/bin/env python

"""Module for global fitting titrations (pH and cl) on 2 datasets
"""
import os
import sys
import argparse
import numpy as np
from lmfit import Parameters, Minimizer, minimize, conf_interval, report_fit
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import optimize


def ci_report(ci):
    """return text of a report for confidence intervals"""
    maxlen = max([len(i) for i in ci])
    buff = []
    add = buff.append
    convp = lambda x: ("%.2f" % (x[0]*100))+'%'
    # I modified "%.5f"
    conv = lambda x: "%.6G" % x[1]
    title_shown = False
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen)] +
                        [i.rjust(10) for i in map(convp, row)]))
            title_shown = True
        add("".join([name.rjust(maxlen)] +
                    [i.rjust(10) for i in map(conv,  row)]))
    return '\n'.join(buff)


def residual(pars, x, data=None, titration_type=None):
    """residual function for lmfit
    Parameters
    ----------
    pars: lmfit Parameters()
    x : list of x vectors
    data : list of y vectors

    Return
    ------
    a vector for the residues (yfit - data)
    or the fitted values
    """
    vals = pars.valuesdict()
    SA1 = vals['SA1']
    SB1 = vals['SB1']
    K = vals['K']
    SA2 = vals['SA2']
    SB2 = vals['SB2']
    if titration_type == 'pH':
        model1 = (SB1 + SA1 * 10 ** (K - x[0])) / (1 + 10 ** (K - x[0]))
        model2 = (SB2 + SA2 * 10 ** (K - x[1])) / (1 + 10 ** (K - x[1]))
    elif titration_type == 'cl':
        model1 = (SA1 + SB1 * x[0] / K) / (1 + x[0] / K)
        model2 = (SA2 + SB2 * x[1] / K) / (1 + x[1] / K)
    else:
        print('Error: residual call must indicate a titration type')
        sys.exit()
    if data is None:
        return np.r_[model1, model2]
    return np.r_[model1 - data[0], model2 - data[1]]


def main():
    description = "Fit a pH or Cl titration file: x y1 y2"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file',
                        help='the file <x y1 y2> without heads')
    parser.add_argument('out_folder',
                        help='The folder to output the .txt and .png files')
    parser.add_argument('-t', '--titration-of', dest='titration_type',
                        action="store", default="pH", choices=["pH", "cl"],
                        help='Type of titration, pH or cl')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Printout runtime information.increase verbosity')
    parser.add_argument('--boot', dest='nboot', type=int,
                        help='bootstraping using <n> iterations')
    args = parser.parse_args()
    ttype = args.titration_type
    #df = pd.read_csv(args.file, sep=' ', names=['x', 'y1', 'y2'])
    df = pd.read_csv(args.file)
    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)
    fit_params = Parameters()
    fit_params.add('SA1', value=df.y1[df.x == min(df.x)].values[0], min=0)
    fit_params.add('SB1', value=df.y1[df.x == max(df.x)].values[0], min=0)
    fit_params.add('SA2', value=df.y2[df.x == min(df.x)].values[0], min=0)
    fit_params.add('SB2', value=df.y2[df.x == max(df.x)].values[0], min=0)
    if args.titration_type == "pH":
        fit_params.add('K', value=7, min=4, max=10)
    elif args.titration_type == "cl":
        fit_params.add('K', value=20, min=0, max=1000)
    mini = Minimizer(residual, fit_params, fcn_args=([df.x, df.x],),
                   fcn_kws={'data': [df.y1, df.y2], 'titration_type': ttype})
    res = mini.minimize()
    report_fit(fit_params)
    ci = conf_interval(mini, res, sigmas=[.674, .95])
    print(ci_report(ci))
    # plotting
    xfit = np.linspace(df.x.min(), df.x.max(), 100)
    yfit = residual(fit_params, [xfit, xfit], titration_type=ttype)   # kws={}
    yfit = yfit.reshape(2, len(yfit) // 2)
    plt.plot(df.x, df.y1, 'o', df.x, df.y2, 's', xfit, yfit[0], '-',
             xfit, yfit[1], '-')
    plt.grid(True)
    f_out = os.path.join(args.out_folder, os.path.split(args.file)[1])
    plt.savefig(f_out + ".png")
    if args.nboot:
        bootstrap(df, args.nboot, fit_params, f_out, ttype)


def bootstrap(df, nboot, fit_params, f_out, ttype):
    """Perform bootstrap to estimate parameters variance
    Parameters
    ----------
    df : DataFrame
    nboot : int
    fit_params: lmfit.fit_params
    f_out : string

    Output
    ------
    print results
    plot
    """
    import seaborn as sns
    n_points = len(df)
    kds = []
    sa1 = []
    sb1 = []
    sa2 = []
    sb2 = []
    for i in range(nboot):
        boot_idxs = np.random.randint(0, n_points-1, n_points)
        df2 = df.loc[boot_idxs]
        df2.reset_index(drop=True, inplace=True)
        boot_idxs = np.random.randint(0, n_points-1, n_points)
        df3 = df.loc[boot_idxs]
        df3.reset_index(drop=True, inplace=True)
        try:
            res = minimize(residual, fit_params, args=([df2.x, df3.x],),
                     kws={'data': [df2.y1, df3.y2], 'titration_type': ttype})
            kds.append(res.params['K'].value)
            sa1.append(res.params['SA1'].value)
            sb1.append(res.params['SB1'].value)
            sa2.append(res.params['SA2'].value)
            sb2.append(res.params['SB2'].value)
        except:
            print(df2)
            print(df3)

    dff = pd.DataFrame({'K': kds, 'SA1': sa1, 'SB1': sb1, 'SA2': sa2,
                        'SB2': sb2})
    print("bootstrap: ",
          round(dff.K.quantile(.025), 3),
          round(dff.K.quantile(.163), 3),
          round(dff.K.median(), 3),
          round(dff.K.quantile(.837), 3),
          round(dff.K.quantile(.975), 3))
    sns.set_style('darkgrid')
    g = sns.PairGrid(dff)
    # g.map_diag(sns.kdeplot, lw=3)
    g.map_diag(plt.hist, alpha=0.4)
    g.map_upper(plt.scatter, s=9, alpha=0.6)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    plt.savefig(f_out + "-bs" + ".png")


if __name__ == '__main__':
    main()
