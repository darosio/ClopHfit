#!/usr/bin/python
#

import argparse
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns


def main():
    description = "plot a summary of a tecan fit analysis"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file',
                        help="file <well llower lower value upper uupper>")
    args = parser.parse_args()
    df = pd.read_table(args.file, sep=' ')
    # l = list(scheme.well[scheme.sample == 'buffer'])
    fig, ax = plt.subplots(figsize=(9, 12))
    # df.sort(columns=['lower'], inplace=True)
    N = len(df)
    ax.errorbar(df.pKa, range(N), xerr=[df.pKa-df.lower, df.upper-df.pKa], fmt='ok', ecolor='gray', alpha=0.5)
    ax.vlines([8.00], 0, N, linewidth=5, alpha=0.2)
    ax.set_xlabel("pH")
    ax.set_ylabel("mutant ID")
    ax.set_ylim([-1, N+1])
    # ax.set_xlim([4.5, 9.0])
    plt.show()


main()
