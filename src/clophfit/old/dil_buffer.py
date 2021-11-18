#!/usr/bin/python
#

import argparse
import os
import sys
import pandas as pd


corr = {'pH': [1, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12],
        'cl': [1.000000, 1.017857, 1.035714, 1.053571, 1.071429, 1.089286,
               1.107143, 1.160714, 1.196429]}

corr = {'pH': [1, 1.02, 1.04, 1.06, 1.08, 1.1],
        'cl': [1.000000, 1.017857, 1.035714, 1.053571, 1.071429, 1.089286,
               1.107143, 1.160714, 1.196429]}


def main():
    description = "make correction for dilution"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('scheme',
                        help="scheme.txt file")
    args = parser.parse_args()

    if not (os.path.exists('./pH') and os.path.exists('./cl')):
        print('must be in a meas folder after pr.tecan run list.pH and list.cl')
        sys.exit(1)

    scheme = pd.read_table(args.scheme)
    l = list(scheme.well[scheme.sample == 'buffer'])

    df = pd.read_csv('pH/' + l[0] + '.dat')
    for f in l[1:]:
        df = df.append(pd.read_csv('pH/' + f + '.dat'))
    print('pH')
    print('y1 :', df.y1.mean(), df.y1.std())
    print('y2 :', df.y2.mean(), df.y2.std())

    df = pd.read_csv('cl/' + l[0] + '.dat')
    for f in l[1:]:
        df = df.append(pd.read_csv('cl/' + f + '.dat'))
    print('cl')
    print('y1 :', df.y1.mean(), df.y1.std())
    print('y2 :', df.y2.mean(), df.y2.std())


main()
