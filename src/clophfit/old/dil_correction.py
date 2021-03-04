#!/usr/bin/python
#

import argparse
import os
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
    parser.add_argument('ffile',
                        help="a file x y1 y2 --> y1*corr y2*corr")
    parser.add_argument('out',
                        help="destination directory")
    parser.add_argument('buffer', nargs=2,
                        help="buffer values for y1 y2")
    parser.add_argument("-t", "--titration-of", action="store", default="pH",
                        choices=["pH", "cl"], dest='titration_type')
    args = parser.parse_args()
    df = pd.read_csv(args.ffile)
    df.y1 -= float(args.buffer[0])
    df.y2 -= float(args.buffer[1])
    df.y1 *= corr[args.titration_type]
    df.y2 *= corr[args.titration_type]
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    df.to_csv(os.path.join(args.out, args.ffile), index=False)


main()
