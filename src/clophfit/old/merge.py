#!/usr/bin/python
#

import pandas as pd
import numpy as np
import sys

try:
    list_file = str(sys.argv[1])
    b = pd.read_table(list_file, header=None)
    b.columns = ['file']
    files = []
    for f in b['file']:
        tmp = pd.read_table(f, header=None, skiprows=2)
        tmp.columns = ['lambda', 'F']
        files.append(tmp)
    df = pd.DataFrame(
            np.transpose([f.F for f in files]),
            columns=['s'+str(i) for i in range(len(files))],
            index=files[0]['lambda'])
    df.to_csv('table.csv')

except:
    print('merge need a file named "list"')
    print(sys.argv[1])
    print(list_file)
