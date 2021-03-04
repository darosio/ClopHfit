#!/usr/bin/bash
#

mkdir png;
for i in *.dat; 
    do fit_titration_global.py $i png -t cl > png/$i.txt; 
    done
