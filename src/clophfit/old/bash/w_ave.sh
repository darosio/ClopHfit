#!/usr/bin/bash
#

head -5 fit/37C/*.txt | grep  "^K =" | awk '{print $3}' > tmpx
head -5 fit/37C/*.txt | grep  "^sK =" | awk '{print $3}' > tmps

paste tmpx tmps | \
    awk '{sx+=$1/$2^2; ss+=1/$2^2}END{
            printf("K\tsK\ttemp")
            printf("\n%.3g\t%.2g\t%.1f\n", sx/ss, 1/ss^0.5, 37)}'

head -5 fit/20C/*.txt | grep  "^K =" | awk '{print $3}' > tmpx
head -5 fit/20C/*.txt | grep  "^sK =" | awk '{print $3}' > tmps

paste tmpx tmps | \
    awk '{sx+=$1/$2^2; ss+=1/$2^2}END{
            printf("%.3g\t%.2g\t%.1f\n", sx/ss, 1/ss^0.5, 20)}'

rm tmpx tmps
