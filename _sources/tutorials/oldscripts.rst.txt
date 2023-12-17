    :Author: Daniele Arosio



Old scripts
-----------

``fit_titration.py``
~~~~~~~~~~~~~~~~~~~~

- input ← csvtable and note \_file

  - csvtable

  .. image:: ../_static/csvtable.png

  - note \_file

  .. image:: ../_static/note_file.png

- output → pK spK and pdf of analysis


It is a unique script for pK and Cl and various methods:

1. svd

2. bands

3. single lambda

and bootstrapping



I do not know how to unittest
TODO

- average spectra

- join spectra ['B', 'E', 'F']

- compute band integral (or sums)

``fit_titration_global.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A script for fitting tuples (y1, y2) of values for each concentration (x). It uses lmfit confint and bootstrap.

- input ← x y1 y2 (file)

  - file

  .. image:: ../_static/file.png

- output →

  - params: K SA1 SB1 SA2 SB2

  - fit.png

  - correl.png

It uses lmfit confint and bootstrap. In global fit the best approach was using lmfit without bootstrap.

.. code:: bash

    for i in *.dat; do gfit $i png2 --boot 99 > png2/$i.txt; done

IBF database uses
~~~~~~~~~~~~~~~~~

Bash scripts (probably moved into prtecan) for:

- ``fit_titration_global.py``

  - `../../src/clophfit/old/bash/fit.tecan <../../src/clophfit/old/bash/fit.tecan>`_

  - `../../src/clophfit/old/bash/fit.tecan.cl <../../src/clophfit/old/bash/fit.tecan.cl>`_

- ``fit_titration.py``

  .. code:: sh

      cd 2014-xx-xx

      (prparser) pr.enspire *.csv

      fit_titration.py meas/Copy_daniele00_893_A.csv A02_37_note.csv -d fit/37C | tee fit/svd_Copy_daniele00_893_A_A02_37_note.txt

      w_ave.sh > pKa.txt

      head pKa??/pKa.txt >> Readme.txt


      # fluorimeter data
      ls > list
      merge.py list
      fit_titration *.csv fluo_note

see: `/home/dati/ibf/IBF/Database/Data and protocols_Liaisan/library after Omnichange mutagenesis/Readme_howto.txt </home/dati/ibf/IBF/Database/Data and protocols_Liaisan/library after Omnichange mutagenesis/Readme_howto.txt>`_

new
---

- Tit works for a plate organized with a pH-value for each row and a Cl conc
  value for each column.

- Old fit works with a list of well and corresponding cl conc orpH value; the
  kind ["pH", "Cl"] decides

Some measurements e.g. "A", "C" and "D" are at T = 20 others ("B", "E" and "F")
are taken at T = 37.
This temp is present in each metadata. (e.g. measurements["A"]["metadata"]["Temp"]).

A future note file format:

- well, pH, Cl, T, mut, meas labels

  .. code:: sh

      echo "Well,pH,Cl,Name,Temp,Labels" > v224H_note.csv
      awk -F '\t' 'NR>1 {print $1 "," $2 "," $3 ",V224H,37.0,A B"}' v224H_note >> v224H_note.csv
