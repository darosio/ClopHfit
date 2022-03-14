Description of prtecan parser
=============================

**Tecan** file has the following structure.

	+------------+
	|  metadata  |
	+------------+
	|  label 1   |
	+------------+

	+------------+
	|  label N   |
	+------------+

Each **Labelblock** contains: metadata and 96 (float) values, e.g.
excitation emission filter-based fluorescence.

A format that can be formalized using the following Backus-Naur form:

::

  TF         ::= LABELBLOCK
  PRE        ::=
  LABELBLOCK ::= 'Label: Label'\d+ '\n' ATTRIBUTE+ PLATE ATTRIBUTE+

XXX to be completed...

OOA
...

The program need to be able to:

+ *group* **Data** into 96 dataframes. A single measurement_condition is
  changed (titrated) over the elements of a group (**TecanfilesGroup**,
  **LabelblockGroup**) and stored into **Titration**.

+ *export* **Titration** with the properly associated
  **Fitting_function** to *obtain*: fitting_parameters and
  fitting_quality.

In the case of multi-label file each well contains more than a single
measurement under the same measurement_conditions (e.g. ex400nm and
ex485nm). Each measurement will be exported into a separate column and
Titration could be **fitted globally**.


OOD
...

**Labelblock** contains:

+ metadata describing the measurement type and condition.
+ a dictionary for data with a key for each well.
