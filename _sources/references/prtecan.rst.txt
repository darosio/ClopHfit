prtecan
~~~~~~~

**Tecan** file has the following structure.

	+------------+
	|  metadata  |
	+------------+
	|  label 1   |
	+------------+

	+------------+
	|  label N   |
	+------------+

Each **Labelblock** contains:
- metadata describing the measurement type and condition;
- 96 (float) values, e.g. excitation emission filter-based fluorescence.

A format that can be formalized using the following Backus-Naur form:

::

  TF         ::= LABELBLOCK
  PRE        ::=
  LABELBLOCK ::= 'Label: Label'\d+ '\n' ATTRIBUTE+ PLATE ATTRIBUTE+

XXX to be completed...


Object-Oriented Analysis (OOA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Object-Oriented Design (OOD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Metadata class has two private attributes: value and unit. It is used to store metadata values and their corresponding units.

The Labelblock class has several private attributes: lines, path, data, metadata, data_norm, buffer_wells, data_buffersubtracted, data_buffersubtracted_norm, buffer, buffer_sd, buffer_norm, and buffer_norm_sd. It represents a block of labels and its associated data. It has several public methods, including __eq__(), __almost_eq__(), and others.

The Tecanfile class has two private attributes: path and metadata, and a public attribute labelblocks. It represents a Tecan file, and has a one-to-many relationship with Labelblock. It also has a one-to-many relationship with Metadata.

The LabelblocksGroup class has several private attributes: labelblocks, allequal, metadata, data, data_norm, buffer_wells, data_buffersubtracted, and data_buffersubtracted_norm. It represents a group of Labelblock objects and their associated data. It has several public methods.

The TecanfilesGroup class has several private attributes: tecanfiles, labelblocksgroups, and metadata. It represents a group of Tecanfile objects, and has a one-to-many relationship with Tecanfile and LabelblocksGroup.

The Titration class has several private attributes: tecanfiles, conc, additions, buffer_wells, data, and data_nrm. It represents a titration experiment and its associated data. It has several public methods, including load_additions(), and export_data().

The PlateScheme class has several private attributes: file, buffer, crtl, and names. It represents a scheme for a plate, and has several public methods.

The TitrationAnalysis class has several private attributes: scheme, datafit, and heys. It represents an analysis of a titration experiment, and has several public methods, including load_scheme(), fit(), and plot_k(), among others.

Overall, the UML scheme defines several classes that are used to represent a titration experiment, its associated data, and the analysis of the experiment. These classes are organized into groups and have relationships with each other to form a cohesive object-oriented design.

UML scheme
''''''''''

.. include:: ./prtecan.uml.rst
