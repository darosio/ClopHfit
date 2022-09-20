===
API
===

This part of the documentation lists the full API reference of all public
classes and functions.

clophfit.binding module
-----------------------

.. automodule:: clophfit.binding
   :members:
   :undoc-members:
   :show-inheritance:

clophfit.prtecan
----------------

.. automodule:: clophfit.prtecan

.. _prtecan parse:

Classes to parse a file
~~~~~~~~~~~~~~~~~~~~~~~

.. uml::

   left to right direction

   Labelblock "1..*" --o Tecanfile

   class Labelblock{
     tecanfile: Tecanfile | None
     lines: list_of_lines
     +metadata : dict
     +data : dict {'H12':float}
	 __eq__()
     __almost_eq__()
    }

	class Tecanfile{
	  path : str
	  +metadata : dict
	  +labelblocks : list
	  {static} +read_xls()
	  {static} +lookup_csv_lines()
	  ~__hash__()
    }


.. autoclass:: Labelblock
   :members:
   :special-members: KEYS, __eq__
   :private-members:
   :undoc-members:


.. autoclass:: Tecanfile
   :members:
   :noindex:

.. _prtecan group:

Classes to group files
~~~~~~~~~~~~~~~~~~~~~~

.. uml::

    LabelblocksGroup "*" --o TecanfilesGroup
    TecanfilesGroup <|-- Titration
    Titration <|-- TitrationAnalysis

	class LabelblocksGroup{
	  labelblocks: list[Labelblock]
	  +metadata: dict
	  +temperature: Sequence[float]
	  +data: dict[str, list[float]]
	  {abstract} buffer: dict[str, list[float]]
    }

   class Titration{
    conc : list of float
    labelblocksgroups : list of LabelblocksGroup
    __init__(listfile)
    export_dat(path)
    }
   class TitrationAnalysis{
    scheme : DataFrame
    conc : list of float
    labelblocksgroups : list of LabelblocksGroup
    __init__(titration, schemefile)
    dilution_correction(additionsfile)
    metadata_normalization()
    subtract_bg()
    +calculate_conc(additions, conc_stock)
    fits(............)
    }
   class TecanfilesGroup{
    labelblocksgroups : list of LabelblocksGroup
    __init__(filenames)
    }

.. autoclass:: LabelblocksGroup
   :members:
   :show-inheritance:

.. autoclass:: TecanfilesGroup
   :members:
   :show-inheritance:

.. autoclass:: Titration
   :members:
   :show-inheritance:

.. autoclass:: TitrationAnalysis
   :members:
   :show-inheritance:
   :noindex:
