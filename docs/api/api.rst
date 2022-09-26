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

	LabelblocksGroup ..> Labelblock
    TecanfilesGroup ..> Tecanfile

	LabelblocksGroup "*" --o TecanfilesGroup
    TecanfilesGroup <|-- Titration
    Titration "1" --o TitrationAnalysis

	class LabelblocksGroup{
	  labelblocks: list[Labelblock]
	  +metadata: dict
	  +temperatures: Sequence[float]
	  +data: dict[str, list[float]]
	  {abstract} buffer: Optional[dict[str, list[float]]]
    }

	class TecanfilesGroup{
      filenames: list[str]
	  +metadata: dict
	  +labelblocksgroups: list[LabelblocksGroup]
    }

	class Titration{
	  listfile: str
	  +metadata: dict
	  +labelblocksgroups: list[LabelblocksGroup]
	  +conc: list[float]
	  +export_dat(path)
    }

	class TitrationAnalysis{
	  titration: Titration
	  schemefile: str | None
	  +scheme: pd.Series[Any]
	  +conc: Sequence[float]
	  +labelblocksgroups: list[LabelblocksGroup]
	  +additions: Sequence[float]
	  +subtract_bg()
	  +dilution_correction(additionsfile)
	  {static} +calculate_conc(additions, stock, ini=0)
	  +metadata_normalization()
	  +fit(kind, ini, fin, no_weight, tval=0.95)
	  +plot_k()
	  +plot_well()
	  +plot_all_wells()
	  +plot_ebar()
	  +print_fitting()
	  +plot_buffers()
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
