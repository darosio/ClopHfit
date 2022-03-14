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

   Labelblock "*" --o Tecanfile

   class Labelblock{
    KEYS
    metadata : dict
    data : dict {'H12':float}
    __init__(parent_tecanfile, lines)
    __eq__(KEYS)
    }
   class Tecanfile{
    path : str
    metadata : dict
    labelblocks : list of Labelblock
    __init__(path)
    +read_xls(path)
    +lookup_csv_lines(csvl, pattern, col):
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
   class LabelblocksGroup{
    metadata : dict
    temperatures : list of float
    data : dict of list of float {'H12':[]}
    __init__(labelblocks)
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
