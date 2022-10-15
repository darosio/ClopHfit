===
API
===

This part of the documentation lists the full API reference of all public
classes and functions.

clophfit.binding module
++++++++++++++++++++++++

.. automodule:: clophfit.binding
   :undoc-members:

clophfit.prtecan
++++++++++++++++

.. automodule:: clophfit.prtecan
   :no-members:
   :noindex:

module functions
-------------------
.. autosummary::

   clophfit.prtecan.fit_titration
   clophfit.prtecan.fz_kd_singlesite
   clophfit.prtecan.fz_pk_singlesite
   clophfit.prtecan.extract_metadata
   clophfit.prtecan.strip_lines
   clophfit.prtecan._merge_md

classes to parse a file
-----------------------
.. autosummary::

   clophfit.prtecan.Metadata
   clophfit.prtecan.Labelblock
   clophfit.prtecan.Tecanfile

classes to group files
----------------------
.. autosummary::

   clophfit.prtecan.LabelblocksGroup
   clophfit.prtecan.TecanfilesGroup
   clophfit.prtecan.Titration
   clophfit.prtecan.TitrationAnalysis

.. include:: ../../docs/api/prtecan.uml.rst

.. autoclass:: Metadata

.. autoclass:: Labelblock
   :special-members: __almost_eq__, __eq__
   :exclude-members: lines
.. autoclass:: Tecanfile
   :exclude-members: path

.. autoclass:: LabelblocksGroup
.. autoclass:: TecanfilesGroup
.. autoclass:: Titration
   :show-inheritance: True
.. autoclass:: TitrationAnalysis

.. autofunction:: strip_lines
.. autofunction:: extract_metadata
.. autofunction:: _merge_md
.. autofunction:: fz_pk_singlesite
.. autofunction:: fz_kd_singlesite
.. autofunction:: fit_titration

..
   .. automodule:: clophfit.prtecan
	  :exclude-members: Labelblock,
