prenspire
~~~~~~~~~

Here is the UML for the old implementation with EnSpireFiles class.

.. note::

    It can export spectra, grouped by pH values for Cl titrations, or at Cl=0
    for pH titrations, to ../Tables-v??.

:example:

Usage: enspireconvert  foo.csv  foo-note (--out Table)

.. warning::

   It was converting initial spectra like:

   + SpectraA  280 : 300 - 650
   + SpectraC  260 - (Max_A-20) : Max_A
   + SpectraB  Max_C : (Max_C+20) - 650


UML
^^^

.. include:: ./prenspire.uml.rst
