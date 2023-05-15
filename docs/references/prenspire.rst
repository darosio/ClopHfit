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

UML devel
^^^^^^^^^

.. uml::

   class EnSpireFiles {
    -_note
    -_ini
    -_fin
    +metadata_pre: [][]
    +data_list: [][]
    +metadata_post: [][]
    +well_list: list
    +spectrum_list: SpectrumList
    -_plate
    -__init__(csv_f, note_f)
    -_get_data_ini()
    -_check_lists()
    -_get_list_from_note()
    -_get_list_from_platemap()
   }

   class SpectrumList {
       +get_spectrum(): Spectrum
       +plot()
       +totable()
       +check_totable_ex_em()
       +check_totable_pH() still problems
       +check_totable_cl()
       +totablefile()
       +get_pH_titration
       +get_buffer
       +get_cl_titration
    }

   class Spectrum {
       +ex: []
       +em: []
       +y: []
       +pH
       +cl
       +npts
       +well_name
       +plot()
       +isEx()
       +isEm()
       +get_maxx()
    }

   EnSpireFiles "1" <-- "1" SpectrumList
   EnSpireFiles "1" o-- "*" Spectrum : "> read from"
