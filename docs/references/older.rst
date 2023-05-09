Older Description
===================

Plate Reader data Parser (and some analysis). To parse::

 1) **Tecan** data files, containing single fluorescence values  (scalar).
    Data are exported (from Tecan) into .xls file, then converted to csv
    (e.g. soffice --convert-to csv -outdir tmp *.xls).

    Each **Labelblock** contains 96 single (float) values,
    i.e. excitation emission filter-based fluorescence.

 2) **EnSpire** data files, containing spectra of various types e.g.
    excitation, emission, absorption and possibly at different temperatures.

OOA
---

The program need to be able to:

  + *read* csv file(s) and an **experimental_note** file
  + *fill* **Data** into a list (DB in future?)
  + *group* **Data** into **Titration** (in which a single measurement_condition is changed)
  + *fit* **Titration** with the proper **Fitting_function** and obtain: fitting_params, fitting_quality.

.. note::
  Also a well may contain more than a single measure under the same
  measurement_conditions (ex400nm and ex485nm), which could be
  **fitted globally**. Temperature scan is when temperature in
  *measurement_conditions* is changed; here it might be necessary to correct
  pH_value accordingly (with temperature changes).

OOD
---

**Data** contains:

  + a (tuple) of values or a (tuple) of spectra
  + metadata describing the data type
  + measurement_conditions like pH_value, Cl_conc, temperature
  + sample_information like mutant_name, prep_id.

.. uml::

   class Data{
       + meas_conditions
       + sample_info
       + data_type
       + data: tuple
   }
   class meas_conditions {
       + ph
       + cl
       + temp
   }
   class sample_info {
       + mutant
       + prep ID
   }
   Data o--- meas_conditions
   Data o-- sample_info

Outline
=======

.. I write outline in the opening of a module *.py

Basically, EnSpire exports file containing rows like:

[Well, Sample,   MeasA:Result,   MeasA:WavelengthExc,  MeasA:WavelengthEms]

Here is the UML for the implementation with TecanFile class.

.. uml::

   TecanFile "*" o-- "*" Titration
   TecanFile o-- Labelblock

   TecanFile : all_tecanfiles
   TecanFile : cls_data{A01: Titration}
   TecanFile : filename
   TecanFile : data {A01: L1,L2}
   TecanFile : _idxs []
   TecanFile : _list_labelblocks []
   TecanFile : isEqual(self.metadata==other.metadata)
   TecanFile : get_cls_data()

   Titration : conc array
   Titration : data
   Titration : fits()

   class Labelblock {
    metadata {}
    data []
    }

prparser.enspire
----------------

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


.. uml::

   class EnspireFile {
    +metadata: {}
    +measurements: {}
    +wells: []
    +samples: []
    -ini
    -fin
    -data_list
    -metadata_post
    -well_list_platemap: []
    -platemap: [][]
    __init__(csv_f)
    extract_measurements()
    export_measurements()
   }

   class ExpNote {
       + note_list: [][]
       + wells: []
       + titrations: []
       __init__(note_file)
       + check_wells(EnspireFile)
       + build_titrations()
   }

   class Titration {
    +conc: []
    +data: {}
    +fitting_func
    +results: ?
    __init__(conc, data, **kwargs)
    +fit()
   }

   Titration "*" --* "1" ExpNote  : < extract_titrations()
   EnspireFile "1" *-- "1" measurements : > extract_measurements()
   measurement "*" -* "1" measurements
   Titration "*" --o "1" globTitration
   data "1" --* "1" Titration

   class data << (D,orchid) >> {
       "A": DataFrame(index=lambda, columns=[conc, well])
       ..
   }

   class measurements << (D,orchid) >> {
       "A": measurement
       .
       .
   }

   class measurement << (D,orchid) >> {
       "metadata": {}
       "lambda": []
       "A01": [y]
       .
       "H12": [y]
   }


notes for future (maybe)
------------------------

.. note:: finish CONC CLASS
       ::

         def __init__(self, c, a, ca = 1000, r, vini = 1960):
            v = vini + a.cumsum()
            c = np.zeros(len(a))
            for i in range(len(a)):
                c[i] = ( ca * a[i] + c[i-1] * v[i-1] ) / v[i]

       .. math::

                c_i = \frac{a_i c_i}{V_i^T} + \frac{V^T_{i-1} - V_i}{V^T_i}

.. note:: snipped for future
       ::

         rownames = tuple('ABCDEFGH')
         t = []
         for i in range(12):
            for r in rownames:
               t.append((r,i+1))
