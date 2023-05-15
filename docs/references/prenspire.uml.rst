.. uml::

   class EnspireFile {
    #file: Path
    #verbose: int=0
    +metadata: dict
    +measurements: {}
    +wells: []
    _ini
    _fin
    _data_list: list[list[str]]
    _metadata_post: list[list[str]]
    _well_list_platemap: list
    _platemap:
    _filename: str
    extract_measurements()
    export_measurements()
   }


   class ExpNote {
       #note_file: Path
       #verbose: int = 0
       +wells: list
       +titrations: list[Titration]
       +check_wells()
       +build_titrations()
   }

   class Titration {
    #conc: Sequence[float]
    #data: dict[str, pd.DataFrame]
    #cl: str | None = None,
    #ph: str | None = None,
    +plot()
    +fit?()
   }

   Titration::conc *-- Titration::data : dataframe index
   Titration "*" --* "1" ExpNote::titrations  : > build_titrations()
   EnspireFile "1" *-- "1" measurements : > extract_measurements()
   measurement "*" -* "1" measurements

   class measurements << (D,orchid) >> {
       "A": measurement
       "B": measurement
       ⋮
   }

   class measurement << (D,orchid) >> {
       "metadata": {}
       "lambda": []
       "A01": [y]
       .
       "H12": [y]
   }
