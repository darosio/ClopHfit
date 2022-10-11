.. uml::

   left to right direction

   class Labelblock{
     lines: list_of_lines
     +metadata : dict
     +data : dict {'H12':float}
     +@data_normalized : dict {'H12':float}
	 +@buffer_wells: list
     +@data_buffersubtracted : dict
     +@data_buffersubtracted_norm : dict
	 +@buffer: float
	 +@sd_buffer: float
	 +@buffer_norm: float
	 +@sd_buffer_norm: float
	 __eq__()
     __almost_eq__()
   }

   Labelblock "1..*" --o Tecanfile

   class Tecanfile{
     path : Path
	 +metadata : dict
	 +labelblocks : list
	 {static} +read_xls()
	 {static} +lookup_csv_lines()
   }

   LabelblocksGroup ..> Labelblock
   TecanfilesGroup ..> Tecanfile

   LabelblocksGroup "*" --o TecanfilesGroup
   TecanfilesGroup <|-- Titration
   Titration "1" --o TitrationAnalysis

   class LabelblocksGroup{
     labelblocks: list[Labelblock]
	 allequal: bool
	 buffer: None
	 +metadata: dict
	 +data: dict[str, list[float]]
   }

   class TecanfilesGroup{
     tecanfiles: list[Tecanfile]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +metadata: dict
   }

   class Titration{
     listfile: Path | str
	 +metadata: dict
	 +conc: Sequence[float]
	 +export_dat()
   }

   class TitrationAnalysis{
     titration: Titration
	 schemefile: str | Path
	 +scheme: pd.Series[Any]
	 +conc: Sequence[float]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +additions: Sequence[float]
	 {static} +calculate_conc()
	 +subtract_bg()
	 +dilution_correction()
	 +metadata_normalization()
	 +fit()
	 +plot_k()
	 +plot_well()
	 +plot_all_wells()
	 +plot_ebar()
	 +print_fitting()
	 +plot_buffers()
   }
