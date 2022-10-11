.. uml::

   left to right direction

   class Labelblock{
     lines: list_of_lines
     +metadata : dict
     +data : dict {'H12':float}
     +data_normalized : dict {'H12':float}
	 +buffer_wells: list
     +data_buffersubtracted : dict
     +data_buffersubtracted_norm : dict
	 +buffer: float
	 +sd_buffer: float
	 +buffer_norm: float
	 +sd_buffer_norm: float
	 __eq__()
     __almost_eq__()
   }

   Labelblock "1..*" --o Tecanfile

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
