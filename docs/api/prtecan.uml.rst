.. uml::
      
   class Labelblock{
     lines: list_of_lines
	 path: Path, optional
     +metadata : dict
     +data : dict e.g.{'H12':float}
     +@data_normalized : dict
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

   class Tecanfile{
     path : Path
	 +metadata : dict
	 +labelblocks : list
   }

   class Metadata{
     value: float|int|str
	 unit: list[float|int|str]
   }

   Tecanfile "1..*" o-- Labelblock
   Tecanfile::metadata "1..*" *-- Metadata
   Labelblock::metadata "1..*" *-- Metadata

   class LabelblocksGroup{
     labelblocks: list[Labelblock]
	 allequal: bool
	 buffer: None
	 +metadata: dict
	 +data: dict[str, list[float]]
   }
   
   LabelblocksGroup::labelblocks "(ordered)" o-- Labelblock

   LabelblocksGroup "*" --o TecanfilesGroup
   TecanfilesGroup <|-- Titration
   Titration "1" --o TitrationAnalysis

   class TecanfilesGroup{
     tecanfiles: list[Tecanfile]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +metadata: dict
   }

   TecanfilesGroup::tecanfiles "*..1" --o Tecanfile

   class Titration{
     listfile: Path|str
	 +metadata: dict
	 +conc: Sequence[float]
	 +export_dat()
   }

   class TitrationAnalysis{
     titration: Titration
	 schemefile: str|Path
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


..
   left to right direction
