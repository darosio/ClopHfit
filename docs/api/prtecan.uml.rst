.. uml::

   class Labelblock{
     #lines: list_of_lines
	 #path: Path, optional
     +metadata : dict
     +data : dict e.g.{'H12':float}
     +@data_normalized : dict
	 +@buffer_wells: list[str]
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
     #path : Path
	 +metadata : dict
	 +labelblocks : list
   }

   class Metadata{
     #value: float|int|str
	 #unit: list[float|int|str]
   }

   Tecanfile "1..*" o-- Labelblock
   Tecanfile::metadata "1..*" *-- Metadata
   Labelblock::metadata "1..*" *-- Metadata

   class LabelblocksGroup{
     #labelblocks: list[Labelblock]
	 #allequal: bool
	 +metadata: dict
	 +data: dict[str, list[float]]
     +@data_normalized : dict
	 +@buffer_wells: list[str]
     +@data_buffersubtracted : dict
     +@data_buffersubtracted_norm : dict
   }

   LabelblocksGroup::labelblocks "(ordered)" o-- Labelblock
   LabelblocksGroup::buffer_wells "0..1" <--> Labelblock::buffer_wells : (same)

   class TecanfilesGroup{
     #tecanfiles: list[Tecanfile]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +metadata: dict
   }

   TecanfilesGroup "*" o-- LabelblocksGroup
   TecanfilesGroup::tecanfiles "1..*" o-- Tecanfile

   class Titration{
     #tecanfiles: list[Tecanfile]
     #conc: Sequence
	 #fromlistfile(Path)
	 +load_additions(Path)
	 +@additions: list
	 +@buffer_wells: list
	 +@data_dilutioncorrected: list[dict]
	 +@data_dilutioncorrected_norm: list[dict]
	 +export_dat()
   }

   Titration --|> TecanfilesGroup

   class TitrationAnalysis{
     #titration: Titration
	 #schemefile: str|Path
	 +scheme: pd.Series[Any]
	 +conc: Sequence[float]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +additions: Sequence[float]
	 _get_heys()
	 +fit()
	 +plot_k()
	 +plot_well()
	 +plot_all_wells()
	 +plot_ebar()
	 +print_fitting()
	 +plot_buffers()
   }

   TitrationAnalysis "1" o-- Titration

..
   left to right direction
