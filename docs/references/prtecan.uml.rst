.. uml::

   class BufferWellsMixin {
     +@buffer_wells: list[str]
   }

   Labelblock .up.|> BufferWellsMixin : Uses >
   LabelblocksGroup .up.|> BufferWellsMixin : Uses >
   Titration .up.|> BufferWellsMixin : Uses >

   class Labelblock{
     #_lines: list_of_lines
	 #path: Path|None
     +metadata: dict[str, Metadata]
     +data: dict[str, float]
     +@data_norm: dict[str, float]
	 +@buffer_wells: list[str]
     +@data_buffersubtracted: dict[str, float]
     +@data_buffersubtracted_norm: dict[str, float]
	 +@buffer: float
	 +@buffer_sd: float
	 +@buffer_norm: float
	 +@buffer_norm_sd: float
	 __eq__()
     __almost_eq__()
   }

   class LabelblocksGroup{
     #labelblocks: list[Labelblock]
	 #allequal: bool
	 +metadata: dict[str, Metadata]
	 +@data: dict[str, list[float]]|None
     +@data_norm: dict[str, list[float]]
	 +@buffer_wells: list[str]
     +@data_buffersubtracted: dict[str, list[float]]|None|{}
     +@data_buffersubtracted_norm: dict[str, list[float]]|{}
   }

   class Metadata{
     #value: float|int|str
	 #unit: list[float|int|str]
   }

   class Tecanfile{
     #path: Path
	 +metadata: dict[str, Metadata]
	 +labelblocks: list[Labelblock]
   }

   Tecanfile "1..*" o-- Labelblock
   Tecanfile::metadata "*" *-- Metadata
   Labelblock::metadata "*" *-- Metadata


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
     #conc: ArrayF
     #is_ph: bool
	 #fromlistfile()
	 +@additions: list[float]
	 +load_additions(Path)
	 +@data: list[dict[str, list[float]]|None|{}]
	 +@data_nrm: list[dict[str, list[float]]|{}]
	 +@scheme: PlateScheme
	 +load_scheme(Path)
	 +export_data()
   }

   Titration --|> TecanfilesGroup

   class PlateScheme{
     #file: Path|None
	 +@buffer: list[str]|[]
	 +@crtl: list[str]|[]
	 +@names: dict[str, set[str]]|{}
   }

   class TitrationAnalysis{
     +keys_unk: list[str]
	 #fromlistfile()
     +@fitdata: Sequence[dict[str, list[float]]
     +@fitdata_params: dict[str, bool]
     +@fitkws: Kwargs
     +@results: list[dict[str, FitResult]]
     +@result_dfs: list[pd.DataFrame]
	 +load_scheme(Path)
	 +fit()
	 +plot_k()
	 +plot_all_wells()
	 +plot_ebar()
	 +print_fitting()
	 +plot_buffers()
	 +export_data()
   }

   TitrationAnalysis --|> Titration
   TitrationAnalysis "0..1" *-- PlateScheme

..
   left to right direction
