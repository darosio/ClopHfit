.. uml::

   class Metadata{
     #value: float|int|str
	 #unit: list[float|int|str]
   }

   class Labelblock{
     #_lines: list_of_lines
	 +filename: str
     +metadata: dict[str, Metadata]
     +data: dict[str, float]
     +data_norm: dict[str, float]
	 __eq__()
     __almost_eq__()
   }

   class Tecanfile{
     #path: Path
	 +metadata: dict[str, Metadata]
	 +labelblocks: list[Labelblock]
   }

   class LabelblocksGroup{
     #labelblocks: list[Labelblock]
	 #allequal: bool
	 +metadata: dict[str, Metadata]
	 +@data: dict[str, list[float]]
     +@data_norm: dict[str, list[float]]
   }


   class TecanfilesGroup{
     #tecanfiles: list[Tecanfile]
	 +labelblocksgroups: list[LabelblocksGroup]
	 +metadata: dict[str, Metadata]
     +n_labels: int
   }


   class PlateScheme{
     #file: Path|None
	 +@buffer: list[str]|[]
	 +@crtl: list[str]|[]
	 +@names: dict[str, set[str]]|{}
   }

   class TitrationConfig{
     #bg: bool
     #bg_adj: bool
     #dil: bool
     #nrm: bool
     #bg_mth: str
   }

   class FitResult{
     #figure: Figure
     #result: MinimizerResult
     #mini: Minimizer
   }

   class TecanConfig{
     #out_fp: Path
     #verbose: int
     #comb: bool
     #lim: tuple[]|None
     #sel: tuple[]|None
     #title: str
     #fit: bool
     #png: bool
     #pdf: bool
   }

   class Buffer{
     #tit: Titration
     +@wells: list[str]
     +@dataframes: list[DataFrame]
     +@dataframes_nrm: list[DataFrame]
     +@bg: list[ArrayF]
     +@bg_sd: list[ArrayF]
     +fit_results: list[BufferFit]
     +fit_results_nrm: list[BufferFit]
	 +plot()
   }

   class Titration{
     #conc: ArrayF
     #is_ph: bool
     +buffer: Buffer
     +@params: TitrationConfig
	 +@additions: list[float]
     +@fit_keys: list[str]
     +@bg: list[ArrayF]
     +@bg_sd: list[ArrayF]
	 +@data: list[dict[str, ArrayF]]
	 +@scheme: PlateScheme
     +keys_unk: list[str]
     +@results: list[dict[str, FitResult]]
     +@result_dfs: list[pd.DataFrame]

	 +update_fit_keys(list[str])
	 #fromlistfile(Path|str, bool)
     +load_additions(Path)
	 +load_scheme(Path)
   	 +export_data_fit(TecanConfig)
	 +fit()
	 +print_fitting(int)
	 +plot_temperature()
	 +export_png(int, Path|str)
   }

   class TitrationPlotter{
     #tit: Titration
	 +plot_k(int, str)
	 +plot_all_wells(int, Path|str)
	 +plot_ebar(int, str, str)
   }


   Labelblock  "1..*" --*  Tecanfile
   Labelblock  "1..*" --o  LabelblocksGroup::labelblocks : ordered
   Tecanfile  "1..*" --o  TecanfilesGroup::tecanfiles : ordered

   LabelblocksGroup  "1..*" --*  TecanfilesGroup::labelblocksgroups

   TecanfilesGroup  <|--  Titration

   Titration::buffer  *--*  Buffer::tit : interdependent
   Titration::scheme  *-- "0..1"  PlateScheme
   Titration::results  *-- "*"  FitResult
   Titration::params  -  TitrationConfig : data processing <
      Titration::export_data_fit  -  TecanConfig : cli params <

   TitrationPlotter::tit  o--  Titration

..
   left to right direction
   Metadata  "*" --*  Tecanfile::metadata
   Metadata  "*" --*  Labelblock::metadata
   Metadata  "*" --*  LabelblocksGroup::metadata
   Metadata  "*" --*  TecanfilesGroup::metadata
