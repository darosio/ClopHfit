"""ClopHfit: Parse plate-reader and fit ClopHensor titrations."""
from pathlib import Path

from pkg_resources import get_distribution  # type: ignore

__version__ = get_distribution("clophfit").version
__default_enspire_out_dir__ = Path(f"Meas-{__version__}")
