"""ClopHfit: Parse plate-reader and fit ClopHensor titrations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("clophfit")
except PackageNotFoundError:
    __version__ = "unknown"

__enspire_out_dir__ = f"Meas-{__version__}"
__tecan_out_dir__ = f"out-{__version__}"
