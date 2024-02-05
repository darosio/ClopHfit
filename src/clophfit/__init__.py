"""ClopHfit: Parse plate-reader and fit ClopHensor titrations."""

from pkg_resources import get_distribution  # type: ignore

__version__ = get_distribution("clophfit").version
__enspire_out_dir__ = f"Meas-{__version__}"
__tecan_out_dir__ = f"out-{__version__}"
