"""ClopHfit: Parse plate-reader and fit ClopHensor titrations."""

import logging

from clophfit._config import (
    __enspire_out_dir__,
    __tecan_out_dir__,
    __version__,
    configure_logging,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())  # noqa: RUF067

__all__ = [
    "__enspire_out_dir__",
    "__tecan_out_dir__",
    "__version__",
    "configure_logging",
]
