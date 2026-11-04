#!/usr/bin/env python3
"""
datasets/__init__.py
Central registry — maps dataset name strings to adapter classes.
To add a new dataset: implement DatasetAdapter in a new file,
then add one line to REGISTRY below.
"""

from .base             import DatasetAdapter
from .ckplus           import CKPlusAdapter
from .oulu             import OuluAdapter
from .oulu_acropped    import OuluAcroppedAdapter
from .afew             import AFEWAdapter
from .enterface05      import ENTERFACE05Adapter
from .caer             import CAERAdapter
from .afew_caer_joint  import AFEWCAERJointAdapter
from .dfew             import DFEWAdapter
from .ferv39k          import FERV39kAdapter

REGISTRY = {
    'CK+':           CKPlusAdapter,
    'Oulu':          OuluAdapter,
    'Oulu_Acropped': OuluAcroppedAdapter,
    'AFEW':          AFEWAdapter,
    'eNTERFACE05':   ENTERFACE05Adapter,
    'CAER':          CAERAdapter,
    'AFEW+CAER':     AFEWCAERJointAdapter,
    'DFEW':          DFEWAdapter,
    'FERV39k':       FERV39kAdapter,
}


def get_adapter(dataset_name: str):
    """
    Returns the adapter class (not instance) for the given dataset name.

    Usage:
        adapter = get_adapter('AFEW+CAER')(args)
        samples = adapter.get_all_samples()
    """
    if dataset_name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[dataset_name]