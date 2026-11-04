#!/usr/bin/env python3
"""
datasets/oulu_acropped.py
Oulu-CASIA VIS-strong-illumination subset, but using the *pre-cropped*
VL_Acropped frames. Identical splits and labels as OuluAdapter — only the
frames source differs (and we use --already_cropped during landmark/patch
extraction).
"""
from .oulu import OuluAdapter


class OuluAcroppedAdapter(OuluAdapter):
    dataset_name = 'Oulu_Acropped'