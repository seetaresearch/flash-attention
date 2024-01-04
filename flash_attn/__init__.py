# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Flash Attention for Dragon."""

import os as _os
import platform as _platform

from dragon.core.framework.backend import load_library as _load_library

# Functions
from flash_attn_dragon.flash_attn_interface import flash_attn_func
from flash_attn_dragon.flash_attn_interface import flash_attn_kvpacked_func
from flash_attn_dragon.flash_attn_interface import flash_attn_qkvpacked_func
from flash_attn_dragon.flash_attn_interface import flash_attn_with_kvcache

# Version
from flash_attn_dragon.version import version as __version__

# Libraries
_load_library(
    _os.path.join(
        _os.path.abspath(_os.path.dirname(__file__)),
        "lib",
        "{}dragon_flashattn.{}".format(
            "" if _platform.system() == "Windows" else "lib",
            "dll" if _platform.system() == "Windows" else "so",
        ),
    )
)

# Attributes
__all__ = [_s for _s in dir() if not _s.startswith("_")]
