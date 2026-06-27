import sys

from .model import ADAPEL
from .model import base, nuisance, stacking, bootstrap, diagnostics, clinical, config

# Make submodules accessible as adapel.<module>
sys.modules['adapel.base'] = base
sys.modules['adapel.nuisance'] = nuisance
sys.modules['adapel.stacking'] = stacking
sys.modules['adapel.bootstrap'] = bootstrap
sys.modules['adapel.diagnostics'] = diagnostics
sys.modules['adapel.clinical'] = clinical
sys.modules['adapel.config'] = config

__all__ = ["ADAPEL"]
