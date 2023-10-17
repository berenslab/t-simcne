from pathlib import Path

from .tsimcne import PLtSimCNE, TSimCNE

__file_path = Path(__file__).resolve()

__version__ = (__file_path.parent.parent / "VERSION").read_text().strip()
