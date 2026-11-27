"""Perception presentation helpers reused from the original vision adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Union

def align_mask_visualization_path(
    mask_path: Union[str, Path],
    rgb_path: Union[str, Path],
) -> Path:
    """Rename GSAM mask viz to ``{rgb_stem}_masked{ext}`` (agent-side, before LLM read)."""
    mask_src = Path(mask_path).resolve()
    dest = mask_src.parent / f"{Path(rgb_path).stem}_masked{mask_src.suffix}"
    if mask_src != dest.resolve():
        if dest.exists():
            dest.unlink()
        mask_src.rename(dest)
    return dest
