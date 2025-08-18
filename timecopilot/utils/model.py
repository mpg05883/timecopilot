import re
import warnings
from pathlib import Path

from tqdm import tqdm

from .common import timestamp_info


def find_best_checkpoint(
    dirpath: str | Path,
    verbose: bool = True,
) -> Path | None:
    """
    Finds the checkpoint with the lowest absolue value of validation loss in a
    given directory.

    Args:
        dirpath (str | Path): The full directory path where the desired
            checkpoints are stored. Checkpoints must contain 'val_loss' in
            their name, e.g., 'val_loss=0.123.ckpt'. to be considered valid.
        verbose (bool, optional): Whether to display a progress bar while
            searching for the checkpoint and print the found checkpoint.
            Defaults to True.

    Returns:
        Path | None: A Path object pointing to the checkpoint with the lowest
            absolute validation loss, or None if no valid checkpoints are
            found.
    """
    dirpath = Path(dirpath)

    if not (checkpoints := list(dirpath.glob("*.ckpt"))):
        # Initialize message outside of warnings.warn so the f-string is
        # properly formatted
        message = f"No checkpoints found in {(dirpath)}!"
        warnings.warn(message, stacklevel=1)
        return None

    # Search for checkpoints with positive or negative validation losses
    pattern = re.compile(r"val_loss=([-+]?[0-9]*\.?[0-9]+)")
    best_checkpoint, lowest_val_loss = None, float("inf")

    kwargs = {
        "desc": "Comparing val losses",
        "unit": "ckpt",
        "total": len(checkpoints),
        "disable": not verbose,
    }

    for checkpoint in tqdm(checkpoints, **kwargs):
        if not (match := pattern.search(checkpoint.name)):
            continue
        if abs(val_loss := float(match.group(1))) >= lowest_val_loss:
            continue
        best_checkpoint, lowest_val_loss = checkpoint, abs(val_loss)

    if best_checkpoint is None:
        message = (
            f"No checkpoints found in {dirpath}! Ensure checkpoints include "
            f"'val_loss' in their name."
        )
        warnings.warn(message, stacklevel=1)
        return None

    if verbose:
        timestamp_info(f"Best checkpoint: {best_checkpoint.name}")
        timestamp_info(f"Lowest val loss: {lowest_val_loss:4f}")

    return best_checkpoint
