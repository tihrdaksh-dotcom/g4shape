"""Samplers whose order does not depend on the number of DDP ranks (stop / resume support).

``ResumableBatchSampler``: one global shuffle per epoch (``seed + epoch``), cut into global
batches of ``global_batch`` samples; each rank takes its contiguous share of every global
batch. Global batch ``b`` holds the same samples whatever the number of GPUs, so a run
stopped after ``b`` batches can resume on a different number of GPUs from batch ``b``
(``start=(epoch, b)``) and still see every sample of the epoch exactly once.

``ShardSampler``: an unshuffled split of the dataset into one contiguous shard per rank
(no padding; ``LitG4Model`` gathers uneven shards). For val / test under DDP.

Rank and world size are read from ``torch.distributed`` when iterating (Lightning starts
the process group before it touches the loaders), so both work unchanged on one device.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ["ResumableBatchSampler", "ShardSampler", "rank_world"]


def rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class ResumableBatchSampler(Sampler[list[int]]):
    """Batches of this rank's dataset indices (pass as ``DataLoader(batch_sampler=...)``).

    ``drop_last``: drop the last incomplete global batch (only when the dataset has more
    than one global batch). ``start=(epoch, batches_done)``: in that epoch, skip the first
    ``batches_done`` global batches. ``len()`` is always the full epoch: Lightning counts
    the skipped batches as done when it resumes mid-epoch.
    """

    def __init__(self, num_samples: int, global_batch: int, seed: int = 0, drop_last: bool = True,
                 start: Optional[tuple[int, int]] = None):
        if num_samples <= 0 or global_batch <= 0:
            raise ValueError("num_samples and global_batch must be positive")
        self.num_samples, self.global_batch, self.seed = num_samples, global_batch, seed
        self.drop_last = drop_last and num_samples > global_batch
        self.start = start
        self.epoch = 0
        # Lightning calls set_epoch on ``loader.batch_sampler.sampler``, not on the batch sampler
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:  # called by Lightning at every epoch start
        self.epoch = epoch

    def __len__(self) -> int:
        n, b = self.num_samples, self.global_batch
        return n // b if self.drop_last else -(-n // b)

    def skipped(self) -> int:
        if self.start is None or self.start[0] != self.epoch:
            return 0
        return min(self.start[1], len(self))

    def __iter__(self) -> Iterator[list[int]]:
        rank, world = rank_world()
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(self.num_samples)
        for b in range(self.skipped(), len(self)):
            share = np.array_split(order[b * self.global_batch:(b + 1) * self.global_batch], world)[rank]
            if share.size == 0:
                raise ValueError(f"global batch {b} has fewer samples than the {world} ranks")
            yield share.tolist()


class ShardSampler(Sampler[int]):
    """Indices of this rank's contiguous shard, in order (all indices on one device)."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def _shard(self) -> range:
        rank, world = rank_world()
        lo, hi = (self.num_samples * rank) // world, (self.num_samples * (rank + 1)) // world
        return range(lo, hi)

    def __len__(self) -> int:
        return len(self._shard())

    def __iter__(self) -> Iterator[int]:
        return iter(self._shard())
