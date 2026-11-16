"""Deterministic temporal split primitives for the unified AdCL-FD protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalBlock:
    """A half-open interval on the full entity timeline."""

    index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Temporal block index cannot be negative")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Temporal block must be a non-empty interval")

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class UnifiedTemporalSplit:
    """The immutable chronological 50/10/40 split for one entity."""

    train_source_length: int
    full_length: int
    train_end: int
    train_fraction: float
    post_train_blocks: int
    seed: int
    validation_block_index: int
    all_post_train_blocks: tuple[TemporalBlock, ...]

    def __post_init__(self) -> None:
        if self.train_source_length < 1:
            raise ValueError("train_source_length must be positive")
        if self.full_length <= self.train_source_length:
            raise ValueError("full_length must exceed train_source_length")
        if self.train_fraction != 0.5:
            raise ValueError("train_fraction must be exactly 0.5")
        if self.post_train_blocks != 2:
            raise ValueError("post_train_blocks must be exactly 2")
        if len(self.all_post_train_blocks) != self.post_train_blocks:
            raise ValueError("post-train block count does not match metadata")
        if self.validation_block_index != 0:
            raise ValueError("Validation must be the first post-training block")
        expected_train_end = self.full_length // 2
        expected_validation_end = 3 * self.full_length // 5
        if self.train_end != expected_train_end:
            raise ValueError("Training boundary must be floor(N / 2)")
        expected_blocks = (
            TemporalBlock(0, expected_train_end, expected_validation_end),
            TemporalBlock(1, expected_validation_end, self.full_length),
        )
        if self.all_post_train_blocks != expected_blocks:
            raise ValueError("Blocks must follow chronological 50/10/40 boundaries")
        if self.train_end < 1 or self.train_end >= self.full_length:
            raise ValueError("train_end must be inside the full timeline")
        expected_start = self.train_end
        for index, block in enumerate(self.all_post_train_blocks):
            if block.index != index or block.start != expected_start:
                raise ValueError("Post-train blocks must be ordered and contiguous")
            expected_start = block.end
        if expected_start != self.full_length:
            raise ValueError("Post-train blocks must cover the full post-train suffix")

    @property
    def validation(self) -> TemporalBlock:
        return self.all_post_train_blocks[self.validation_block_index]

    @property
    def normal_fit_end(self) -> int:
        """End of the training candidate region inside the original normal source."""
        return min(self.train_end, self.train_source_length)

    @property
    def test_blocks(self) -> tuple[TemporalBlock, ...]:
        return tuple(
            block
            for block in self.all_post_train_blocks
            if block.index != self.validation_block_index
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "fdpg_adcl_chronological_50_10_40_v1",
            "train_source_length": self.train_source_length,
            "full_length": self.full_length,
            "train_end": self.train_end,
            "train_fraction": self.train_fraction,
            "post_train_blocks": self.post_train_blocks,
            "seed": self.seed,
            "validation_block_index": self.validation_block_index,
            "all_post_train_blocks": [
                block.to_dict() for block in self.all_post_train_blocks
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UnifiedTemporalSplit":
        if payload.get("protocol") != "fdpg_adcl_chronological_50_10_40_v1":
            raise ValueError("Unsupported temporal split protocol")
        raw_blocks = payload.get("all_post_train_blocks")
        if not isinstance(raw_blocks, list):
            raise ValueError("Temporal split metadata must contain block mappings")
        blocks = tuple(
            TemporalBlock(
                index=int(item["index"]),
                start=int(item["start"]),
                end=int(item["end"]),
            )
            for item in raw_blocks
        )
        return cls(
            train_source_length=int(payload["train_source_length"]),
            full_length=int(payload["full_length"]),
            train_end=int(payload["train_end"]),
            train_fraction=float(payload["train_fraction"]),
            post_train_blocks=int(payload["post_train_blocks"]),
            seed=int(payload["seed"]),
            validation_block_index=int(payload["validation_block_index"]),
            all_post_train_blocks=blocks,
        )


def split_full_series(
    *,
    train_length: int,
    test_length: int,
    seed: int,
    train_fraction: float = 0.5,
    post_train_blocks: int = 2,
) -> UnifiedTemporalSplit:
    """Create a reproducible split from a chronological train/test pair."""
    if train_length < 1 or test_length < 1:
        raise ValueError("train_length and test_length must be positive")
    if not np.isfinite(train_fraction) or train_fraction != 0.5:
        raise ValueError("train_fraction must be exactly 0.5")
    if post_train_blocks != 2:
        raise ValueError("post_train_blocks must be exactly 2")
    full_length = int(train_length + test_length)
    train_end = full_length // 2
    validation_end = 3 * full_length // 5
    if not 0 < train_end < validation_end < full_length:
        raise ValueError("Sequence is too short for three non-empty chronological partitions")
    # Seed controls model randomness, never the partition or test timestamps.
    blocks = (
        TemporalBlock(index=0, start=train_end, end=validation_end),
        TemporalBlock(index=1, start=validation_end, end=full_length),
    )
    validation_index = 0
    return UnifiedTemporalSplit(
        train_source_length=int(train_length),
        full_length=full_length,
        train_end=train_end,
        train_fraction=float(train_fraction),
        post_train_blocks=int(post_train_blocks),
        seed=int(seed),
        validation_block_index=validation_index,
        all_post_train_blocks=tuple(blocks),
    )
