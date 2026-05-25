import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

# Fields saved to disk per batch. position_ids reconstructable from attention_mask
# but stored here to avoid any ambiguity in compute_log_prob.
_TENSOR_FIELDS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
    "token_level_scores",
]
_DRAFT_LOG_PROB_FIELD = "draft_log_probs"

_INDEX_FILE = "meta.json"


class ReplayBuffer:
    """Disk-backed experience replay buffer with LRU hot cache.

    Each entry stores one complete training batch (after rollout + reward).
    old_log_probs are stored as draft_log_probs for optional speculative-style
    verification, but training still recomputes current-policy old_log_probs.

    Args:
        cache_dir: Directory for .npz files and index.
        max_size: Maximum number of batches to keep on disk. Oldest evicted first.
        p_fresh: Probability of doing a fresh rollout instead of replaying.
                 1.0 = always fresh (buffer disabled in practice).
        hot_cache_size: Number of recent batches to keep in CPU memory (LRU).
    """

    def __init__(
        self,
        cache_dir: str,
        max_size: int = 50,
        p_fresh: float = 0.5,
        hot_cache_size: int = 5,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.p_fresh = p_fresh
        self.hot_cache_size = hot_cache_size

        self._index_path = self.cache_dir / _INDEX_FILE
        self._index: list[dict] = self._load_index()
        # OrderedDict: batch_id → dict[str, Tensor | np.ndarray] (CPU)
        self._hot: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_replay(self) -> bool:
        """Return True if this step should skip rollout and use cached data."""
        return len(self._index) > 0 and random.random() >= self.p_fresh

    def put(self, batch: DataProto, step: int) -> None:
        """Save a batch to disk after fresh rollout + reward computation."""
        batch_id = f"b{step:07d}"
        file_path = self.cache_dir / f"{batch_id}.npz"

        arrays: dict[str, np.ndarray] = {}
        for key in _TENSOR_FIELDS:
            if key in batch.batch.keys():
                arrays[key] = batch.batch[key].cpu().numpy()

        if "old_log_probs" in batch.batch.keys():
            arrays[_DRAFT_LOG_PROB_FIELD] = batch.batch["old_log_probs"].cpu().numpy()

        uid = batch.non_tensor_batch.get("uid")
        if uid is not None:
            arrays["uid"] = np.array(uid, dtype=object)

        np.savez_compressed(str(file_path), **arrays)

        self._index.append(
            {
                "batch_id": batch_id,
                "file": str(file_path),
                "step_generated": step,
                "use_count": 0,
                "last_used_step": -1,
            }
        )

        # Evict oldest entry when over capacity
        if len(self._index) > self.max_size:
            evicted = self._index.pop(0)
            evicted_path = Path(evicted["file"])
            if evicted_path.exists():
                evicted_path.unlink()
            self._hot.pop(evicted["batch_id"], None)

        self._save_index()

    def sample(self, step: int) -> DataProto:
        """Sample a random cached batch and return it as DataProto.

        draft_log_probs may be included for verification, but caller must still
        use current-policy old_log_probs for training.
        """
        entry = random.choice(self._index)
        batch_id = entry["batch_id"]

        if batch_id in self._hot:
            self._hot.move_to_end(batch_id)
            tensors = self._hot[batch_id]
        else:
            tensors = self._load_npz(entry["file"])
            if len(self._hot) >= self.hot_cache_size:
                self._hot.popitem(last=False)
            self._hot[batch_id] = tensors

        entry["use_count"] += 1
        entry["last_used_step"] = step
        self._save_index()

        return self._to_dataproto(tensors)

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f).get("batches", [])
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump({"batches": self._index}, f, indent=2)

    def _load_npz(self, file_path: str) -> dict:
        data = np.load(file_path, allow_pickle=True)
        tensors: dict = {}
        for key in _TENSOR_FIELDS:
            if key in data:
                tensors[key] = torch.from_numpy(data[key])
        if _DRAFT_LOG_PROB_FIELD in data:
            tensors[_DRAFT_LOG_PROB_FIELD] = torch.from_numpy(data[_DRAFT_LOG_PROB_FIELD])
        if "uid" in data:
            tensors["uid"] = data["uid"]
        return tensors

    def _to_dataproto(self, tensors: dict) -> DataProto:
        batch_size = next(
            v.shape[0] for k, v in tensors.items() if isinstance(v, torch.Tensor)
        )
        td = TensorDict(
            {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)},
            batch_size=[batch_size],
        )
        uid = tensors.get("uid")
        non_tensor = {"uid": uid} if uid is not None else {}
        return DataProto(batch=td, non_tensor_batch=non_tensor)
