"""Offline initialization for a fixed next-token draft head.

The utilities in this module are intentionally independent from the Ray PPO
loop. They can be run before RL training with the current HuggingFace-style
policy model, then the returned ``DraftNextTokenHead`` can be kept frozen and
used to propose candidate tokens during rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class OfflineDraftTrainingConfig:
    """Configuration for offline draft-head initialization."""

    max_new_tokens: int = 64
    num_rollouts_per_prompt: int = 1
    prompt_batch_size: int = 4
    draft_batch_size: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    train_on_prompt_tokens: bool = False
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    hidden_multiplier: int = 2
    dropout: float = 0.0


@dataclass
class DraftCandidate:
    """Candidate tokens returned by ``DraftNextTokenHead.propose_next_token``."""

    token_ids: torch.Tensor
    log_probs: torch.Tensor
    logits: torch.Tensor


class DraftNextTokenHead(nn.Module):
    """Small next-token head trained from frozen-policy rollout states.

    The module does not own or register the policy model, so passing
    ``draft_model.parameters()`` to an optimizer cannot accidentally update the
    policy. During rollout, call ``propose_next_token(policy_model, ...)`` with
    the current policy to get candidate tokens. The method uses ``no_grad`` for
    both policy and draft computations.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 2

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, vocab_size),
        )
        self.offline_init_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return next-token logits from policy hidden states."""
        hidden_states = hidden_states.to(dtype=_infer_module_dtype(self))
        return self.net(hidden_states)

    def freeze(self) -> "DraftNextTokenHead":
        """Freeze draft weights for use inside the RL rollout loop."""
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> "DraftNextTokenHead":
        """Enable draft-head training."""
        self.train()
        for param in self.parameters():
            param.requires_grad_(True)
        return self

    def loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy next-token loss.

        Args:
            hidden_states: Policy hidden states aligned with ``labels``.
            labels: Target next-token ids.
            loss_mask: Optional boolean/0-1 mask for tokens to train on.
        """
        logits = self(hidden_states)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)

        if loss_mask is None:
            return F.cross_entropy(flat_logits, flat_labels)

        flat_mask = loss_mask.reshape(-1).bool()
        if not torch.any(flat_mask):
            return flat_logits.sum() * 0.0
        return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])

    @torch.no_grad()
    def propose_next_token(
        self,
        policy_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_candidates: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> DraftCandidate:
        """Generate candidate next tokens for a rollout loop.

        The policy forward pass is used only to produce hidden states; gradients
        are disabled so the draft path does not affect policy-gradient updates.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")

        was_training = policy_model.training
        policy_model.eval()
        self.eval()
        try:
            outputs = _policy_forward_hidden_states(policy_model, input_ids, attention_mask)
            hidden_states = outputs.hidden_states[-1]
            last_hidden = _gather_last_valid_hidden(hidden_states, attention_mask)
            logits = self(last_hidden) / temperature
            log_probs = F.log_softmax(logits, dim=-1)

            if do_sample:
                probs = log_probs.exp()
                token_ids = torch.multinomial(probs, num_samples=min(num_candidates, logits.size(-1)))
                candidate_log_probs = log_probs.gather(dim=-1, index=token_ids)
            else:
                candidate_log_probs, token_ids = torch.topk(log_probs, k=min(num_candidates, logits.size(-1)), dim=-1)

            return DraftCandidate(token_ids=token_ids, log_probs=candidate_log_probs, logits=logits)
        finally:
            if was_training:
                policy_model.train()


class _RolloutTokenDataset(Dataset):
    def __init__(self, items: list[dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def train_offline_draft_model(
    policy_model: nn.Module,
    dataset_of_prompts: Iterable[Any],
    tokenizer: Optional[Any] = None,
    config: Optional[OfflineDraftTrainingConfig] = None,
    draft_model: Optional[DraftNextTokenHead] = None,
    device: Optional[torch.device | str] = None,
) -> DraftNextTokenHead:
    """Train and return a frozen next-token draft head.

    Args:
        policy_model: Current policy used to generate rollouts and hidden states.
        dataset_of_prompts: Prompt strings, tensors, or dicts with ``input_ids``.
        tokenizer: Required when prompts are raw strings.
        config: Offline rollout and draft-head training settings.
        draft_model: Optional existing head to initialize further.
        device: Device for draft training. Defaults to the policy parameter device.

    Returns:
        A trained ``DraftNextTokenHead`` with all draft parameters frozen.
    """
    if config is None:
        config = OfflineDraftTrainingConfig()
    if device is None:
        device = _infer_module_device(policy_model)
    device = torch.device(device)

    hidden_size = int(getattr(policy_model.config, "hidden_size"))
    vocab_size = int(getattr(policy_model.config, "vocab_size"))
    if draft_model is None:
        draft_model = DraftNextTokenHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            intermediate_size=hidden_size * config.hidden_multiplier,
            dropout=config.dropout,
        )
    draft_model.to(device)
    draft_model.unfreeze()

    previous_requires_grad = _set_requires_grad(policy_model, requires_grad=False)
    policy_was_training = policy_model.training
    policy_model.eval()

    try:
        rollout_dataset = _collect_policy_rollouts(
            policy_model=policy_model,
            dataset_of_prompts=dataset_of_prompts,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            [param for param in draft_model.parameters() if param.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_loader = DataLoader(
            rollout_dataset,
            batch_size=config.draft_batch_size,
            shuffle=True,
            collate_fn=lambda batch: _collate_rollout_items(batch, config.pad_token_id),
        )

        total_loss = 0.0
        total_steps = 0
        for _ in range(config.num_train_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)

                with torch.no_grad():
                    outputs = _policy_forward_hidden_states(policy_model, input_ids, attention_mask)
                    features = outputs.hidden_states[-1][:, :-1, :].detach()

                labels = input_ids[:, 1:]
                loss_mask = loss_mask[:, 1:]
                loss = draft_model.loss(features, labels, loss_mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(draft_model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_steps += 1

        draft_model.offline_init_metrics = {
            "num_rollout_sequences": float(len(rollout_dataset)),
            "train_steps": float(total_steps),
            "mean_loss": total_loss / max(total_steps, 1),
        }
        return draft_model.freeze()
    finally:
        _restore_requires_grad(policy_model, previous_requires_grad)
        if policy_was_training:
            policy_model.train()


@torch.no_grad()
def _collect_policy_rollouts(
    policy_model: nn.Module,
    dataset_of_prompts: Iterable[Any],
    tokenizer: Optional[Any],
    config: OfflineDraftTrainingConfig,
    device: torch.device,
) -> _RolloutTokenDataset:
    if not isinstance(dataset_of_prompts, Dataset):
        dataset_of_prompts = list(dataset_of_prompts)

    prompt_loader = DataLoader(
        dataset_of_prompts,
        batch_size=config.prompt_batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_prompts(batch, tokenizer, config.pad_token_id),
    )
    items: list[dict[str, torch.Tensor]] = []

    pad_token_id = _resolve_pad_token_id(config, tokenizer)
    eos_token_id = _resolve_eos_token_id(config, tokenizer)

    for prompt_batch in prompt_loader:
        prompt_ids = prompt_batch["input_ids"].to(device)
        prompt_attention_mask = prompt_batch["attention_mask"].to(device)
        prompt_width = prompt_ids.size(1)
        num_return_sequences = config.num_rollouts_per_prompt
        if not config.do_sample and config.num_rollouts_per_prompt > 1:
            prompt_ids = prompt_ids.repeat_interleave(config.num_rollouts_per_prompt, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(config.num_rollouts_per_prompt, dim=0)
            num_return_sequences = 1

        generate_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        if config.do_sample:
            generate_kwargs.update(
                {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                }
            )

        generated = policy_model.generate(**generate_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        if sequences.ndim != 2:
            raise ValueError(f"expected generated sequences to be 2-D, got {sequences.shape}")

        attention_mask = sequences.ne(pad_token_id).to(dtype=torch.long)
        target_positions = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0)
        if config.train_on_prompt_tokens:
            loss_mask = attention_mask.bool()
        else:
            loss_mask = attention_mask.bool() & (target_positions >= prompt_width)

        for input_ids, seq_mask, token_loss_mask in zip(sequences, attention_mask, loss_mask, strict=True):
            items.append(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "attention_mask": seq_mask.detach().cpu(),
                    "loss_mask": token_loss_mask.detach().cpu(),
                }
            )

    if not items:
        raise ValueError("dataset_of_prompts produced no rollouts")
    return _RolloutTokenDataset(items)


def _policy_forward_hidden_states(
    policy_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Any:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if attention_mask is None:
        kwargs.pop("attention_mask")
    try:
        outputs = policy_model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        outputs = policy_model(**kwargs)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("policy_model must return hidden_states when output_hidden_states=True")
    return outputs


def _collate_prompts(batch: list[Any], tokenizer: Optional[Any], pad_token_id: Optional[int]) -> dict[str, torch.Tensor]:
    first = batch[0]
    if isinstance(first, str):
        if tokenizer is None:
            raise ValueError("tokenizer is required when dataset_of_prompts contains strings")
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    if isinstance(first, dict):
        input_ids = [_as_1d_long_tensor(item["input_ids"]) for item in batch]
        if "attention_mask" in first:
            attention_mask = [_as_1d_long_tensor(item["attention_mask"]) for item in batch]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]
        return _pad_token_fields(input_ids, attention_mask, pad_token_id)

    input_ids = [_as_1d_long_tensor(item) for item in batch]
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    return _pad_token_fields(input_ids, attention_mask, pad_token_id)


def _collate_rollout_items(batch: list[dict[str, torch.Tensor]], pad_token_id: Optional[int]) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    loss_mask = [item["loss_mask"] for item in batch]
    padded = _pad_token_fields(input_ids, attention_mask, pad_token_id)
    padded["loss_mask"] = _pad_1d_tensors(loss_mask, pad_value=0).bool()
    return padded


def _pad_token_fields(
    input_ids: list[torch.Tensor],
    attention_mask: list[torch.Tensor],
    pad_token_id: Optional[int],
) -> dict[str, torch.Tensor]:
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    return {
        "input_ids": _pad_1d_tensors(input_ids, pad_value=pad_id),
        "attention_mask": _pad_1d_tensors(attention_mask, pad_value=0),
    }


def _pad_1d_tensors(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(t.numel() for t in tensors)
    out = tensors[0].new_full((len(tensors), max_len), fill_value=pad_value)
    for idx, tensor in enumerate(tensors):
        out[idx, : tensor.numel()] = tensor
    return out


def _as_1d_long_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    tensor = tensor.detach().clone().long()
    if tensor.ndim != 1:
        raise ValueError(f"expected 1-D token tensor, got {tensor.shape}")
    return tensor


def _gather_last_valid_hidden(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1, :]
    positions = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
    masked_positions = positions.masked_fill(~attention_mask.bool(), 0)
    last_indices = masked_positions.max(dim=-1).values
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


def _infer_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _resolve_pad_token_id(config: OfflineDraftTrainingConfig, tokenizer: Optional[Any]) -> int:
    if config.pad_token_id is not None:
        return int(config.pad_token_id)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _resolve_eos_token_id(config: OfflineDraftTrainingConfig, tokenizer: Optional[Any]) -> Optional[int]:
    if config.eos_token_id is not None:
        return int(config.eos_token_id)
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return None


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> list[bool]:
    previous = []
    for param in module.parameters():
        previous.append(param.requires_grad)
        param.requires_grad_(requires_grad)
    return previous


def _restore_requires_grad(module: nn.Module, previous: list[bool]) -> None:
    for param, requires_grad in zip(module.parameters(), previous, strict=True):
        param.requires_grad_(requires_grad)


__all__ = [
    "DraftCandidate",
    "DraftNextTokenHead",
    "OfflineDraftTrainingConfig",
    "train_offline_draft_model",
]
