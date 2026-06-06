import copy
import hashlib
import logging
import math
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


logger = logging.getLogger(__file__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_bool(config: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(config, key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def stable_prompt_key(prompt_token_ids: list[int], multi_modal_data: Optional[Any] = None) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(prompt_token_ids, dtype=np.int64).tobytes())
    if multi_modal_data is not None:
        try:
            h.update(pickle.dumps(multi_modal_data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            h.update(repr(multi_modal_data).encode("utf-8", errors="replace"))
    return h.hexdigest()


@dataclass
class TrajectoryNode:
    token_id: Optional[int] = None
    parent_id: Optional[int] = None
    children: dict[int, int] = field(default_factory=dict)
    visit_count: int = 0
    reward_sum: float = 0.0
    reward_max: float = float("-inf")
    avg_old_logprob: float = 0.0
    avg_nll: float = 0.0
    logq_tree_token: float = 0.0
    nllq_tree_token: float = 0.0
    policy_version_of_stats: int = 0

    @property
    def reward_mean(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.reward_sum / float(self.visit_count)


@dataclass
class DraftProposal:
    token_id: int
    child_node_id: int
    logq_tree_token: float

    @property
    def q_prob(self) -> float:
        return math.exp(self.logq_tree_token)


@dataclass
class PolicyKVCacheEntry:
    policy_version: int
    prompt_key: str
    value: Any


class PolicyVersionCache:
    def __init__(self):
        self.policy_version = 0
        self._entries: dict[str, PolicyKVCacheEntry] = {}

    def bump_policy_version(self) -> None:
        self.policy_version += 1
        self._entries.clear()

    def get(self, key: str, prompt_key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.policy_version != self.policy_version or entry.prompt_key != prompt_key:
            return None
        return entry.value

    def put(self, key: str, prompt_key: str, value: Any) -> None:
        self._entries[key] = PolicyKVCacheEntry(self.policy_version, prompt_key, value)


class TrajectoryTree:
    def __init__(self):
        self.nodes: list[TrajectoryNode] = [TrajectoryNode()]
        self.roots: dict[str, int] = {}

    def get_or_create_root(self, prompt_key: str) -> int:
        root_id = self.roots.get(prompt_key)
        if root_id is not None:
            return root_id
        if not self.roots and self.nodes[0].parent_id is None and self.nodes[0].token_id is None:
            root_id = 0
        else:
            root_id = len(self.nodes)
            self.nodes.append(TrajectoryNode())
        self.roots[prompt_key] = root_id
        return root_id

    def find_node(self, prompt_key: str, tokens: list[int]) -> Optional[int]:
        node_id = self.roots.get(prompt_key)
        if node_id is None:
            return None
        for token_id in tokens:
            node_id = self.nodes[node_id].children.get(int(token_id))
            if node_id is None:
                return None
        return node_id

    def observe(
        self,
        prompt_key: str,
        tokens: list[int],
        old_logprobs: Optional[list[float]] = None,
        reward: Optional[float] = None,
        policy_version: int = 0,
    ) -> None:
        node_id = self.get_or_create_root(prompt_key)
        for i, token_id in enumerate(tokens):
            token_id = int(token_id)
            node = self.nodes[node_id]
            child_id = node.children.get(token_id)
            if child_id is None:
                child_id = len(self.nodes)
                self.nodes.append(TrajectoryNode(token_id=token_id, parent_id=node_id))
                node.children[token_id] = child_id
            child = self.nodes[child_id]
            child.visit_count += 1
            child.policy_version_of_stats = policy_version
            if reward is not None and math.isfinite(float(reward)):
                reward_f = float(reward)
                child.reward_sum += reward_f
                child.reward_max = max(child.reward_max, reward_f)
            if old_logprobs is not None and i < len(old_logprobs):
                logp = float(old_logprobs[i])
                if math.isfinite(logp):
                    n = float(child.visit_count)
                    child.avg_old_logprob += (logp - child.avg_old_logprob) / n
                    child.avg_nll += ((-logp) - child.avg_nll) / n
            node_id = child_id

    def child_distribution(
        self,
        node_id: int,
        *,
        max_branch_width: int = 0,
        min_visits: int = 1,
        count_alpha: float = 1.0,
        reward_lambda: float = 0.0,
        nll_eta: float = 0.0,
        use_reward_prior: bool = False,
        use_nll_prior: bool = False,
        vocab_size: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
        node = self.nodes[node_id]
        scored: list[tuple[int, float, int]] = []
        for token_id, child_id in node.children.items():
            if vocab_size is not None and (token_id < 0 or token_id >= vocab_size):
                continue
            child = self.nodes[child_id]
            if child.visit_count < min_visits:
                continue
            score = max(float(child.visit_count), 0.0) ** float(count_alpha)
            if use_reward_prior:
                score *= math.exp(float(reward_lambda) * child.reward_mean)
            if use_nll_prior:
                score *= math.exp(-float(nll_eta) * child.avg_nll)
            if math.isfinite(score) and score > 0.0:
                scored.append((token_id, score, child_id))
        if not scored:
            return [], []
        scored.sort(key=lambda x: x[1], reverse=True)
        if max_branch_width and max_branch_width > 0:
            scored = scored[: int(max_branch_width)]
        total = sum(score for _, score, _ in scored)
        if not math.isfinite(total) or total <= 0.0:
            return [], []
        token_ids = [token_id for token_id, _, _ in scored]
        probs = [score / total for _, score, _ in scored]
        for prob, (_, _, child_id) in zip(probs, scored, strict=True):
            child = self.nodes[child_id]
            child.logq_tree_token = math.log(prob)
            child.nllq_tree_token = -child.logq_tree_token
        return token_ids, probs

    def propose_branch(
        self,
        node_id: int,
        *,
        max_depth: int,
        rng: random.Random,
        **dist_kwargs: Any,
    ) -> list[DraftProposal]:
        proposals: list[DraftProposal] = []
        cur_node_id = node_id
        for _ in range(max(1, int(max_depth))):
            token_ids, probs = self.child_distribution(cur_node_id, **dist_kwargs)
            if not token_ids:
                break
            token_id = rng.choices(token_ids, weights=probs, k=1)[0]
            q_prob = probs[token_ids.index(token_id)]
            child_id = self.nodes[cur_node_id].children[token_id]
            proposals.append(DraftProposal(token_id=token_id, child_node_id=child_id, logq_tree_token=math.log(q_prob)))
            cur_node_id = child_id
        return proposals


def residual_distribution_from_probs(
    p_probs: torch.Tensor,
    child_token_ids: torch.Tensor,
    q_probs_for_children: torch.Tensor,
) -> torch.Tensor:
    residual = p_probs.clone()
    residual[child_token_ids] -= q_probs_for_children.to(device=p_probs.device, dtype=p_probs.dtype)
    residual.clamp_min_(0)
    normalizer = residual.sum()
    if normalizer <= 0 or not torch.isfinite(normalizer):
        raise ValueError("residual distribution has zero or non-finite mass")
    return residual / normalizer


def emitted_distribution_after_verify(
    p_probs: torch.Tensor,
    child_token_ids: torch.Tensor,
    q_probs_for_children: torch.Tensor,
) -> torch.Tensor:
    q = torch.zeros_like(p_probs)
    q[child_token_ids] = q_probs_for_children.to(device=p_probs.device, dtype=p_probs.dtype)
    accepted = torch.minimum(p_probs, q)
    residual_mass = 1.0 - accepted.sum()
    if residual_mass <= 1e-12:
        return accepted / accepted.sum()
    residual = residual_distribution_from_probs(p_probs, child_token_ids, q_probs_for_children)
    return accepted + residual_mass * residual


class HistoryTreeSpeculativeRollout:
    def __init__(self, config: Any, pad_token_id: int, eos_token_id: Optional[int] = None):
        self.config = _cfg_get(config, "history_tree_speculation", None)
        self.enabled = _cfg_bool(self.config, "enabled", False)
        self.max_depth = int(_cfg_get(self.config, "max_depth", 1))
        self.max_branch_width = int(_cfg_get(self.config, "max_branch_width", 8))
        self.min_visits = int(_cfg_get(self.config, "min_visits", 1))
        self.count_alpha = float(_cfg_get(self.config, "count_alpha", 1.0))
        self.reward_lambda = float(_cfg_get(self.config, "reward_lambda", 0.0))
        self.nll_eta = float(_cfg_get(self.config, "nll_eta", 0.0))
        self.use_reward_prior = _cfg_bool(self.config, "use_reward_prior", False)
        self.use_nll_prior = _cfg_bool(self.config, "use_nll_prior", False)
        self.exact_residual = _cfg_bool(self.config, "exact_residual", True)
        self.debug_verify_distribution = _cfg_bool(self.config, "debug_verify_distribution", False)
        self.candidate_prompt_logprobs = int(_cfg_get(self.config, "candidate_prompt_logprobs", 20))
        self.max_residual_attempts = int(_cfg_get(self.config, "max_residual_attempts", 0))
        self.pad_token_id = int(pad_token_id)
        self.eos_token_ids = {int(eos_token_id)} if eos_token_id is not None else set()
        self.policy_cache = PolicyVersionCache()
        self.tree = TrajectoryTree()
        self.rng = random.Random(int(_cfg_get(config, "seed", 0)))
        self._warned_unavailable = False

    @property
    def policy_version(self) -> int:
        return self.policy_cache.policy_version

    def bump_policy_version(self) -> None:
        self.policy_cache.bump_policy_version()

    def _dist_kwargs(self) -> dict[str, Any]:
        return {
            "max_branch_width": self.max_branch_width,
            "min_visits": self.min_visits,
            "count_alpha": self.count_alpha,
            "reward_lambda": self.reward_lambda,
            "nll_eta": self.nll_eta,
            "use_reward_prior": self.use_reward_prior,
            "use_nll_prior": self.use_nll_prior,
        }

    def _clone_sampling_params(self, sampling_params: Any, **overrides: Any) -> Any:
        params = copy.deepcopy(sampling_params)
        for key, value in overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params

    def _vllm_input(self, token_ids: list[int], multi_modal_data: Optional[Any]) -> dict[str, Any]:
        item = {"prompt_token_ids": list(token_ids)}
        if multi_modal_data is not None:
            item["multi_modal_data"] = multi_modal_data
        return item

    def _sample_online_one(
        self,
        inference_engine: Any,
        prefix: list[int],
        multi_modal_data: Optional[Any],
        sampling_params: Any,
        lora_request: Optional[Any],
    ) -> tuple[int, float]:
        params = self._clone_sampling_params(sampling_params, max_tokens=1, n=1, logprobs=1)
        outputs = inference_engine.generate(
            prompts=[self._vllm_input(prefix, multi_modal_data)],
            sampling_params=params,
            lora_request=[lora_request] if lora_request is not None else None,
            use_tqdm=False,
        )
        sample = outputs[0].outputs[0]
        token_id = int(sample.token_ids[0])
        logprob = sample.logprobs[0][token_id].logprob
        return token_id, float(logprob)

    def _score_candidate_online(
        self,
        inference_engine: Any,
        prefix: list[int],
        token_id: int,
        multi_modal_data: Optional[Any],
        sampling_params: Any,
        lora_request: Optional[Any],
    ) -> Optional[float]:
        params = self._clone_sampling_params(
            sampling_params,
            max_tokens=1,
            n=1,
            logprobs=1,
            prompt_logprobs=self.candidate_prompt_logprobs,
        )
        try:
            outputs = inference_engine.generate(
                prompts=[self._vllm_input([*prefix, int(token_id)], multi_modal_data)],
                sampling_params=params,
                lora_request=[lora_request] if lora_request is not None else None,
                use_tqdm=False,
            )
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning(f"history tree candidate scoring failed; falling back to normal vLLM rollout: {exc}")
                self._warned_unavailable = True
            return None
        prompt_logprobs = getattr(outputs[0], "prompt_logprobs", None)
        if not prompt_logprobs:
            return None
        last = prompt_logprobs[-1]
        if last is None or int(token_id) not in last:
            return None
        return float(last[int(token_id)].logprob)

    def _sample_residual_one(
        self,
        inference_engine: Any,
        prefix: list[int],
        node_id: int,
        multi_modal_data: Optional[Any],
        sampling_params: Any,
        lora_request: Optional[Any],
    ) -> tuple[int, float]:
        if not self.exact_residual:
            return self._sample_online_one(inference_engine, prefix, multi_modal_data, sampling_params, lora_request)
        token_ids, q_probs = self.tree.child_distribution(node_id, **self._dist_kwargs())
        q_by_token = {int(t): float(q) for t, q in zip(token_ids, q_probs, strict=True)}
        attempts = 0
        while True:
            attempts += 1
            token_id, logp = self._sample_online_one(inference_engine, prefix, multi_modal_data, sampling_params, lora_request)
            p = math.exp(logp)
            q = q_by_token.get(token_id, 0.0)
            if q <= 0.0:
                return token_id, logp
            if p > q and self.rng.random() <= (1.0 - q / p):
                return token_id, logp
            if self.max_residual_attempts > 0 and attempts >= self.max_residual_attempts:
                logger.warning("history tree residual sampling reached max_residual_attempts; falling back to p sample")
                return self._sample_online_one(inference_engine, prefix, multi_modal_data, sampling_params, lora_request)

    def generate_sequences(
        self,
        inference_engine: Any,
        vllm_inputs: list[dict[str, Any]],
        sampling_params: Any,
        response_length: int,
        eos_token_id: Any,
        lora_requests: Optional[list[Any]] = None,
        ignore_eos: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        eos_ids = set(int(x) for x in eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {int(eos_token_id)}
        self.eos_token_ids.update(eos_ids)
        responses: list[list[int]] = []
        rollout_log_probs: list[list[float]] = []
        metrics = {
            "history_tree_enabled": 1.0,
            "tree_hit_rate": 0.0,
            "draft_tokens_proposed": 0.0,
            "draft_tokens_accepted": 0.0,
            "residual_rejection_rate": 0.0,
            "normal_fallback_rate": 0.0,
            "eos_from_draft_count": 0.0,
            "eos_from_residual_count": 0.0,
            "cache_hit_rate": 0.0,
            "verifier_extra_forward_count": 0.0,
        }
        tree_lookup_count = 0
        tree_hit_count = 0
        residual_count = 0
        normal_count = 0

        for seq_idx, item in enumerate(vllm_inputs):
            prefix = list(item["prompt_token_ids"])
            multi_modal_data = item.get("multi_modal_data")
            prompt_key = stable_prompt_key(prefix, multi_modal_data)
            generated: list[int] = []
            logps: list[float] = []
            lora_request = lora_requests[seq_idx] if lora_requests is not None else None

            while len(generated) < response_length:
                node_id = self.tree.find_node(prompt_key, generated)
                tree_lookup_count += 1
                proposals = []
                if node_id is not None:
                    proposals = self.tree.propose_branch(
                        node_id,
                        max_depth=min(self.max_depth, response_length - len(generated)),
                        rng=self.rng,
                        **self._dist_kwargs(),
                    )
                if not proposals:
                    normal_count += 1
                    token_id, logp = self._sample_online_one(
                        inference_engine, prefix, multi_modal_data, sampling_params, lora_request
                    )
                    generated.append(token_id)
                    logps.append(logp)
                    prefix.append(token_id)
                    if not ignore_eos and token_id in eos_ids:
                        break
                    continue

                tree_hit_count += 1
                stopped_block = False
                for proposal in proposals:
                    if len(generated) >= response_length:
                        break
                    metrics["draft_tokens_proposed"] += 1.0
                    metrics["verifier_extra_forward_count"] += 1.0
                    logp = self._score_candidate_online(
                        inference_engine,
                        prefix,
                        proposal.token_id,
                        multi_modal_data,
                        sampling_params,
                        lora_request,
                    )
                    if logp is None:
                        if not self._warned_unavailable:
                            logger.warning(
                                "history tree speculation needs vLLM prompt_logprobs for exact verification; "
                                "falling back to normal vLLM rollout"
                            )
                            self._warned_unavailable = True
                        return None
                    log_alpha = min(0.0, logp - proposal.logq_tree_token)
                    if math.log(max(self.rng.random(), 1e-12)) <= log_alpha:
                        metrics["draft_tokens_accepted"] += 1.0
                        generated.append(proposal.token_id)
                        logps.append(logp)
                        prefix.append(proposal.token_id)
                        if not ignore_eos and proposal.token_id in eos_ids:
                            metrics["eos_from_draft_count"] += 1.0
                            stopped_block = True
                            break
                        continue

                    residual_count += 1
                    token_id, residual_logp = self._sample_residual_one(
                        inference_engine,
                        prefix,
                        node_id,
                        multi_modal_data,
                        sampling_params,
                        lora_request,
                    )
                    generated.append(token_id)
                    logps.append(residual_logp)
                    prefix.append(token_id)
                    if not ignore_eos and token_id in eos_ids:
                        metrics["eos_from_residual_count"] += 1.0
                    stopped_block = True
                    break
                if stopped_block and (not ignore_eos and generated[-1] in eos_ids):
                    break

            responses.append(generated)
            rollout_log_probs.append(logps)
            self.tree.observe(prompt_key, generated, logps, reward=None, policy_version=self.policy_version)

        if tree_lookup_count > 0:
            metrics["tree_hit_rate"] = float(tree_hit_count) / float(tree_lookup_count)
        proposed = metrics["draft_tokens_proposed"]
        metrics["acceptance_rate"] = metrics["draft_tokens_accepted"] / proposed if proposed > 0 else 0.0
        metrics["average_accepted_length"] = metrics["draft_tokens_accepted"] / max(float(tree_hit_count), 1.0)
        denom = float(tree_hit_count + normal_count + residual_count)
        metrics["normal_fallback_rate"] = float(normal_count) / max(denom, 1.0)
        metrics["residual_rejection_rate"] = float(residual_count) / max(float(tree_hit_count), 1.0)
        return {"responses": responses, "rollout_log_probs": rollout_log_probs, "metrics": metrics}

    def update_tree_from_batch(self, batch: Any) -> dict[str, float]:
        if not self.enabled:
            return {"history_tree_enabled": 0.0}
        required = {"prompts", "responses", "response_mask", "old_log_probs"}
        if not required.issubset(set(batch.batch.keys())):
            return {"history_tree_update_missing_fields": 1.0}
        prompts = batch.batch["prompts"].detach().cpu()
        responses = batch.batch["responses"].detach().cpu()
        masks = batch.batch["response_mask"].detach().cpu()
        old_log_probs = batch.batch["old_log_probs"].detach().cpu()
        rewards = None
        if "token_level_scores" in batch.batch.keys():
            rewards = batch.batch["token_level_scores"].detach().cpu().sum(dim=-1)
        updated = 0
        for i in range(responses.shape[0]):
            prompt_tokens = [int(t) for t in prompts[i].tolist() if int(t) != self.pad_token_id]
            valid = int(masks[i].sum().item())
            tokens = [int(t) for t in responses[i, :valid].tolist()]
            logps = [float(x) for x in old_log_probs[i, :valid].tolist()]
            reward = float(rewards[i].item()) if rewards is not None else None
            self.tree.observe(stable_prompt_key(prompt_tokens), tokens, logps, reward=reward, policy_version=self.policy_version)
            updated += 1
        return {"history_tree_updated_sequences": float(updated), "history_tree_nodes": float(len(self.tree.nodes))}
