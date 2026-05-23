"""Domain vocabulary utilities for narrative-causal token supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DomainVocab:
    tokenizer: Any
    mode: str = "full"
    terms: list[str] | None = None
    token_ids: list[int] | None = None

    @classmethod
    def from_config(cls, tokenizer, data_cfg: dict[str, Any] | None) -> "DomainVocab":
        data_cfg = data_cfg or {}
        requested_mode = data_cfg.get("vocab_mode")
        if requested_mode is None:
            has_boundaries = bool(
                data_cfg.get("domain_vocab_path")
                or data_cfg.get("domain_terms")
                or data_cfg.get("semantic_boundaries")
                or data_cfg.get("causal_terms")
            )
            mode = "bounded" if has_boundaries else "full"
        else:
            mode = str(requested_mode).lower()
        if mode == "full":
            return cls(tokenizer=tokenizer, mode="full", terms=None)

        terms = _load_terms_from_config(data_cfg)
        if not terms:
            raise ValueError("vocab_mode is not 'full' but no domain terms were provided.")
        return cls(tokenizer=tokenizer, mode=mode, terms=terms)

    def allowed_token_ids(self) -> list[int]:
        if self.token_ids is not None:
            return sorted({int(token_id) for token_id in self.token_ids})
        if self.mode == "full":
            return list(range(int(self.tokenizer.vocab_size)))

        allowed: set[int] = set()
        for term in self.terms or []:
            for candidate in (term, f" {term}"):
                token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(token_ids) == 1:
                    allowed.add(token_ids[0])
        if not allowed:
            raise ValueError("Domain vocabulary produced no single-token ids for this tokenizer.")
        return sorted(allowed)


def _load_terms_from_config(data_cfg: dict[str, Any]) -> list[str]:
    path = data_cfg.get("domain_vocab_path")
    if path:
        return _load_terms_file(path)

    for key in ("domain_terms", "semantic_boundaries", "causal_terms"):
        if isinstance(data_cfg.get(key), list):
            return [str(term).strip() for term in data_cfg[key] if str(term).strip()]
    return []


def _load_terms_file(path: str) -> list[str]:
    terms = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    if not terms:
        raise ValueError(f"No domain vocabulary terms found in {path}")
    return terms


def find_target_token_id(tokenizer, text: str, allowed_token_ids: set[int]) -> int | None:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.extend([stripped, f" {stripped}"])
    for word in stripped.split():
        candidates.extend([word, f" {word}"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1 and token_ids[0] in allowed_token_ids:
            return token_ids[0]
    for candidate in candidates:
        for token_id in tokenizer.encode(candidate, add_special_tokens=False):
            if token_id in allowed_token_ids:
                return token_id
    if len(allowed_token_ids) >= int(tokenizer.vocab_size):
        token_ids = tokenizer.encode(stripped, add_special_tokens=False)
        if token_ids:
            return token_ids[-1]
    return None
