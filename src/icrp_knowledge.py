"""Deterministic retrieval over the local ICRP JSONL knowledge index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def _tokens(value: str) -> set[str]:
    return {item.lower() for item in TOKEN_RE.findall(value)}


class ICRPKnowledgeBase:
    """Small local retriever with source-page provenance.

    The implementation intentionally uses lexical retrieval so experiments are
    reproducible without an embedding service. An embedding index can be added
    later without changing the record schema or caller interface.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.documents = self._read_jsonl(self.root / "documents.jsonl")
        self.chunks = self._read_jsonl(self.root / "chunks.jsonl")
        self._chunk_terms = [_tokens(item["text"]) for item in self.chunks]

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge index file not found: {path}")
        records: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        publication: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        scored: List[tuple[float, Dict[str, Any]]] = []
        for terms, record in zip(self._chunk_terms, self.chunks):
            if publication and record["publication"].lower() != publication.lower():
                continue
            if category and record["category"].lower() != category.lower():
                continue
            overlap = len(query_terms & terms)
            if not overlap:
                continue
            score = overlap / max(1.0, len(query_terms) ** 0.5)
            if query.lower() in record["text"].lower():
                score += 1.0
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [dict(record, retrieval_score=round(score, 6)) for score, record in scored[:top_k]]

    def context(
        self,
        query: str,
        *,
        top_k: int = 5,
        publication: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        records = self.search(
            query, top_k=top_k, publication=publication, category=category
        )
        if not records:
            return "No matching ICRP knowledge chunks were found."
        blocks = []
        for record in records:
            source = (
                f"[{record['publication']}, PDF page {record['page']}, "
                f"chunk {record['chunk_id']}]"
            )
            blocks.append(f"{source}\n{record['text']}")
        return "\n\n".join(blocks)

    def source_manifest(self) -> Dict[str, Any]:
        return self.manifest


def load_default(root: Path = Path("data/icrp_knowledge_base_v2")) -> ICRPKnowledgeBase:
    return ICRPKnowledgeBase(root)


class StructuredICRPKnowledgeBase(ICRPKnowledgeBase):
    """Query the v2 structured layer while retaining v1 text retrieval."""

    def __init__(self, root: Path = Path("data/icrp_knowledge_base_v2")):
        super().__init__(root)
        self.sections = self._read_jsonl(self.root / "sections.jsonl")
        self.entities = self._read_jsonl(self.root / "entities.jsonl")
        self.mechanism_facts = self._read_jsonl(self.root / "mechanism_facts.jsonl")
        self.parameter_candidates = self._read_jsonl(self.root / "parameter_candidates.jsonl")
        self.model_templates = self._read_jsonl(self.root / "model_templates.jsonl")
        self.knowledge_cards = self._read_jsonl(self.root / "knowledge_cards.jsonl")
        theoretical_path = self.root / "theoretical_verified_parameters.jsonl"
        self.theoretical_verified_parameters = (
            self._read_jsonl(theoretical_path) if theoretical_path.exists() else []
        )

    def search_facts(
        self,
        query: str,
        *,
        top_k: int = 10,
        fact_type: Optional[str] = None,
        publication: Optional[str] = None,
        reviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        query_terms = _tokens(query)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for fact in self.mechanism_facts:
            if publication and fact["source_ref"]["publication"].lower() != publication.lower():
                continue
            if fact_type and fact_type not in fact.get("fact_type", []):
                continue
            if reviewed_only and fact.get("review_status") != "verified":
                continue
            text_terms = _tokens(fact.get("claim", ""))
            overlap = len(query_terms & text_terms)
            if overlap:
                scored.append((overlap / max(1.0, len(query_terms) ** 0.5), fact))
        scored.sort(key=lambda item: (-item[0], item[1]["fact_id"]))
        return [dict(fact, retrieval_score=round(score, 6)) for score, fact in scored[:top_k]]

    def search_parameters(
        self,
        query: str,
        *,
        top_k: int = 10,
        publication: Optional[str] = None,
        verified_only: bool = True,
    ) -> List[Dict[str, Any]]:
        query_terms = _tokens(query)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for candidate in self.parameter_candidates:
            if publication and candidate["source_ref"]["publication"].lower() != publication.lower():
                continue
            if verified_only and candidate.get("review_status") != "verified":
                continue
            terms = _tokens(candidate.get("context", "")) | _tokens(candidate.get("unit_raw", ""))
            overlap = len(query_terms & terms)
            if overlap:
                scored.append((overlap / max(1.0, len(query_terms) ** 0.5), candidate))
        scored.sort(key=lambda item: (-item[0], item[1]["parameter_candidate_id"]))
        return [dict(item, retrieval_score=round(score, 6)) for score, item in scored[:top_k]]

    def templates_for(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = _tokens(query)
        scored = []
        for template in self.model_templates:
            terms = _tokens(" ".join([
                template.get("model_family", ""),
                template.get("route", ""),
                " ".join(template.get("compartment_hints", [])),
                " ".join(template.get("process_hints", [])),
            ]))
            overlap = len(query_terms & terms)
            scored.append((overlap, template))
        scored.sort(key=lambda item: (-item[0], item[1]["template_id"]))
        return [dict(item, retrieval_score=score) for score, item in scored[:top_k]]

    def verified_parameter_priors(self, query: str) -> List[Dict[str, Any]]:
        """Return reviewed parameter evidence for retrieval only.

        The v2 builder marks all regex-extracted values as needs_review. This
        method intentionally returns an empty list until a human review file
        promotes a record to verified. Even then, the record cannot become an
        executable ODE value without a new source-audited registry release.
        """
        query_terms = _tokens(query)
        records = []
        for item in self.theoretical_verified_parameters:
            terms = _tokens(" ".join([
                item.get("element", ""), item.get("model_family", ""),
                item.get("source_compartment", ""), item.get("target_compartment", ""),
                item.get("parameter_name", ""),
            ]))
            overlap = len(query_terms & terms)
            if overlap:
                records.append((overlap, item))
        records.sort(key=lambda value: (-value[0], value[1]["parameter_candidate_id"]))
        return [dict(item, retrieval_score=score) for score, item in records]

    def knowledge_cards_for(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = _tokens(query)
        scored = []
        for card in self.knowledge_cards:
            terms = _tokens(" ".join([
                card.get("title", ""),
                card.get("category", ""),
                card.get("model_family", ""),
                " ".join(card.get("route_hints", [])),
                " ".join(card.get("compartment_hints", [])),
                " ".join(card.get("process_hints", [])),
            ]))
            overlap = len(query_terms & terms)
            if overlap:
                scored.append((overlap / max(1.0, len(query_terms) ** 0.5), card))
        scored.sort(key=lambda item: (-item[0], item[1]["card_id"]))
        return [dict(card, retrieval_score=round(score, 6)) for score, card in scored[:top_k]]

    def structured_context(self, query: str, *, top_k: int = 4) -> Dict[str, Any]:
        """Return bounded, source-aware context suitable for an LLM prompt."""
        return {
            "cards": self.knowledge_cards_for(query, top_k=top_k),
            "templates": self.templates_for(query, top_k=top_k),
            "facts": self.search_facts(query, top_k=top_k),
            "verified_parameters": self.verified_parameter_priors(query),
            "parameter_policy": (
                "verified_parameters are retrieval evidence only; executable "
                "ODE values must come from the verified model registry"
            ),
        }
