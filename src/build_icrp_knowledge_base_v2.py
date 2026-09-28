"""Build a structured, provenance-preserving ICRP knowledge layer.

This command consumes the local v1 page index. It does not silently convert
PDF text into executable parameters: extracted numeric candidates are marked
``needs_review`` and the fixed ODE registry remains the only parameter source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PUBLICATION_MODEL_TEMPLATES = {
    "ICRP Publication 66": {
        "template_id": "icrp66_human_respiratory_tract",
        "model_family": "respiratory_tract",
        "route": "inhalation",
        "compartment_hints": ["extrathoracic", "thoracic", "lymph_nodes", "blood"],
        "process_hints": ["deposition", "particle_transport", "dissolution", "absorption_to_blood"],
    },
    "ICRP Publication 100": {
        "template_id": "icrp100_human_alimentary_tract",
        "model_family": "alimentary_tract",
        "route": "ingestion",
        "compartment_hints": ["stomach", "small_intestine", "right_colon", "left_colon", "rectosigmoid", "blood"],
        "process_hints": ["transit", "absorption_to_blood", "faecal_excretion"],
    },
    "ICRP Publication 130": {
        "template_id": "icrp130_occupational_intake_framework",
        "model_family": "occupational_intake",
        "route": "inhalation_or_ingestion",
        "compartment_hints": ["intake", "blood", "systemic_tissues", "urinary_path", "faecal_path"],
        "process_hints": ["intake_to_blood", "systemic_retention", "urinary_excretion", "faecal_excretion"],
    },
    "ICRP Publication 134": {
        "template_id": "icrp134_occupational_intake_part2",
        "model_family": "element_specific_biokinetics",
        "route": "occupational_intake",
        "compartment_hints": ["intake", "blood", "systemic_tissues", "excreta"],
        "process_hints": ["element_specific_absorption", "retention", "excretion"],
    },
    "ICRP Publication 137": {
        "template_id": "icrp137_occupational_intake_part3",
        "model_family": "element_specific_biokinetics",
        "route": "occupational_intake",
        "compartment_hints": ["intake", "blood", "liver", "kidneys", "bone", "excreta"],
        "process_hints": ["rapid_clearance", "intermediate_clearance", "slow_retention", "urinary_excretion", "faecal_excretion"],
    },
    "ICRP Publication 141": {
        "template_id": "icrp141_occupational_intake_part4",
        "model_family": "element_specific_biokinetics",
        "route": "occupational_intake",
        "compartment_hints": ["intake", "blood", "systemic_tissues", "excreta"],
        "process_hints": ["absorption", "retention", "excretion"],
    },
    "ICRP Publication 151": {
        "template_id": "icrp151_occupational_intake_part5",
        "model_family": "element_specific_biokinetics",
        "route": "occupational_intake",
        "compartment_hints": ["intake", "blood", "systemic_tissues", "excreta"],
        "process_hints": ["absorption", "retention", "excretion", "progeny"],
    },
}

ENTITY_PATTERNS = {
    "nuclide": re.compile(r"\b(?:\d{1,3}[A-Z][a-z]?|[A-Z][a-z]?[- ]\d{1,3})\b"),
    "compartment": re.compile(
        r"\b(?:blood|stomach|small intestine|large intestine|right colon|left colon|"
        r"rectosigmoid|thyroid|liver|kidney(?:s)?|bone|skeleton|lung(?:s)?|"
        r"lymph nodes?|urinary bladder|systemic tissues?|extrathoracic|thoracic)\b",
        re.IGNORECASE,
    ),
    "route": re.compile(r"\b(?:inhalation|ingestion|injected|intravenous|wound|bloodstream)\b", re.IGNORECASE),
    "measurement": re.compile(r"\b(?:urine|faeces|feces|bioassay|whole[- ]body|retention)\b", re.IGNORECASE),
}

MECHANISM_RULES = {
    "compartment_definition": re.compile(r"\bcompartment\b", re.IGNORECASE),
    "transfer_process": re.compile(r"\b(?:transfer|transport|moves? from|movement from|release from)\b", re.IGNORECASE),
    "absorption": re.compile(r"\b(?:absorb|absorption|uptake|dissolution)\b", re.IGNORECASE),
    "retention": re.compile(r"\b(?:retention|retained|clearance|clear)\b", re.IGNORECASE),
    "excretion": re.compile(r"\b(?:excretion|excreted|urinary|faecal|fecal)\b", re.IGNORECASE),
    "half_time": re.compile(r"\bhalf[- ]?time\b", re.IGNORECASE),
    "parameter_table": re.compile(r"\b(?:parameter values?|transfer coefficients?|table\s+\d+(?:\.\d+)?)\b", re.IGNORECASE),
}

NUMBER_WITH_UNIT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*[−-]?\s*\d+)?)\s*(?P<unit>d(?:ays?)?\^?-?1|d\s*[−-]?1|h(?:ours?)?|y(?:ears?)?|%|fraction)",
    re.IGNORECASE,
)
HALF_TIME = re.compile(
    r"(?:half[- ]?time[^\d]{0,60})(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h(?:ours?)?|d(?:ays?)?|y(?:ears?)?)",
    re.IGNORECASE,
)
SECTION_HEADING = re.compile(r"^\s*((?:\d+\.){1,4}\d*|[A-Z]\.)\s+(.{4,160})$")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")
            count += 1
    return count


def _source(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": chunk["document_id"],
        "publication": chunk["publication"],
        "page": int(chunk["page"]),
        "chunk_id": chunk["chunk_id"],
        "official_url": chunk.get("official_url"),
    }


def _normalize_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.replace(" ", "_").replace("-", "_")


def _extract_sections(chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    current: Dict[str, str] = {}
    for chunk in chunks:
        lines = str(chunk.get("text", "")).replace("\r", "").splitlines()
        headings = []
        for line in lines:
            match = SECTION_HEADING.match(line.strip())
            if match:
                headings.append({"number": match.group(1), "title": match.group(2).strip()})
        if headings:
            current = headings[-1]
        records.append({
            "section_id": f"{chunk['chunk_id']}:section",
            "document_id": chunk["document_id"],
            "publication": chunk["publication"],
            "page": chunk["page"],
            "heading": current.get("title", ""),
            "heading_number": current.get("number", ""),
            "source_ref": _source(chunk),
            "status": "heuristic",
        })
    return records


def _extract_entities(chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[tuple[str, str], Dict[str, Any]] = {}
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for entity_type, pattern in ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                raw = re.sub(r"\s+", " ", match.group(0)).strip()
                key = (entity_type, _normalize_entity(raw))
                item = seen.setdefault(key, {
                    "entity_id": f"{entity_type}:{_normalize_entity(raw)}",
                    "entity_type": entity_type,
                    "canonical_name": _normalize_entity(raw),
                    "surface_forms": set(),
                    "mention_count": 0,
                    "source_refs": [],
                    "status": "candidate",
                })
                item["surface_forms"].add(raw)
                item["mention_count"] += 1
                if len(item["source_refs"]) < 20:
                    item["source_refs"].append(_source(chunk))
    result = []
    for item in seen.values():
        item["surface_forms"] = sorted(item["surface_forms"])
        result.append(item)
    result.sort(key=lambda item: (item["entity_type"], item["canonical_name"]))
    return result


def _extract_facts(chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text", "")).replace("\r", " ")).strip()
        if not text:
            continue
        labels = [name for name, pattern in MECHANISM_RULES.items() if pattern.search(text)]
        if not labels:
            continue
        # Keep short evidence units so the LLM sees a claim-sized context.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue
            sentence_labels = [name for name, pattern in MECHANISM_RULES.items() if pattern.search(sentence)]
            if not sentence_labels:
                continue
            facts.append({
                "fact_id": f"fact:{chunk['chunk_id']}:{len(facts)}",
                "fact_type": sentence_labels,
                "claim": sentence[:1200],
                "source_ref": _source(chunk),
                "extraction_method": "sentence_rule_tags",
                "review_status": "needs_review",
                "allowed_use": ["retrieval", "hypothesis_generation"],
                "forbidden_use": ["direct_parameter_injection"],
            })
    return facts


def _extract_parameter_candidates(chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text", "")).replace("\r", " ")).strip()
        contexts = []
        for match in HALF_TIME.finditer(text):
            contexts.append(("half_time", match, match.group(0)))
        for match in NUMBER_WITH_UNIT.finditer(text):
            contexts.append(("numeric_with_unit", match, match.group(0)))
        for kind, match, raw in contexts:
            start = max(0, match.start() - 160)
            end = min(len(text), match.end() + 160)
            records.append({
                "parameter_candidate_id": f"parameter:{chunk['chunk_id']}:{len(records)}",
                "kind": kind,
                "value_raw": match.group("value"),
                "unit_raw": match.group("unit"),
                "context": text[start:end],
                "source_ref": _source(chunk),
                "extraction_method": "regex_numeric_candidate",
                "review_status": "needs_review",
                "usable_as_ode_prior": False,
            })
    return records


def _model_templates(chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_publication: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_publication[str(chunk["publication"])].append(chunk)
    result = []
    for publication, template in PUBLICATION_MODEL_TEMPLATES.items():
        candidates = by_publication.get(publication, [])
        evidence = []
        keywords = ["biokinetic model", "compartment", "transfer", "excretion"]
        for chunk in candidates:
            text = str(chunk.get("text", "")).lower()
            if sum(keyword in text for keyword in keywords) >= 2:
                evidence.append(_source(chunk))
            if len(evidence) >= 8:
                break
        result.append({
            **template,
            "publication": publication,
            "template_status": "retrieval_context_only",
            "compartments": [
                {"id": item, "status": "hint"} for item in template["compartment_hints"]
            ],
            "transfers": [],
            "parameters": [],
            "evidence_refs": evidence,
            "allowed_use": ["retrieval", "planner_context"],
            "forbidden_use": [
                "runtime_model_generation", "transfer_editing",
                "clinical_assessment", "unreviewed_parameter_fitting",
            ],
        })
    return result


def _knowledge_cards(
    documents: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Create compact, auditable cards for LLM retrieval.

    Cards deliberately contain mechanism hints and source refs, not copied
    publication prose or unreviewed numeric parameters.
    """
    facts_by_publication: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_publication[str(fact["source_ref"]["publication"])].append(fact)
    entities_by_publication: Dict[str, Counter] = defaultdict(Counter)
    for entity in entities:
        for source in entity.get("source_refs", []):
            entities_by_publication[str(source["publication"])][entity["entity_type"]] += int(
                entity.get("mention_count", 0)
            )
    cards = []
    for document in documents:
        publication = str(document["publication"])
        template = next(
            (item for item in templates if item.get("publication") == publication), None
        )
        pub_facts = facts_by_publication.get(publication, [])
        evidence_refs = []
        for fact in pub_facts[:12]:
            ref = fact.get("source_ref")
            if ref not in evidence_refs:
                evidence_refs.append(ref)
        cards.append({
            "card_id": f"card:{document['document_id']}",
            "publication": publication,
            "document_id": document["document_id"],
            "title": document["title"],
            "category": document["category"],
            "official_url": document.get("official_url"),
            "model_family": template.get("model_family") if template else document["category"],
            "route_hints": [template.get("route")] if template and template.get("route") else [],
            "compartment_hints": template.get("compartment_hints", []) if template else [],
            "process_hints": template.get("process_hints", []) if template else [],
            "entity_summary": dict(entities_by_publication.get(publication, {})),
            "mechanism_fact_count": len(pub_facts),
            "evidence_refs": evidence_refs,
            "parameter_policy": "No numeric value is executable until manually verified.",
            "allowed_use": ["retrieval", "planner_context"],
            "forbidden_use": [
                "runtime_model_generation", "transfer_editing",
                "clinical_decision", "unreviewed_ODE_prior",
            ],
            "review_status": "source_indexed_structure_candidate",
        })
    return cards


def _apply_parameter_reviews(
    parameters: List[Dict[str, Any]], review_path: Path | None
) -> Dict[str, int]:
    if review_path is None or not review_path.exists():
        return {"reviewed": 0, "verified": 0, "rejected": 0}
    reviews = _read_jsonl(review_path)
    by_id = {item.get("parameter_candidate_id"): item for item in parameters}
    counts = {"reviewed": 0, "verified": 0, "rejected": 0}
    for review in reviews:
        candidate_id = str(review.get("parameter_candidate_id", ""))
        item = by_id.get(candidate_id)
        if item is None:
            continue
        status = str(review.get("review_status", "needs_review"))
        if status not in {"verified", "rejected", "needs_review"}:
            continue
        item["review_status"] = status
        item["review"] = {
            "reviewer": str(review.get("reviewer", "")),
            "review_note": str(review.get("review_note", "")),
            "reviewed_at": str(review.get("reviewed_at", "")),
        }
        if status == "verified":
            required = ("parameter_name", "value", "unit")
            if not all(key in review for key in required):
                item["review_status"] = "needs_review"
                continue
            item["verified_parameter"] = {
                "parameter_name": str(review["parameter_name"]),
                "value": float(review["value"]),
                "unit": str(review["unit"]),
                "lower": review.get("lower"),
                "upper": review.get("upper"),
                "applicability": review.get("applicability", {}),
            }
            item["usable_as_ode_prior"] = True
            counts["verified"] += 1
        elif status == "rejected":
            item["usable_as_ode_prior"] = False
            counts["rejected"] += 1
        counts["reviewed"] += 1
    return counts


def build(input_dir: Path, output_dir: Path, review_path: Path | None = None) -> Dict[str, Any]:
    chunks = _read_jsonl(input_dir / "chunks.jsonl")
    documents = _read_jsonl(input_dir / "documents.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = _extract_sections(chunks)
    entities = _extract_entities(chunks)
    facts = _extract_facts(chunks)
    parameters = _extract_parameter_candidates(chunks)
    review_counts = _apply_parameter_reviews(parameters, review_path)
    templates = _model_templates(chunks)
    cards = _knowledge_cards(documents, templates, facts, entities)
    # Executable transfer tables are maintained separately in the verified
    # ICRP model registry.  The knowledge layer must not manufacture a second
    # parameter source or silently promote extracted numbers to ODE values.
    theoretical_parameters: List[Dict[str, Any]] = []
    counts = {
        "documents": len(documents),
        "chunks": len(chunks),
        "sections": len(sections),
        "entities": len(entities),
        "mechanism_facts": len(facts),
        "parameter_candidates": len(parameters),
        "model_templates": len(templates),
        "knowledge_cards": len(cards),
        "theoretical_verified_parameters": 0,
        "parameter_reviews": review_counts,
    }
    for name, records in (
        ("documents.jsonl", documents),
        ("chunks.jsonl", chunks),
        ("sections.jsonl", sections),
        ("entities.jsonl", entities),
        ("mechanism_facts.jsonl", facts),
        ("parameter_candidates.jsonl", parameters),
        ("model_templates.jsonl", templates),
        ("knowledge_cards.jsonl", cards),
        ("theoretical_verified_parameters.jsonl", theoretical_parameters),
    ):
        _write_jsonl(output_dir / name, records)
    manifest = {
        "schema_version": "icrp_knowledge_base_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": str(input_dir),
        "counts": counts,
        "status_policy": {
            "candidate": "retrieval only until source review",
            "needs_review": "must be manually checked against cited PDF page/table",
            "verified": "reviewed retrieval evidence only; executable ODE values still require an explicit verified registry release",
        },
        "parameter_policy": "No automatically extracted numeric candidate is directly usable as an ODE prior.",
        "executable_model_policy": "Only data/icrp_model_registry_v1 contains executable verified ODE topology and rates.",
        "verified_registry": "data/icrp_model_registry_v1",
        "review_file": str(review_path) if review_path else None,
        "files": {
            "documents": "documents.jsonl",
            "chunks": "chunks.jsonl",
            "sections": "sections.jsonl",
            "entities": "entities.jsonl",
            "mechanism_facts": "mechanism_facts.jsonl",
            "parameter_candidates": "parameter_candidates.jsonl",
            "model_templates": "model_templates.jsonl",
            "knowledge_cards": "knowledge_cards.jsonl",
            "theoretical_verified_parameters": "theoretical_verified_parameters.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/icrp_knowledge_base_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/icrp_knowledge_base_v2"))
    parser.add_argument(
        "--review-file",
        type=Path,
        help="Optional JSONL file containing manually checked parameter candidates.",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.input_dir, args.output_dir, args.review_file), ensure_ascii=True))


if __name__ == "__main__":
    main()
