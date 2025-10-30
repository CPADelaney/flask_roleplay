#!/usr/bin/env python3
"""CLI utility for loading documents into the hosted OpenAI vector store."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag.vector_store import (
    get_hosted_vector_store_ids,
    hosted_vector_store_enabled,
    legacy_vector_store_enabled,
    upsert_hosted_vector_documents,
)

logger = logging.getLogger(__name__)


def _normalise_metadata(values: Sequence[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for raw in values:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Metadata values must be KEY=VALUE pairs: {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Metadata key cannot be empty: {raw!r}")
        metadata[key] = value
    return metadata


def _ensure_metadata(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError("Document metadata must be a JSON object")


def _normalise_document(raw: Any, *, source: Path, index: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object for document #{index} in {source}")

    text = raw.get("text")
    if text is None:
        text = raw.get("content") or raw.get("memory_text")
    if text is None:
        raise ValueError(f"Document #{index} in {source} is missing a 'text' field")
    if not isinstance(text, str):
        text = str(text)

    document: Dict[str, Any] = {"text": text}

    identifier = raw.get("id") or raw.get("memory_id")
    metadata = _ensure_metadata(raw.get("metadata"))
    if identifier is None:
        identifier = metadata.get("memory_id")
    if isinstance(identifier, str) and identifier:
        document["id"] = identifier
        metadata.setdefault("memory_id", identifier)

    filename = raw.get("filename") or metadata.get("filename")
    if isinstance(filename, str) and filename.strip():
        document["filename"] = filename.strip()

    document["metadata"] = metadata
    return document


def _load_documents(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        documents: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}: {exc}"
                ) from exc
            documents.append(_normalise_document(entry, source=path, index=line_number))
        return documents

    if isinstance(payload, list):
        return [
            _normalise_document(item, source=path, index=idx + 1)
            for idx, item in enumerate(payload)
        ]
    return [_normalise_document(payload, source=path, index=1)]


def _iter_document_paths(directory: Path) -> Iterable[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".json", ".jsonl"}:
            yield entry


def _resolve_vector_store_id(
    explicit_id: Optional[str], configured_ids: Iterable[str]
) -> Optional[str]:
    if explicit_id:
        return explicit_id

    env_name = os.getenv("OPENAI_VECTOR_STORE_NAME")
    if env_name:
        return env_name.strip()

    for candidate in configured_ids:
        if candidate:
            return candidate
    return None


async def _async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)

    if args.input and args.directory:
        logger.error("Specify either an input file or --dir, not both")
        return 2

    document_sources: List[Path] = []

    if args.directory:
        directory = Path(args.directory).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            logger.error("Input directory does not exist: %s", directory)
            return 2
        document_sources.extend(_iter_document_paths(directory))
        if not document_sources:
            logger.warning("No JSON/JSONL documents found in %s; nothing to upload", directory)
            return 0
        document_path = directory
    elif args.input:
        document_path = Path(args.input).expanduser().resolve()
        if not document_path.exists():
            logger.error("Input file does not exist: %s", document_path)
            return 2
        document_sources.append(document_path)
    else:
        logger.error("No input provided. Pass a JSON file or --dir pointing to fixtures")
        return 2

    documents: List[Dict[str, Any]] = []
    for source in document_sources:
        try:
            documents.extend(_load_documents(source))
        except ValueError as exc:
            logger.error("%s", exc)
            return 2

    if not documents:
        logger.warning("No documents found in %s; nothing to upload", document_path)
        return 0

    configured_ids = get_hosted_vector_store_ids()
    vector_store_id = _resolve_vector_store_id(args.vector_store_id, configured_ids)
    if not vector_store_id:
        logger.error(
            "No vector store ID configured. Provide --vector-store-id or set "
            "OPENAI_VECTOR_STORE_NAME/AGENTS_VECTOR_STORE_IDS."
        )
        return 2

    if legacy_vector_store_enabled():
        logger.error(
            "ENABLE_LEGACY_VECTOR_STORE is active; hosted vector store uploads are disabled"
        )
        return 2

    if not hosted_vector_store_enabled([vector_store_id]):
        logger.error(
            "Hosted vector store backend is disabled. Check Agents dependencies or set "
            "DISABLE_AGENTS_VECTOR_STORE=0."
        )
        return 2

    common_metadata: Dict[str, Any] = {}
    if args.collection:
        common_metadata["collection"] = args.collection

    if args.metadata:
        try:
            common_metadata.update(_normalise_metadata(args.metadata))
        except ValueError as exc:
            logger.error("%s", exc)
            return 2

    for document in documents:
        metadata = _ensure_metadata(document.get("metadata"))
        metadata.update(common_metadata)
        document["metadata"] = metadata

    if args.dry_run:
        logger.info(
            "Dry-run complete. Prepared %s documents for vector store %s", len(documents), vector_store_id
        )
        return 0

    stored_ids = await upsert_hosted_vector_documents(
        documents,
        vector_store_id=vector_store_id,
        metadata={
            "component": "scripts.load_vector_store",
            "operation": "bulk_upsert",
            "document_path": str(document_path),
        },
    )

    logger.info(
        "Uploaded %s/%s documents to vector store %s", len(stored_ids), len(documents), vector_store_id
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load JSON/JSONL documents into the configured OpenAI vector store",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a JSON or JSONL file containing documents with text and metadata",
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        help="Load every JSON/JSONL document from this directory",
    )
    parser.add_argument(
        "--vector-store-id",
        dest="vector_store_id",
        help="Override the target vector store ID",
    )
    parser.add_argument(
        "--collection",
        help="Set the collection metadata on each uploaded document",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Additional metadata entries to attach to every document",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print progress without uploading documents",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
