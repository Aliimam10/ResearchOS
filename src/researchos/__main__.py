"""Small command-line entry points for a local ResearchOS corpus."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .retrieval import RetrievalFilters
from .service import ResearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local ResearchOS corpus.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    upload = subcommands.add_parser("upload", help="ingest one or more local PDFs")
    upload.add_argument("files", nargs="+", type=Path)

    discover = subcommands.add_parser("discover", help="search OpenAlex and add a scholarly topic corpus")
    discover.add_argument("topic")
    discover.add_argument("--limit", type=int, default=50)
    discover.add_argument("--year-from", type=int)
    discover.add_argument("--metadata-only", action="store_true", help="do not download OA PDFs")

    search = subcommands.add_parser("search", help="run hybrid retrieval over the local corpus")
    search.add_argument("question")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--year-from", type=int)
    search.add_argument("--year-to", type=int)

    ask = subcommands.add_parser("ask", help="run the evidence-first research workflow")
    ask.add_argument("question")
    ask.add_argument("--year-from", type=int)
    ask.add_argument("--year-to", type=int)

    subcommands.add_parser("status", help="show the local corpus counts")
    args = parser.parse_args()
    service = ResearchService()

    if args.command == "upload":
        for path in args.files:
            result = service.ingest_upload(path.name, path.read_bytes())
            print(f"{result.document.title}: {result.chunks_created} citable chunks")
            for warning in result.warnings:
                print(f"  warning: {warning}")
    elif args.command == "discover":
        result = service.discover_topic(
            args.topic,
            limit=args.limit,
            year_from=args.year_from,
            download_full_text=not args.metadata_only,
        )
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "search":
        hits = service.search(
            args.question,
            limit=args.limit,
            filters=RetrievalFilters(year_from=args.year_from, year_to=args.year_to),
        )
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))
    elif args.command == "ask":
        result = service.ask(args.question, filters={"year_from": args.year_from, "year_to": args.year_to})
        print(result.answer)
        print("\nResearch trace:")
        print(" → ".join(result.trace))
    else:
        print(json.dumps(service.corpus_summary(), indent=2))


if __name__ == "__main__":
    main()
