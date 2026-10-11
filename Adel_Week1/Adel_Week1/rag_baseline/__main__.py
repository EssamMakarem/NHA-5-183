import argparse
import json
from pathlib import Path

from .models import Encoder, Generator
from .pipeline import RAG, build_index


def main():
    parser = argparse.ArgumentParser(description="Week 1 English artifact RAG baseline")
    parser.add_argument("command", choices=["setup", "index", "ask"])
    parser.add_argument("--corpus", type=Path, default=Path("data/knowledge_en.json"))
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--artifact")
    parser.add_argument("--question")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "ask" and (not args.artifact or not args.question):
        parser.error("ask requires --artifact and --question")
    encoder = Encoder(local_only=args.command != "setup")
    if args.command in {"setup", "index"}:
        if args.command == "setup":
            Generator(local_only=False)
        index = build_index(args.corpus, args.index, encoder)
        print(f"Indexed {len(index['chunks'])} chunks from {len(index['artifact_ids'])} artifact records.")
    else:
        rag = RAG(args.corpus, args.index, encoder, Generator())
        result = rag.ask(args.artifact, args.question)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
