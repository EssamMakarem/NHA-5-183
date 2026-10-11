"""Run development questions with the cached models and save actual responses."""
import json
from pathlib import Path
from rag_baseline.models import Encoder, Generator
from rag_baseline.pipeline import RAG

ROOT = Path(__file__).resolve().parent


def main():
    cases = json.loads((ROOT / "tests/questions.json").read_text(encoding="utf-8-sig"))
    rag = RAG(ROOT / "data/knowledge_en.json", ROOT / "index", Encoder(), Generator())
    results = []
    for case in cases:
        response = rag.ask(case["artifact_id"], case["question"])
        errors = []
        if case["supported"]:
            if response["status"] != "answered":
                errors.append("Supported question not answered")
            for term in case["expected_terms"]:
                if term.casefold() not in response["answer"].casefold():
                    errors.append(f"Missing factual evidence: {term}")
            if case["expected_source"] not in [s["url"] for s in response["sources"]]:
                errors.append("Missing exact supporting source")
            for term in case.get("expected_title_terms", []):
                if not any(term.casefold() in s["title"].casefold() for s in response["sources"]):
                    errors.append(f"Wrong source title: {term}")
            for source in response["sources"]:
                if source["artifact_id"] != case["artifact_id"]:
                    errors.append("Source from a different artifact")
                if source["evidence"] not in rag.records[case["artifact_id"]]["text"]:
                    errors.append("Evidence differs from the source text")
        else:
            if response["status"] != "insufficient_evidence" or response["sources"]:
                errors.append("Unsupported question received an apparent answer")
        results.append({"case": case["id"], "passed": not errors, "errors": errors,
                        "expected_supported": case["supported"], "response": response})
        print(f"{case['id']}: {'PASS' if not errors else 'FAIL'} | {response['status']} | {response['answer']}", flush=True)
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    for filename, case_id in (("english_answer.json", "S3"), ("unsupported_question.json", "U1")):
        response = next(r["response"] for r in results if r["case"] == case_id)
        (output / filename).write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed = sum(row["passed"] for row in results)
    report = {"evaluation_type": "development_smoke_questions_not_unseen_accuracy",
              "passed": passed, "total": len(results),
              "corpus_sha256": rag.index["corpus_sha256"],
              "indexed_chunks": len(rag.index["chunks"]),
              "records": len(rag.records),
              "limitations": "These cases were used during development. They are not a broad independent benchmark or a guarantee for every English question.",
              "results": results}
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Model checks: {passed}/{len(results)} passed", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
