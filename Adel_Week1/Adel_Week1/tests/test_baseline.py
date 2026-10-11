import json

import numpy as np
import pytest

from rag_baseline.data import chunk_records, load_records
from rag_baseline.pipeline import RAG, build_index, missing_attribute, supporting_sentence, explicit_confirmation


class FakeEncoder:
    def encode(self, texts):
        values = np.zeros((len(texts), 384), dtype=np.float32)
        values[:, 0] = 1
        return values


class FakeGenerator:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = 0

    def answer(self, question, context):
        self.calls += 1
        return self.candidate


def record(artifact_id="A", text="The ram shelters King Taharqo between its front legs."):
    return {"artifact_id": artifact_id, "title": "Ram", "language": "en",
            "text": text, "source_url": "https://example.org/" + artifact_id}


def pipeline(tmp_path, records, candidate="King Taharqo"):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(records), encoding="utf-8")
    encoder, generator = FakeEncoder(), FakeGenerator(candidate)
    build_index(corpus, tmp_path / "index", encoder)
    return RAG(corpus, tmp_path / "index", encoder, generator), corpus


def test_chunk_spans_cover_original_text_and_preserve_source():
    source = record(text=" ".join(f"word{i}" for i in range(203)))
    chunks = chunk_records([source])
    coverage = set()
    for chunk in chunks:
        assert chunk["text"] == source["text"][chunk["start_char"]:chunk["end_char"]]
        assert chunk["source_url"] == source["source_url"]
        assert len(chunk["text"].split()) <= 80
        coverage.update(chunk["text"].split())
    assert coverage == set(source["text"].split())
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_placeholder_is_not_evidence():
    assert chunk_records([record(text="_")]) == []


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_records([record()], 5, 5)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([record(), record()]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path)


def test_artifact_filter_before_ranking_and_index_roundtrip(tmp_path):
    rag, _ = pipeline(tmp_path, [record(), record("B", "The crown belongs to Nefertiti.")])
    assert [c["artifact_id"] for c in rag.retrieve("B", "Which king is with the ram?")] == ["B"]
    assert np.allclose(rag.vectors[:, 0], 1)


def test_answer_has_correct_source_and_generated_candidate(tmp_path):
    rag, _ = pipeline(tmp_path, [record()])
    answer = rag.ask("A", "Which king is between the ram's legs?")
    assert answer["status"] == "answered"
    assert answer["generator_used"]
    assert answer["generated_candidate"] == "King Taharqo"
    assert answer["sources"][0]["url"] == "https://example.org/A"
    assert answer["answer"] == record()["text"] + " [1]"


def test_hallucinated_candidate_is_rejected(tmp_path):
    rag, _ = pipeline(tmp_path, [record()], "King Tutankhamun")
    assert rag.ask("A", "Which king?")["status"] == "insufficient_evidence"


@pytest.mark.parametrize("artifact_id", ["UNKNOWN", "EMPTY"])
def test_no_evidence_does_not_invoke_generator(tmp_path, artifact_id):
    rag, _ = pipeline(tmp_path, [record(), record("EMPTY", "_")])
    answer = rag.ask(artifact_id, "Who made this?")
    assert answer["status"] == "insufficient_evidence"
    assert not answer["generator_used"]
    assert answer["sources"] == []


@pytest.mark.parametrize("question", ["What is its height in cm?", "What is the auction price?", "What is its current room number?"])
def test_missing_attributes_fall_back_even_with_high_similarity(tmp_path, question):
    rag, _ = pipeline(tmp_path, [record()])
    answer = rag.ask("A", question)
    assert answer["status"] == "insufficient_evidence"
    assert not answer["generator_used"]


def test_measurement_guard_does_not_block_supplied_measurement():
    assert missing_attribute("What is its height in cm?", "The figure is 17 cm high.") is None


def test_replica_qualification_and_negative_sentence_are_preserved():
    replica = "It is an educational exhibit and is not a scan of the ancient original."
    assert supporting_sentence("ancient original", replica) == replica
    negative = "However, not enough of it remains to determine anything about its content."
    assert supporting_sentence("its content", negative) == negative


def test_changed_corpus_requires_new_index(tmp_path):
    rag, corpus = pipeline(tmp_path, [record()])
    corpus.write_text(json.dumps([record(text="Changed source.")]), encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        RAG(corpus, tmp_path / "index", rag.encoder, rag.generator)


def test_confirmation_requires_the_entire_predicate_and_keeps_negation():
    evidence = "It is displayed as an educational 3D exhibit and is not a scan of the ancient original."
    assert explicit_confirmation("Is this exhibit a scan of the ancient original?", evidence) == evidence
    assert explicit_confirmation("Is this exhibit a scan of a different original?", evidence) is None
    assert explicit_confirmation("Is this figurine 25 cm tall?", "It depicts a goddess.") is None
    assert explicit_confirmation("Is this figure 25 cm tall?", "The figure is 25 cm tall and painted red.") is None


def test_source_can_answer_confirmation_when_generator_abstains(tmp_path):
    source = record(text="It is a modern replica of the bust and is not a scan of the ancient original.")
    rag, _ = pipeline(tmp_path, [source], "not enough information")
    result = rag.ask("A", "Is this exhibit a scan of the ancient original?")
    assert result["status"] == "answered"
    assert result["answer_method"] == "explicit_source_confirmation"
    assert "not a scan" in result["answer"]
