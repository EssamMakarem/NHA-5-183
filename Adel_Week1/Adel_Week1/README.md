# Adel — Week 1

This package answers English questions about a selected museum artifact using Doha's knowledge file. It includes text chunking, a semantic embedding index, retrieval, local answer generation, source references, and an insufficient-information fallback.

## Files

- `data/knowledge_en.json`: the shared knowledge-file snapshot, with stable artifact IDs and source URLs.
- `rag_baseline/data.py`: corpus checks and overlapping text chunks with original character offsets.
- `rag_baseline/models.py`: MiniLM sentence embeddings and FLAN-T5-small answer generation.
- `rag_baseline/pipeline.py`: index creation, artifact filtering, retrieval, and source checks.
- `rag_baseline/__main__.py`: setup, indexing, and question commands.
- `index/`: the small saved chunk list and embedding matrix.
- `examples/`: recorded answers from actual local model runs.
- `tests/`: checks for chunking, retrieval isolation, source handling, and fallbacks.

The package contains 67 artifact records. Ten descriptions are unavailable; those IDs return the fallback. The index contains only usable descriptions. It does not include image recognition, a website, model training, or later-week integration.

## Setup

Tested with Python 3.11 on Windows and CPU PyTorch. Open a terminal in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m rag_baseline setup
```

The first setup needs an internet connection. It downloads the two public models to the normal Hugging Face cache outside this folder, then builds the index. No paid API or API key is required. Model weights are excluded from the submission ZIP. Allow approximately 500 MB for the weights, plus the Python environment.

Later commands load cached models locally and do not fetch museum pages. Keep the same Hugging Face cache available. If the cache is removed, run setup again. Setup errors are reported rather than replaced with fabricated example answers.

## Run

```powershell
.venv\Scripts\python -m rag_baseline ask --artifact BM_AMUN_RAM --question "Which king is between the ram's front legs?"
.venv\Scripts\python -m rag_baseline ask --artifact BM_TAWERET --question "What is its height in centimetres?"
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python evaluate_examples.py
```

Add `--output answer.json` to save the response. After changing the knowledge file, rebuild with `python -m rag_baseline index`. A mismatched corpus is rejected until the index is rebuilt.

The unit suite checks source handling, chunk coverage, artifact isolation, index/corpus consistency, and evidence fallbacks. `evaluate_examples.py` also runs 17 development questions through the cached MiniLM and FLAN models, checking eight source-supported answers and nine unsupported cases. It writes `examples/english_answer.json`, `examples/unsupported_question.json`, and the detailed `examples/verification.json`. These questions were used during development, so their results are smoke checks rather than unseen accuracy measurements.

Verified on October 6, 2026: all 17 unit tests and all 17 model smoke questions passed. The saved index contains 61 chunks from the 57 nonempty descriptions; all 67 IDs remain in the input corpus.

## How it works

Descriptions are split into chunks of at most 80 words with 16-word overlap. Source URLs, artifact IDs, and original character offsets stay with every chunk. MiniLM creates normalized 384-dimensional embeddings using attention-mask mean pooling. Retrieval first filters to the selected artifact ID, then ranks its chunks by cosine similarity. This prevents a similar artifact from supplying the answer.

FLAN-T5-small generates a short answer from the retrieved text. The answer must match a passage in that text; the displayed answer uses the complete source sentence containing the passage, with a numbered source. Keeping the sentence retains qualifications such as a model being a replica rather than the ancient object. The JSON records both the model's raw candidate and the displayed answer so this check can be inspected.

For simple questions such as "Is this exhibit a scan of the ancient original?", an explicit source statement can answer even when the small model abstains. This route requires the entire question predicate in an affirmative or negative source statement; it does not trust a bare model-generated yes/no. `answer_method` distinguishes it from the generated-span route.

Unknown IDs, blank descriptions, low retrieval similarity, unavailable measurements or prices, and current-information questions return an explicit fallback. The static knowledge file cannot establish today's room number or opening hours.

This is a small English baseline. It does not independently verify Doha's source text, and matching a quotation does not prove that every possible question was understood correctly. General open-ended questions and subtle false premises can still produce a weak answer or a conservative fallback. Source links are references recorded in the dataset, not evidence that pages were fetched at runtime. Arabic answers, fine-tuning, production evaluation, and the older CV evidence adapter are outside this submission.

## Models and data

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0.
- [FLAN-T5-small](https://huggingface.co/google/flan-t5-small), revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, Apache-2.0.

Both are pretrained models, not models trained by the team. The knowledge snapshot is shared input from Doha's Week 1 package; her source and provenance list contains the data attribution and review limits. The source URLs are also retained in this package's corpus and answers.
