"""Pinned, local CPU models; weights are stored in the Hugging Face cache."""
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GENERATION_MODEL = "google/flan-t5-small"
GENERATION_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


class Encoder:
    def __init__(self, local_only=True):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        args = dict(revision=EMBEDDING_REVISION, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, **args)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL, use_safetensors=True, **args).eval()

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        result = []
        for offset in range(0, len(texts), 16):
            encoded = self.tokenizer(texts[offset:offset + 16], padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > 256:
                raise ValueError("Embedding input exceeds 256 tokens; use smaller chunks")
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, dim=1)
            result.append(normalized.cpu().numpy().astype(np.float32))
        return np.concatenate(result)


class Generator:
    def __init__(self, local_only=True):
        args = dict(revision=GENERATION_REVISION, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL, **args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL, use_safetensors=True, **args).eval()

    def answer(self, question, context):
        prompt = (
            "Answer the question using only the context. Copy the answer from the context. "
            "If the context does not contain the answer, say 'not enough information'.\n"
            f"Context: {context}\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("Question and context exceed the generation budget")
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=80, do_sample=False, num_beams=1)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
