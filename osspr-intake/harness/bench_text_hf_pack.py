"""Time ``TextDataset._load_hf_dataset`` on a multi-document calibration corpus.

The method tokenizes each row with a separate ``processor`` call, extends
Python lists of token ids, then slices those lists into 2048-token blocks
before copying them into tensors. Batching the tokenizer and packing the ids
with one tensor cat keeps the same block contents.
"""

from __future__ import annotations

import statistics
import time

import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from transformers import PreTrainedTokenizerFast

import angelslim.data.text_dataset as text_dataset


def _processor() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _corpus() -> DatasetDict:
    # 400 documents, 1600 whitespace tokens each: a small PTQ calibration set
    # that packs into many 2048-token blocks (the function's hardcoded size).
    text = "a b c d " * 400
    texts = [text for _ in range(400)]
    return DatasetDict({"train": Dataset.from_dict({"text": texts})})


def _time_once(processor, corpus) -> float:
    original = text_dataset.load_dataset
    text_dataset.load_dataset = lambda *args, **kwargs: corpus
    dataset = text_dataset.TextDataset.__new__(text_dataset.TextDataset)
    dataset.processor = processor
    dataset.device = "cpu"
    dataset.data = []
    try:
        started = time.perf_counter()
        dataset._load_hf_dataset("offline-text-dataset", num_samples=-1)
        return (time.perf_counter() - started) * 1000.0
    finally:
        text_dataset.load_dataset = original


def main() -> None:
    torch.set_num_threads(4)
    processor = _processor()
    corpus = _corpus()
    _time_once(processor, corpus)  # warmup
    samples = [_time_once(processor, corpus) for _ in range(5)]
    print(f"median_ms {statistics.median(samples):.6f}")


if __name__ == "__main__":
    main()
