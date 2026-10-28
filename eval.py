# -*- coding: utf-8 -*-
"""Compare original and trimmed retrievers on Spanish retrieval benchmarks.

Dataset rows are streamed into RAM. Models must already be available locally.
XQuAD and MuPLeR use their complete Spanish tasks; MIRACL uses its judged pool.
Run: python eval.py
"""

import os
import gc
import json
import hashlib
from math import log2
from time import perf_counter
from typing import Literal

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class Evaluator:
    def __init__(
            self,
            model_original: str = "lightonai/mDenseOn",
            model_trimmed: str = "results/mDenseOn-spa-32768",
            bs: int = 4,
            max_length: int = 1024,
            path_save: str = "eval_results",
            device: str | None = None
            ) -> None:
        self.model_original = model_original
        self.model_trimmed = model_trimmed
        self.bs = bs
        self.max_length = max_length
        self.path_save = path_save
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if bs < 1 or not 1 <= max_length <= 8192:
            raise ValueError("Use bs >= 1 and max_length between 1 and 8192.")

        self.benchmarks = {
            "xquad": {
                "dataset": "mteb/XQuADRetrieval",
                "split": "validation",
                "scope": "complete Spanish query set and corpus"
                },
            "miracl": {
                "dataset": "miracl/miracl",
                "split": "dev",
                "scope": "shared judged-passage pool; NOT official full-corpus MIRACL"
                },
            "mupler": {
                "dataset": "mteb/MuPLeR-retrieval",
                "split": "test",
                "scope": "complete Spanish legal query set and corpus"
                }
            }

    def load_benchmark(self, benchmark: Literal["xquad", "miracl", "mupler"]):
        """Read every Spanish corpus, query and relevance row through streaming."""

        config = self.benchmarks[benchmark]
        cache_dir = os.path.join(self.path_save, ".cache", benchmark)

        # MIRACL stores the annotated positive and negative passages with each query
        if benchmark == "miracl":
            data = list(load_dataset(
                "parquet",
                data_files=(
                    "https://huggingface.co/datasets/miracl/miracl/resolve/"
                    "refs%2Fconvert%2Fparquet/es/dev/0000.parquet"
                    ),
                split="train",
                streaming=True,
                cache_dir=cache_dir
                ))

            corpus = {}
            queries = {}
            relevant = {}
            for row in data:
                query_id = row["query_id"]
                queries[query_id] = row["query"]
                relevant[query_id] = {
                    passage["docid"] for passage in row["positive_passages"]
                    }
                for passage in row["positive_passages"] + row["negative_passages"]:
                    corpus[passage["docid"]] = (
                        passage["title"] + "\n" + passage["text"]
                        ).strip()

        # XQuAD and MuPLeR provide corpus, queries and qrels as separate subsets
        else:
            data = {}
            for name in ("corpus", "queries", "qrels"):
                stream = load_dataset(
                    config["dataset"],
                    f"es-{name}",
                    split=config["split"],
                    streaming=True,
                    cache_dir=cache_dir
                    )
                data[name] = list(stream)

            corpus = {
                row["id"]: (row.get("title", "") + "\n" + row["text"]).strip()
                for row in data["corpus"]
                }
            queries = {row["id"]: row["text"] for row in data["queries"]}
            relevant = {query_id: set() for query_id in queries}
            for row in data["qrels"]:
                if row["score"] > 0:
                    relevant[row["query-id"]].add(row["corpus-id"])

        if not queries or not all(relevant.values()):
            raise ValueError("Every query must have at least one relevant document.")
        if any(not ids.issubset(corpus) for ids in relevant.values()):
            raise ValueError("Relevance judgments reference documents outside the corpus.")

        # identify the streamed data used for this run without saving the dataset
        fingerprint = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

        return corpus, queries, relevant, fingerprint

    def metrics(self, ranking, relevant):
        """ Compute metrics """
        hits = [int(doc_id in relevant) for doc_id in ranking]
        discounts = [1 / log2(rank + 2) for rank in range(10)]
        dcg = sum(hit * discount for hit, discount in zip(hits, discounts))
        ideal = sum(discounts[:min(len(relevant), 10)])
        reciprocal_rank = next(
            (1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0
            )

        return {
            "nDCG@10": dcg / ideal,
            "MRR@10": reciprocal_rank,
            "Recall@10": sum(hits) / len(relevant)
            }

    def evaluate(self, model_id, corpus, queries, relevant):
        """ Run eval """
        model = SentenceTransformer(model_id, device=self.device, local_files_only=True)
        model.max_seq_length = self.max_length
        doc_ids = sorted(corpus)
        inputs = {
            "query": list(queries.values()),
            "document": [corpus[doc_id] for doc_id in doc_ids]
            }
        options = {
            "batch_size": self.bs,
            "normalize_embeddings": True,
            "convert_to_tensor": True
            }
        vectors = {}
        stats = {}
        seconds = 0.0

        # keep prompts, normalization and sequence limits identical for both models
        for kind, texts in inputs.items():
            tokens = model.tokenizer(
                [model.prompts[kind] + text for text in texts],
                truncation=False,
                verbose=False
                )
            lengths = [len(ids) for ids in tokens["input_ids"]]
            stats[kind] = {
                "mean_tokens": sum(lengths) / len(lengths),
                "truncated": sum(length > self.max_length for length in lengths)
                }

            # warm up before timing each encoding pass
            model.encode(texts[:1], prompt_name=kind, show_progress_bar=False, **options)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            started = perf_counter()
            vectors[kind] = model.encode(
                texts, prompt_name=kind, show_progress_bar=True, **options
                ).cpu()
            seconds += perf_counter() - started

        # search the same shared corpus for every query
        scores = vectors["query"] @ vectors["document"].T
        rankings = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        per_query = {
            query_id: self.metrics([doc_ids[i] for i in ranking], relevant[query_id])
            for query_id, ranking in zip(queries, rankings.tolist())
            }
        averages = {
            name: sum(result[name] for result in per_query.values()) / len(per_query)
            for name in ("nDCG@10", "MRR@10", "Recall@10")
            }

        # release the current model before loading the next one
        del model, vectors
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "model": model_id,
            "metrics": averages,
            "encode_seconds": seconds,
            "tokenization": stats,
            "per_query": per_query
            }

    def run(self, benchmarks: list[str] | None = None):
        """ Evaluate both models and save a separate report for each benchmark """
        if benchmarks is None:
            benchmarks = list(self.benchmarks)

        os.makedirs(self.path_save, exist_ok=True)
        torch.set_num_threads(4)
        reports = {}

        for benchmark in benchmarks:
            config = self.benchmarks[benchmark]
            corpus, queries, relevant, fingerprint = self.load_benchmark(benchmark)
            report = {
                "dataset": config["dataset"],
                "language": "es",
                "split": config["split"],
                "scope": config["scope"],
                "streaming": True,
                "queries": len(queries),
                "documents": len(corpus),
                "device": self.device,
                "max_length": self.max_length,
                "batch_size": self.bs,
                "data_sha256": fingerprint,
                "models": {}
                }
            output = os.path.join(self.path_save, f"{benchmark}_es.json")
            print(f"{benchmark}: {len(queries)} queries, {len(corpus)} documents.", flush=True)

            # save after each model so that the baseline is kept if a later run fails
            for label, model_id in (
                    ("original", self.model_original),
                    ("trimmed", self.model_trimmed)
                    ):
                result = self.evaluate(model_id, corpus, queries, relevant)
                report["models"][label] = result
                if label == "trimmed":
                    baseline = report["models"]["original"]["metrics"]
                    report["delta_trimmed_minus_original"] = {
                        name: score - baseline[name]
                        for name, score in result["metrics"].items()
                        }

                with open(output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(benchmark, label, result["metrics"], flush=True)

            reports[benchmark] = report
            print("Delta:", report["delta_trimmed_minus_original"], flush=True)
            print("Results:", output, flush=True)

        return reports


if __name__ == "__main__":

    evaluator = Evaluator(
        model_original="lightonai/mDenseOn",
        model_trimmed="results/mDenseOn-spa-32768",
        bs=4,
        max_length=1024,
        path_save="eval_results"
        )

    reports = evaluator.run()
