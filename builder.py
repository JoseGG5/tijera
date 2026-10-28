# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:48:01 2026

@author: Jose Antonio
"""

""" Builds the tokenizer and the model and uploads to HF """

import os
import json
import tempfile
from textwrap import dedent

from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from huggingface_hub import HfApi

from utils import load_json
from miner import Miner
from selector import Selector

class Builder:
    def __init__(
            self,
            vocab: dict[str, int],
            merges: list[tuple],
            model_trim: str = "lightonai/mDenseOn",
            path_save: str = "results"
            ):
        self.vocab = vocab
        self.merges = merges
        self.model_trim = model_trim
        self.path_save = path_save
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_trim)
    
    
    def build_tokenizer(self) -> None:
        """ Generates the new tokenizer.json file """
        
        # build the result folder
        os.makedirs(self.path_save, exist_ok = True)
        
        # load tokenizer as a dict        
        tokenizer_json = load_json(self.model_trim)
        
        # fix vocab and merges
        tokenizer_json["model"]["vocab"] = self.vocab
        tokenizer_json["model"]["merges"] = self.merges
        
        # fix added tokens
        tokenizer_json["added_tokens"] = [
            {**token, "id": self.vocab[token["content"]]}
            for token in tokenizer_json.get("added_tokens", [])
            if token["content"] in self.vocab
        ]
        
        # fix post processor
        post_processor = tokenizer_json.get("post_processor") or {}
        for tok, spec in post_processor.get("special_tokens", {}).items():
            spec["ids"] = [self.vocab[t] for t in spec["tokens"]]
        
        # on a temp folder we overwrite the tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            # save og tokenizer
            self.model[0].tokenizer.save_pretrained(tmp)
            
            # dump new tokenizer (modifying tokenizer.json)
            with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer_json, f, ensure_ascii=False)

            # load with autotokenizer in memory
            self.new_tokenizer = AutoTokenizer.from_pretrained(tmp)
        
        # check everything went fine
        assert len(self.new_tokenizer) == len(self.vocab), f"{len(self.new_tokenizer)} != {len(self.vocab)}"
        
        # tokenizer is a read-only property backed by processor
        self.model[0].processor = self.new_tokenizer
            
    
    def build_equivalence_ids(self) -> dict[int, int]:
        """ Builds a dict where keys are the old ids of a token and values
        are the new ones """
        
        tokenizer_json = load_json(self.model_trim)
        
        # iter over vocab
        ids_map = {}
        for token, id in tokenizer_json["model"]["vocab"].items():
            if token in self.vocab:
                ids_map[id] = self.vocab[token]
        
        return ids_map

    
    def build_model(self) -> None:
        """ Modifies the architecture """
        
        # build id_map
        id_map = self.build_equivalence_ids()
        
        # get the architecture
        arch = self.model.transformers_model
        
        # sentencetransformers api has a method to get the embeddings avoiding convention naming hell
        current_weights = arch.get_input_embeddings().weight
        
        # build the random new layer
        new_layer = nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=current_weights.size(1),
            padding_idx = self.new_tokenizer.pad_token_id,
            device=current_weights.device,
            dtype=current_weights.dtype
            )
        
        # iter over the current layer and use the id map to update new layer values 
        for idx, embedding in enumerate(current_weights):
            new_id = id_map.get(idx)
            if new_id is not None:
                with torch.no_grad():  # take gradients aout to avoid errors with inplace ops
                    new_layer.weight[new_id] = embedding
        
        # sentencetransformers api is so cool
        arch.set_input_embeddings(new_layer)
        
        # modify cfg
        cfg = arch.config
        cfg.vocab_size = len(self.vocab)
        
        # update special ids in cfg
        for field in (
            "bos_token_id", "eos_token_id", "pad_token_id",
            "cls_token_id", "sep_token_id", "mask_token_id",
        ):
            old_id = getattr(cfg, field, None)
            if old_id is not None:
                setattr(cfg, field, id_map[old_id])
                
        print(f"new embedding matrix size: {self.model.transformers_model.get_input_embeddings().weight.size()}")
        
    
    def build(self, name_model: str, publish: bool = True):
        
        # build assets
        self.build_tokenizer()
        self.build_model()
        
        save_dir = os.path.join(self.path_save, name_model)
       
        # save model 
        self.model.save_pretrained(save_dir, create_model_card=False)
        
        # save model card (description, results and metadata)
        description = f"""
            ## How this model was built
            
            This model was created by trimming the vocabulary of
            `{self.model_trim}` for use with Spanish text.

            ### Inspiration

            This project was inspired by
            [Introduction to Trimming](https://huggingface.co/blog/lbourdois/introduction-to-trimming)
            by Loïck Bourdois and collaborators, which explores vocabulary trimming
            as a way to reduce the size of multilingual models. Their work provided
            the motivation for this Spanish adaptation and its vocabulary selection
            approach. The implementation and evaluation results presented here
            correspond to this project.
            
            ### Procedure
            
            1. Token frequencies were computed using the Spanish (`spa`) split
               of `lbourdois/fineweb-2-trimming`.
            2. Special tokens and the tokens required by the tokenizer's
               byte representation mechanisms were preserved.
            3. Tokens were selected by frequency, including their required BPE
               dependencies, within the target vocabulary budget.
            4. The vocabulary and BPE merges were rebuilt, assigning contiguous
               token IDs.
            5. The original embeddings of the retained tokens were copied
               to their new positions.
            6. Non-special added tokens are retained only when included
               in the selected vocabulary.
            
            The resulting vocabulary contains **{len(self.vocab):,} tokens**.
            No additional training was performed. All other model weights
            remain unchanged.
            
            """

        results_summary = ""
        if (
            self.model_trim == "lightonai/mDenseOn"
            and name_model == "mDenseOn-spa-32768"
            and len(self.vocab) == 32768
        ):
            results_summary = dedent("""\
                ## Model reduction

                The following measurements and retrieval results correspond to the
                evaluated `mDenseOn-spa-32768` build of `lightonai/mDenseOn`, without
                additional training. Re-run evaluation after changing token selection
                or model weights.

                | Measure | Original | Trimmed | Absolute reduction | Reduction (%) |
                | --- | ---: | ---: | ---: | ---: |
                | Model weights (MB) | 1,227.77 | 542.00 | 685.77 MB | 55.85% |
                | Parameters | 306,939,648 | 135,497,472 | 171,442,176 | 55.86% |
                | Vocabulary (tokens) | 256,000 | 32,768 | 223,232 | 87.20% |

                Disk size refers to the FP32 `model.safetensors` files, with
                1 MB = 1,000,000 bytes. Parameter and vocabulary reductions are exact;
                sizes and percentages are rounded. The Transformer backbone weights
                are preserved, and retained token embeddings are copied from the original.

                ## Retrieval evaluation

                Both models were evaluated on identical queries and candidate documents,
                using the model's `query` and `document` prompts, normalized embeddings,
                and cosine similarity. The maximum sequence length was 1,024 tokens and
                the batch size was 4. Inference ran on an NVIDIA GeForce RTX 3060 Laptop GPU.
                All benchmark inputs were read using streaming and held in RAM.
                Scores below use a 0-1 scale. Delta is trimmed minus original.

                ### MIRACL-es: reduced shared corpus

                All 648 Spanish development queries were evaluated against a shared
                corpus of 6,410 unique passages, formed from the union of their annotated
                positive and negative passages. Every query searched the same corpus.

                **This is a reduced-corpus comparison, not the official full-corpus
                MIRACL benchmark. Its scores are not comparable to full-corpus leaderboard
                results.**

                | Metric | Original | Trimmed | Delta |
                | --- | ---: | ---: | ---: |
                | nDCG@10 | 0.8473 | 0.7937 | -0.0537 |
                | MRR@10 | 0.8622 | 0.8087 | -0.0534 |
                | Recall@10 | 0.9631 | 0.9245 | -0.0386 |

                The tokenizer exceeded the sequence limit for 2 documents with the
                original model and 3 with the trimmed model; those documents were truncated.
                No queries were truncated.

                Dataset: [MIRACL](https://huggingface.co/datasets/miracl/miracl).

                ### XQuADRetrieval-es: complete Spanish task

                All 1,184 Spanish validation queries were evaluated against all 240
                passages in the Spanish corpus of the published XQuADRetrieval task.
                Corpus, queries, and relevance judgments were loaded using streaming.
                No queries or documents were truncated for either model.

                | Metric | Original | Trimmed | Delta |
                | --- | ---: | ---: | ---: |
                | nDCG@10 | 0.9776 | 0.9642 | -0.0134 |
                | MRR@10 | 0.9704 | 0.9537 | -0.0168 |
                | Recall@10 | 0.9992 | 0.9958 | -0.0034 |

                Dataset: [XQuADRetrieval](https://huggingface.co/datasets/mteb/XQuADRetrieval).

                ### MuPLeR-retrieval-es: complete Spanish legal task

                All 200 Spanish test queries were evaluated against all 10,000 legal
                passages in the Spanish MuPLeR corpus. The passages come from European
                Union legislation (DGT-Acquis); the queries are synthetic. Every query
                searched the complete Spanish corpus. No queries or documents were
                truncated for either model.

                | Metric | Original | Trimmed | Delta |
                | --- | ---: | ---: | ---: |
                | nDCG@10 | 0.9035 | 0.8817 | -0.0219 |
                | MRR@10 | 0.8815 | 0.8525 | -0.0290 |
                | Recall@10 | 0.9700 | 0.9700 | 0.0000 |

                Dataset: [MuPLeR-retrieval](https://huggingface.co/datasets/mteb/MuPLeR-retrieval).

                ### Interpretation

                Trimming reduces weight-file size by 55.85%, but nDCG@10 is lower on
                all three evaluated tasks. On a 0-100 scale, nDCG@10 decreases by
                5.37 points on the reduced MIRACL corpus, 1.34 points on XQuADRetrieval,
                and 2.19 points on MuPLeR. Recall@10 remains unchanged on MuPLeR,
                while MRR@10 decreases by 2.90 points.
                These results describe the three evaluation settings above; they do not
                establish equivalent quality across all Spanish retrieval tasks.
                """)
         
        metadata = dedent(f"""\
        ---
        library_name: sentence-transformers
        pipeline_tag: sentence-similarity
        language:
        - es
        license: apache-2.0
        base_model:
        - {self.model_trim}
        tags:
        - sentence-transformers
        - trimmed
        ---
    
        # {name_model}
    
        Spanish vocabulary-trimmed version of `{self.model_trim}`.
        """)

        readme_path = os.path.join(save_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(metadata.strip() + "\n\n")
            f.write(dedent(description).strip() + "\n")
            if results_summary:
                f.write("\n" + results_summary.strip() + "\n")
        
    
        if publish:
            api = HfApi()
            repo_id = f"josegg5/{name_model}"
        
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
            )
        
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_dir,
                commit_message="Upload trimmed model and model card",
            )
        
            print(f"Published: https://huggingface.co/{repo_id}")
                    
    
    
if __name__ == "__main__":
    
    miner = Miner(
            bs=10_000,
            mode="fineweb",
            langs=["spa"]
        )
    
    freqs, mandatory, special = miner.mine()
            
    selector = Selector(
            token_freqs=freqs,
            special_tokens=special,
            mandatory_tokens=mandatory,
            obj_tokens=32768
        )
    
    
    vocab, merges = selector.select()
    
    builder = Builder(
        vocab=vocab,
        merges=merges
        )
    
    builder.build(name_model="mDenseOn-spa-32768", publish=True)
      
    

        
