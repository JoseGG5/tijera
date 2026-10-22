# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:19:10 2026

@author: Jose Antonio
"""

""" Use two datasets: fineweb2 merged in english and spanish (both 200k samples), and a domain specific one (funding).
    Get the tokens distribution on each dataset and later assign a simple weighted sum of probabilities
    of each (avoiding fineweb taking strange but important tokens for a given domain)
    
"""


from collections import Counter
from typing import Literal

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def get_special_tokens(model_name: str) -> list[str]:
    """ Uses the AutoTokenizer API from transformer as tokenizers currently
        does not give the option to get the special tokens """
    
    return AutoTokenizer.from_pretrained(model_name).all_special_ids


class Miner:
    def __init__(
            self,
            bs: int,
            obj_tok: int,
            mode: Literal["fineweb", "specific", "mix"],
            model_trim: str = "lightonai/mDenseOn"
            ) -> None:
        self.bs = bs
        self.obj_tok = obj_tok  # (multiple of 8 for gpu going brrr)
        self.mode = mode
        self.model_trim = model_trim  # used to get the tokenizer of the model
        
        self.tokenizer = Tokenizer.from_pretrained(model_trim)
        self.n_tokens_keep = obj_tok - len(get_special_tokens(model_trim))
        self.ds = None
        
    
    def build_corpus(self):
        
        if self.mode == "fineweb":
            # compliant with https://huggingface.co/datasets/lbourdois/fineweb-2-trimming codes
            LANGUAGES_KEEP = ["eng", "spa"]
            
            # load (stream to avoid disk writing) and concat
            dss = []
            for lang in LANGUAGUES_KEEP:
                dss.append(
                    load_dataset(
                        "lbourdois/fineweb-2-trimming",
                        lang,
                        split="train",
                        streaming=True,
                        )
                    )
            ds = concatenate_datasets(dss)
            
        # TODO: Build and upload to HF <3
        elif self.mode == "specific":
            raise ValueError("Unimplemented")
        
        # TODO: Define mining process by giving weights to both datasets (specific and fineweb)
        elif self.mode == "mix":
            raise ValueError("Unimplemented")
 
        else:
            raise ValueError("Unimplemented")
        
        self.ds = ds
        
    
    def mine(self) -> list[int]:
        """ Mines from the selected corpus """
        
        # if the corpus is not built build it.
        if self.ds == None:
            self.build_corpus()

        # only for tqdm to work well (different dataset sizes)
        if self.mode == "fineweb":
            total_iters = int(400_000/self.bs)
        
        elif self.mode == "specific":
            raise ValueError("Unimplemented")
        
        elif self.mode == "mix":
            raise ValueError("Unimplemented")
 
        else:
            raise ValueError("Unimplemented")
        
        # iter dataset and get token frequencies
        token_freqs = Counter()
        total_tokens = 0
        
        for batch in tqdm(self.ds.iter(self.bs), desc="Batches tokenize", total=total_iters):    
            # batch fast (avoids offsets but we don't care)
            encodings_batched = self.tokenizer.encode_batch_fast(
                batch["text"],
                add_special_tokens=False  # we will never trim this ones
                )
            
            # gather the ids from the batch
            ids = []
            for item in encodings_batched:
                ids.extend(item.ids)
            
            # update counter and add total tokens
            token_freqs.update(ids)
            total_tokens += len(ids)
            
        # keep the N most common tokens
        tokens_keep = [tok_id for tok_id, _ in token_freqs.most_common(self.n_tokens_keep)]
        tokens_keep.extend(get_special_tokens(self.model_trim))
        
        return tokens_keep


if __name__ == "__main__":
    
    # compliant with https://huggingface.co/datasets/lbourdois/fineweb-2-trimming codes
    LANGUAGUES_KEEP = ["eng", "spa"]
    BATCH_SIZE_TOKENIZE = 10_000
    MODEL_TRIM = "lightonai/mDenseOn"
    OBJECTIVE_TOKENS = 32_768

    
    miner = Miner(
        bs=BATCH_SIZE_TOKENIZE,
        obj_tok=OBJECTIVE_TOKENS,
        mode="fineweb",
        model_trim=MODEL_TRIM
        )
    
    miner.mine()

