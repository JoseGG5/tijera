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
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel

from utils import load_json


class Miner:
    def __init__(
            self,
            bs: int,
            mode: Literal["fineweb", "specific", "mix"],
            langs: list[str],
            model_trim: str = "lightonai/mDenseOn"
            ) -> None:
        self.bs = bs
        self.mode = mode
        self.model_trim = model_trim  # used to get the tokenizer of the model
        self.langs = langs  # compliant with https://huggingface.co/datasets/lbourdois/fineweb-2-trimming codes
        
        self.tokenizer = Tokenizer.from_pretrained(model_trim)
        self.ds = None
    
    def build_corpus(self):
        
        if self.mode == "fineweb":
            # load (stream to avoid disk writing) and concat
            dss = []
            for lang in self.langs:
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
    
    
    def get_mandatory_values(self) -> dict[str, int]:
        """Return existing mandatory tokens for the active byte mechanisms."""

        def has_type(config, target):
            if isinstance(config, dict):
                return (
                    config.get("type") == target
                    or any(has_type(value, target) for value in config.values())
                )
            if isinstance(config, list):
                return any(has_type(value, target) for value in config)
            return False
    
        json_tokenizer = load_json(self.model_trim)
        vocab = self.tokenizer.get_vocab()
    
        is_bytelevel = has_type(
            json_tokenizer.get("pre_tokenizer"), "ByteLevel"
        )
        has_byte_fallback = json_tokenizer["model"].get("byte_fallback", False)
    
        candidates = set()
        
        if is_bytelevel:
            candidates.update(ByteLevel.alphabet())
        
        # byte fallback tokens
        if has_byte_fallback:
            
            for byte in range(256):
                byte_token = f"<0x{byte:02X}>"
                
                # if exists in og vocab append it
                if byte_token in vocab:
                    candidates.add(byte_token)
                
                # there are some ascii characters that don't have the byte fallback token but a literal token
                # for mDenseOn I've seen the \t token
                elif byte < 128 and chr(byte) in vocab:
                    candidates.add(chr(byte))
            
            candidates.update(f"<0x{i:02X}>" for i in range(256))
    
        # ensure that only tokens from the tokenizer get in the candidate set
        mandatory_tokens = {
            token: vocab[token]
            for token in candidates
            if token in vocab
        }
        
        return mandatory_tokens
        
    
    def get_special_tokens(self) -> list[str]:
        """ Uses the AutoTokenizer API from transformer as tokenizers currently
            does not give the option to get the special tokens """
        
        # there are two types of special tokens:
            # 1. the ones returned by the autotokenizer api ([CLS], [PAD], etc)
            # 2. the ones that are added tokens marked as special ([end_of_turn], etc)
        
        # special tokens returned by the autotokenizer api
        special_ids = set(AutoTokenizer.from_pretrained(self.model_trim).all_special_ids)
        
        json_tokenizer = load_json(self.model_trim)
        
        # special tokens present in added_tokens
        special_ids.update(
            token["id"]
            for token in json_tokenizer.get("added_tokens", [])
            if token.get("special", False)
        )

        return sorted(special_ids)
        
        
    
        
    def mine(self) -> tuple[Counter[int], dict[str, int], dict[str, int]]:
        """ Mines from the selected corpus adn return the top k most
        frequent ones """
        
        # if the corpus is not built build it.
        if self.ds == None:
            self.build_corpus()

        # only for tqdm to work well (different dataset sizes)
        if self.mode == "fineweb":
            total_iters = int(len(self.langs) * 200_000/self.bs)
        
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
        
        mandatory_tokens = self.get_mandatory_values()
        special_ids = self.get_special_tokens()
        special_tokens = {self.tokenizer.id_to_token(id): id for id in special_ids}
        
        return token_freqs, mandatory_tokens, special_tokens


if __name__ == "__main__":

    BATCH_SIZE_TOKENIZE = 10_000
    MODEL_TRIM = "lightonai/mDenseOn"
    
    miner = Miner(
        bs=BATCH_SIZE_TOKENIZE,
        mode="fineweb",
        model_trim=MODEL_TRIM
        )
    
    token_freqs = miner.mine()
    