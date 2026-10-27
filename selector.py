# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:51:20 2026

@author: Jose Antonio
"""

""" Builds the new tokenizer from the mined tokens """

from collections import defaultdict

from tokenizers import Tokenizer
from tqdm import tqdm

from miner import Miner
from utils import load_json


class Selector:
    def __init__(
            self,
            token_freqs: dict[int, int],
            special_tokens: dict[str, int],
            mandatory_tokens: dict[str, int],
            obj_tokens: int,
            model_trim: str = "lightonai/mDenseOn"
            ) -> None:
        self.token_freqs = token_freqs
        self.special_tokens = special_tokens
        self.mandatory_tokens = mandatory_tokens
        self.model_trim = model_trim
        self.tokenizer = Tokenizer.from_pretrained(model_trim)
        
        if obj_tokens < len(mandatory_tokens) + len(special_tokens):
            raise ValueError(f"obj_tokens should be higher than {len(mandatory_tokens) + len(special_tokens)} ({len(mandatory_tokens)} mandatory + {len(special_tokens)} special)")
        
        self.obj_tokens = obj_tokens  # multiple of 8 for gpu going brrr
        self.deps_tree = None
                

    def get_dependencies(self) -> None:
        """ Builds a dict where for each token in the vocab 
            we get all needed merges to preserve (as well as independent tokens).
            Loads tokenizer.json as a json (currently tokenizers api does not allow to inspect merges) """
        
        json_tokenizer = load_json(self.model_trim)
        vocab = json_tokenizer["model"]["vocab"]
        
        parents = defaultdict(list)
        
        # store for each token all the merges that can get me there
        for merge in tqdm(json_tokenizer["model"]["merges"], desc="Inspect merges", total=len(json_tokenizer["model"]["merges"])):
            left, right = merge
            result = vocab[left + right]
            parents[result].append([vocab[left], vocab[right]])
        
        self.deps_tree = parents
    
    
    def get_deps(self, token_id: int) -> set[int]:
        """ Given a token, navigates the dependencies and adds all tokens needed """
        
        required = set()
        pending = [token_id]
    
        while pending:
            current = pending.pop()
    
            if current in required:
                continue
    
            required.add(current)
    
            parents = self.deps_tree.get(current, [])
            if parents:
                # Naive preserve the first one
                left, right = parents[0]
                pending.extend((left, right))
    
        return required
        
    
    def select(self) -> tuple[dict[str, int], list[tuple]]:
        """ Selects the final vocab and merges """
        
        # populate dependencies tree
        self.get_dependencies()
        
        # build the set of final token ids to keep
        final_ids = set(self.mandatory_tokens.values()).union(set(self.special_tokens.values()))
        
        # iter over the most frequent ones and get those and the needed ones
        for token_id, frequency in tqdm(self.token_freqs.most_common(), desc="Needed tokens", total=len(self.token_freqs)):
            required = self.get_deps(token_id)  
            missing = required - final_ids  # all the ones needed to get to that token except the ones already there
        
            if len(final_ids) + len(missing) <= self.obj_tokens:  # TODO: There is a chance we don't get the desired vocab
                final_ids.update(missing)  
    
        # convert to a dict
        final_vocab = {}
        for id in tqdm(final_ids, desc="Building vocab", total=len(final_ids)):
            final_vocab[self.tokenizer.id_to_token(id)] = id
        
        
        # get og merges and iterate over them. if both tokens on a merge and the resulting one
        # are on teh new vocab, keep the merge
        
        # merges can be seen by loading the tokenizer.json 
        json_tokenizer = load_json(self.model_trim)
        merges = json_tokenizer["model"]["merges"]
        
        # iter over merges (on the original order)
        final_merges = []
        for merge in tqdm(merges, desc="Building merges", total=len(merges)):
            left, right = merge
            if left in vocab and right in vocab and left + right in vocab:
                final_merges.append(merge)
        
        return final_vocab, final_merges
    
        

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
        
        
        
    
    



