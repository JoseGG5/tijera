# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:33:00 2026

@author: Jose Antonio
"""

import json

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


def get_special_tokens(model_name: str) -> list[str]:
    """ Uses the AutoTokenizer API from transformer as tokenizers currently
        does not give the option to get the special tokens """
    
    return AutoTokenizer.from_pretrained(model_name).all_special_ids


def load_json(repo):
    path = hf_hub_download(repo, "tokenizer.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)



