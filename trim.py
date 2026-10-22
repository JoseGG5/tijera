# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:33:00 2026

@author: Jose Antonio
"""

from sentence_transformers import SentenceTransformer

# Download from the Hub
model = SentenceTransformer("lightonai/mDenseOn")

# Run inference with multilingual queries and documents
queries = [
    "Quelle planète est connue comme la planète rouge ?",
    "Which planet is known as the Red Planet?",
]
documents = [
    "Was ist los frau klum",
    "Mars, connu pour son apparence rougeâtre, est souvent appelé la planète rouge.",
    "Marte, conocido por su apariencia rojiza, es a menudo llamado el Planeta Rojo.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]

query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents, prompt_name="document")
print(query_embeddings.shape, document_embeddings.shape)
# [2, 768] [4, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)


#%% Plot params

import torch.nn as nn
from matplotlib import pyplot as plt

transformer = model.transformers_model

tok_embeddings_params, other_layers = 0, 0
for name, module in transformer.named_modules():
    n_params = sum(
        p.numel() for p in module.parameters(recurse=False)
    )

    if name == "embeddings.tok_embeddings":
        tok_embeddings_params = n_params
    else:
        other_layers += n_params

plt.figure()
plt.bar(['Token embeddings', 'All other layers'], [int(tok_embeddings_params / 10e6), int(other_layers / 10e6)])
plt.ylabel('# params (Millions)')
print(f"The amount of params in the embedding matrix is {tok_embeddings_params}")


# %%

""" Multilingual models have a pretty huge embedding matrix. If we are only interested
    in a small set of language why would we want that full matrix? We can reduce the vocab_size to
    a number N (needs to be divisible by 8 or 64 to optimize gpu usage, most models nowadays have this).
    
    For this: we can build a corpus of the languages, mine a tokenizer,
    """


