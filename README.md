Code to trim a given model (tested on lighton/mdenseon) but code modularized to work (with minor changes) on other architectures.

Trimming allows to remove tokens from the embedding matrix, which allows to delete tokens not needed for a given language that you want to preserve, or to make it faster
by making the vocab_size a multiple of 8 (for gpu threading).