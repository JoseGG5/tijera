Code to trim a given model (tested on lighton/mdenseon) but code modularized to work (with minor changes) on other architectures.

Trimming allows to remove tokens from the embedding matrix, which allows to delete tokens not needed for a given language that you want to preserve, or to make it faster
by making the vocab_size a multiple of 8 (for gpu threading). Some things in the code remain unimplemented such as trimming on specific corpus, but it's easy to do it
(I would probably use a mix between the specialized corpus and a generic one such as fineweb and consider the origin of mined tokens when selecting the vocab)

Everything in this repo except eval.py is fully written by me. Not because I think I can code better than AI Agents (I can't) or I think AI Agents are useless (they are tremendously useful),
but because I enjoy coding by myself and found this project interesting to optimize models that I wanted to test
for some other things I am building. I learned waaaaay more than if would have used AI Agents, although It would have probably taken half the time (if not less).

Inspo: https://huggingface.co/blog/lbourdois/introduction-to-trimming

AI written summary of the project and repo structure:

## Project summary

This repository contains a modular pipeline for trimming a pretrained tokenizer vocabulary and its model embeddings. The current example targets `lightonai/mDenseOn` and selects a Spanish vocabulary. The goal is to retain useful tokens for a target language while reducing the embedding matrix; choosing a vocabulary size divisible by 8 can also be useful for GPU execution.

### Pipeline

1. **Mine token frequencies (`miner.py`)**: `Miner` streams the selected language split from `lbourdois/fineweb-2-trimming`, tokenizes its text with the original tokenizer, and counts token IDs. It also identifies special tokens and tokens required by byte-level or byte-fallback mechanisms.
2. **Select a vocabulary and BPE merges (`selector.py`)**: `Selector` starts with mandatory and special tokens, then considers tokens in descending frequency order. When retaining a token, it also retains the merge dependencies needed to construct it. It assigns contiguous IDs to the selected vocabulary and keeps compatible merges in their original order.
3. **Rebuild the tokenizer and model (`builder.py`)**: `Builder` replaces the tokenizer vocabulary and merges, updates added and special token IDs, creates a smaller embedding matrix, and copies the original embeddings for retained tokens. It updates the model vocabulary size and special-token IDs, then saves the Sentence Transformers model and a model card. Other model weights are left unchanged; this pipeline does not train or fine-tune the model.
4. **Evaluate (`eval.py`)**: evaluation code is kept separate from the trimming pipeline. See that module for its datasets, metrics, and invocation details.

### Current example and limitations

The example at the bottom of `builder.py` mines the Spanish (`spa`) FineWeb-2 trimming split, selects a vocabulary of 32,768 tokens for `lightonai/mDenseOn`, builds `mDenseOn-spa-32768`, and publishes it to the Hugging Face account configured in the code. The `Builder.build` method defaults to `publish=True`, so set `publish=False` when calling it if you only want a local model build.

`Miner` currently implements only `mode="fineweb"`. The `specific` and `mix` corpus modes raise `ValueError` and are not yet implemented. `main.py` is currently only a placeholder and does not run the pipeline; the example workflow is in `builder.py`. The selector uses the first recorded BPE dependency path for a token, and its source includes a TODO noting that its greedy budget approach may produce fewer tokens than the requested vocabulary size.

The default model, corpus, language, batch size, vocabulary target, and output directory are configured in the example code at the bottom of `builder.py`. The project requires Python 3.12 or newer and declares its dependencies in `pyproject.toml`.


