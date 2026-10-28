Code to trim a given model (tested on lighton/mdenseon) but code modularized to work (with minor changes) on other architectures.

Trimming allows to remove tokens from the embedding matrix, which allows to delete tokens not needed for a given language that you want to preserve, or to make it faster
by making the vocab_size a multiple of 8 (for gpu threading). Some things in the code remain unimplemented such as trimming on specific corpus, but it's easy to do it
(I would probably use a mix between the specialized corpus and a generic one such as fineweb and consider the origin of mined tokens when selecting the vocab)

Everything in this repo except eval.py is fully written by me. Not because I think I can code better than AI Agents (I can't),
but because I wanted to go back to write code by myself and found this project interesting to optimize models that I wanted to test
for some other things I am building. I learned waaaaay more than if would have used AI Agents, although It would have probably taken half the time (if not less).

Inspo: https://huggingface.co/blog/lbourdois/introduction-to-trimming