# Fast-BM25
A fast implementation of [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) in Python.  
BM25 is a simple and fast ranking function for search engines operating on words (tokens).  
It does not play well with misspelling so use it only in contexts where that's not a problem.

The base BM25 implementation is from [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py).

## How to use
Initialize BM25 by passing it a corpus, aka an iterator over tokenized documents (a list of Strings).
```py
from fast_bm25 import BM25

# Load your corpus
corpus = ...

bm25 = new BM25(corpus)
results = bm25.get_top_n(["largest", "city", "in", "Japan"], corpus);
```
*It's not a python package, copy the file if you want to use it*  

## Principle
In a text corpus, the most common words (the, a, an, ...) are often the least informative.  
By cutting them off from the query and only searching documents containing at least a word of the query, 
BM25 gain a lot of speed while loosing very little precision.  
This trade-off is controlled by the parameter `alpha`: higher alpha => more speed and more word cut-off.  
At $\alpha = -\inf$ the algorithm is equivalent to regular BM25.
