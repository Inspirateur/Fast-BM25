import json
import math
import sys
import heapq
PARAM_K1 = 1.5
PARAM_B = 0.75
IDF_CUTOFF = 3


class BM25:
	"""Fast Implementation of Best Matching 25 ranking function.

	Attributes
	----------
	t2d : <token: <doc, freq>>
		Dictionary with terms frequencies for each document in `corpus`.
	idf: <token, idf score>
		Pre computed IDF score for every term.
	doc_len : list of int
		List of document lengths.
	avgdl : float
		Average length of document in `corpus`.
	"""

	def __init__(self, corpus, k1=PARAM_K1, b=PARAM_B, alpha=IDF_CUTOFF):
		"""
		Parameters
		----------
		corpus : list of list of str
			Given corpus.
		k1 : float
			Constant used for influencing the term frequency saturation. After saturation is reached, additional
			presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
			that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
			the type of documents or queries.
		b : float
			Constant used for influencing the effects of different document lengths relative to average document length.
			When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
			[1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
			depends on factors such as the type of documents or queries.
		alpha: float
			IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
			of BM25 but increase performance
		"""

		self.k1 = k1
		self.b = b
		self.alpha = alpha

		self.avgdl = 0
		self.t2d = {}
		self.idf = {}
		self.doc_len = []
		if corpus:
			self._initialize(corpus)

	@property
	def corpus_size(self):
		return len(self.doc_len)

	def _initialize(self, corpus):
		"""Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
		for i, document in enumerate(corpus):
			self.doc_len.append(len(document))

			for word in document:
				if word not in self.t2d:
					self.t2d[word] = {}
				if i not in self.t2d[word]:
					self.t2d[word][i] = 0
				self.t2d[word][i] += 1

		self.avgdl = sum(self.doc_len)/len(self.doc_len)
		to_delete = []
		for word, docs in self.t2d.items():
			idf = math.log(self.corpus_size - len(docs) + 0.5) - math.log(len(docs) + 0.5)
			# only store the idf score if it's above the threshold
			if idf > self.alpha:
				self.idf[word] = idf
			else:
				to_delete.append(word)
		print(f"Dropping {len(to_delete)} terms")
		for word in to_delete:
			del self.t2d[word]

		self.average_idf = sum(self.idf.values())/len(self.idf)

		if self.average_idf < 0:
			print(
				f'Average inverse document frequency is less than zero. Your corpus of {self.corpus_size} documents'
				' is either too small or it does not originate from natural text. BM25 may produce'
				' unintuitive results.',
				file=sys.stderr
			)

	def get_score(self, query, index):
		"""Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

		Parameters
		----------
		query : list of str
			The tokenized query to score.
		index : int
			Index of document in corpus selected to score with `query`.

		Returns
		-------
		float
			BM25 score.
		"""
		score = 0.0
		numerator_constant = self.k1 + 1
		denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
		for token in query:
			if token in self.t2d and index in self.t2d[token]:
				df = self.t2d[token][index]
				idf = self.idf[token]
				score += (idf * df * numerator_constant) / (df + denominator_constant)
		return score

	def get_top_n(self, query, documents, n=5):
		"""
		Retrieve the top n documents for the query.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents
		"""
		assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
		indexes = set(
			i
			for token in query
			if token in self.t2d
			for i in self.t2d[token].keys()
		)
		return [documents[i] for i in heapq.nlargest(n, indexes, key=lambda idx: self.get_score(query, idx))]

	def save(self, filename):
		json_object = {
			"k1": self.k1, "b": self.b, "alpha": self.alpha, "avgdl": self.avgdl,
			"t2d": self.t2d, "idf": self.idf, "doc_len": self.doc_len
		}
		with open(f"{filename}.json", "w") as fsave:
			json.dump(json_object, fsave)

	@staticmethod
	def load(filename):
		with open(f"{filename}.json", "r") as fsave:
			json_object = json.load(fsave)
		# we have to do this terribleness because json does not have int as keys
		for tok in json_object["t2d"]:
			json_object["t2d"][tok] = {int(i): f for i, f in json_object["t2d"][tok].items()}
		bm25 = BM25([])
		bm25.k1 = json_object["k1"]
		bm25.b = json_object["b"]
		bm25.alpha = json_object["alpha"]
		bm25.avgdl = json_object["avgdl"]
		bm25.t2d = json_object["t2d"]
		bm25.idf = json_object["idf"]
		bm25.doc_len = json_object["doc_len"]
		return bm25
