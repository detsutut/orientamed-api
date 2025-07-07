from typing import List, Union, Tuple
from collections import defaultdict, Counter
import random

class TopKReranker:
    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker.

        Args:
            k: RRF constant parameter (default: 60, commonly used value)
        """
        self.k = k

    def __top_k_with_sampling__(self, scored_lists: List[Tuple[Union[str, int], float]], higher_better=True) -> List[
        Tuple[Union[str, int], float]]:
        sorted_docs = sorted(scored_lists, key=lambda x: x[1], reverse=higher_better)
        top_docs = []
        score_buckets = defaultdict(list)

        for doc_id, score in sorted_docs:
            score_buckets[score].append((doc_id, score))

        sorted_scores = sorted(score_buckets.keys(), reverse=higher_better)

        for score in sorted_scores:
            group = score_buckets[score]
            remaining = self.k - len(top_docs)
            if remaining <= 0:
                break
            if len(group) <= remaining:
                top_docs.extend(group)
            else:
                top_docs.extend(random.sample(group, remaining))

        return top_docs

    def rerank(self,
            scored_lists: List[List[Tuple[Union[str, int], float]]],
            seed: Union[int, None] = None, higher_better: Union[bool, List[bool]] = True
    ) -> List[List[Tuple[Union[str, int], float]]]:
        """
        Rerank two independent lists of (document_id, score) and return the top K items from each.
        If scores tie at the K-th position, random sampling is used to break the tie.

        Args:
            scored_lists: Lists of List of (document_id, score) with document_id as str or int
            k: Number of top documents to return per list
            seed: Optional seed for reproducibility

        Returns:
            Tuple of two lists, each containing the top K (document_id, score) pairs
        """
        if seed is not None:
            random.seed(seed)
        top_k_scored_lists = []
        if type(higher_better) is bool:
            higher_better = [higher_better] * len(scored_lists)
        for scored_list, higher_better in zip(scored_lists, higher_better):
            top_k_scored_lists.append(self.__top_k_with_sampling__(scored_list, higher_better=higher_better))
        counter = Counter()
        for sublist in top_k_scored_lists:
            seen = set(x[0] for x in sublist)
            counter.update(seen)
        # Compute percentages
        result = [(key, counter[key] / len(top_k_scored_lists)) for key in counter]
        # Sort by percentage descending
        result.sort(key=lambda x: -x[1])
        return result

class RRFReranker:
    """
    Reciprocal Rank Fusion (RRF) reranker for combining multiple ranked lists.

    RRF Score = sum over all rankers of: 1 / (k + rank_i)
    where k is a constant (typically 60) and rank_i is the rank in ranker i
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker.

        Args:
            k: RRF constant parameter (default: 60, commonly used value)
        """
        self.k = k

    def rerank(self,
               ranked_lists: List[List[Union[str, int]]],
               weights: List[float] = None) -> List[Tuple[Union[str, int], float]]:
        """
        Rerank items using RRF algorithm.

        Args:
            ranked_lists: List of ranked lists, where each list contains items in rank order
            weights: Optional weights for each ranker (default: equal weights)

        Returns:
            List of (item, rrf_score) tuples sorted by RRF score in descending order
        """
        if not ranked_lists:
            return []

        # Set equal weights if not provided
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        elif len(weights) != len(ranked_lists):
            raise ValueError("Number of weights must match number of ranked lists")

        # Calculate RRF scores
        rrf_scores = defaultdict(float)

        for ranker_idx, ranked_list in enumerate(ranked_lists):
            weight = weights[ranker_idx]
            for rank, item in enumerate(ranked_list, 1):  # rank starts from 1
                rrf_scores[item] += weight * (1 / (self.k + rank))

        # Sort by RRF score (descending)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def rerank_with_scores(self,
                           scored_lists: List[List[Tuple[Union[str, int], float]]],
                           weights: List[float] = None) -> List[Tuple[Union[str, int], float]]:
        """
        Rerank items with their original scores using RRF algorithm.

        Args:
            scored_lists: List of lists containing (item, score) tuples,
                         assumed to be sorted by score in descending order
            weights: Optional weights for each ranker

        Returns:
            List of (item, rrf_score) tuples sorted by RRF score in descending order
        """
        # Extract just the items in rank order
        ranked_lists = [[item for item, score in scored_list]
                        for scored_list in scored_lists]

        return self.rerank(ranked_lists, weights)

    def get_top_k(self,
                  ranked_lists: List[List[Union[str, int]]],
                  k: int,
                  weights: List[float] = None) -> List[Union[str, int]]:
        """
        Get top-k items after RRF reranking.

        Args:
            ranked_lists: List of ranked lists
            k: Number of top items to return
            weights: Optional weights for each ranker

        Returns:
            List of top-k items
        """
        reranked = self.rerank(ranked_lists, weights)
        return [item for item, score in reranked[:k]]


if __name__ == "__main__":
    scored_lists_sorted = [[('A', 0.0), ('B', 0.0), ('C', 0.0), ('D', 0.1), ('E', 0.1), ('F', 0.1), ('G', 0.1), ('H', 0.2),
      ('G', 0.3), ('H', 0.3)], [('A', 20), ('B', 20), ('D', 20), ('F', 2), ('I', 2), ('J', 0.2), ('K', 0),
      ('L', 0)]]
    print("RERANK TOP K=5")
    top_k_reranker = TopKReranker(k=5)
    print(top_k_reranker.rerank(scored_lists_sorted, higher_better=[False, True]))
    print("RERANK RRF")
    rrf_reranker = RRFReranker(k=15)
    lists_sorted = [[item[0] for item in list] for list in scored_lists_sorted]
    print(rrf_reranker.rerank(lists_sorted))