import unittest
from core.reranker import RRFReranker


class TestRRFReranker(unittest.TestCase):

    def setUp(self):
        self.ranker1 = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        self.ranker2 = ["doc3", "doc1", "doc5", "doc2", "doc4"]
        self.ranker3 = ["doc2", "doc3", "doc1", "doc4", "doc5"]
        self.scored_list1 = [("doc1", 0.95), ("doc2", 0.88), ("doc3", 0.82), ("doc4", 0.75)]
        self.scored_list2 = [("doc3", 0.91), ("doc1", 0.87), ("doc5", 0.79), ("doc2", 0.71)]
        self.scored_list3 = [("doc2", 0.93), ("doc3", 0.86), ("doc1", 0.80), ("doc4", 0.72)]
        self.reranker = RRFReranker(k=60)

    def test_basic_rerank(self):
        result = self.reranker.rerank([self.ranker1, self.ranker2, self.ranker3])
        self.assertIsInstance(result, list)
        self.assertTrue(result[0][1] >= result[-1][1])  # Descending order
        items = [item for item, _ in result]
        self.assertTrue("doc1" in items)
        self.assertTrue("doc5" in items)

    def test_weighted_rerank(self):
        weights = [0.5, 0.3, 0.2]
        result = self.reranker.rerank([self.ranker1, self.ranker2, self.ranker3], weights)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_weight_mismatch(self):
        with self.assertRaises(ValueError):
            self.reranker.rerank([self.ranker1, self.ranker2], weights=[0.5])

    def test_rerank_with_scores(self):
        result = self.reranker.rerank_with_scores([self.scored_list1, self.scored_list2, self.scored_list3])
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertTrue(result[0][1] >= result[-1][1])  # Descending

    def test_get_top_k(self):
        top_k = self.reranker.get_top_k([self.ranker1, self.ranker2, self.ranker3], k=3)
        self.assertEqual(len(top_k), 3)
        self.assertTrue(all(isinstance(item, str) for item in top_k))

    def test_empty_input(self):
        result = self.reranker.rerank([])
        self.assertEqual(result, [])

    def test_single_ranking(self):
        result = self.reranker.rerank([self.ranker1])
        expected_scores = [(doc, 1 / (60 + rank)) for rank, doc in enumerate(self.ranker1, 1)]
        expected_scores = sorted(expected_scores, key=lambda x: x[1], reverse=True)
        self.assertEqual(result, expected_scores)


if __name__ == "__main__":
    unittest.main()
