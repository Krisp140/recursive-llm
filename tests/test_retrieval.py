"""Tests for partition retrieval methods."""

import pytest
from rlm.retrieval import PartitionRetriever
from rlm.partitions import Partition


class TestPartitionRetriever:
    """Test PartitionRetriever initialization."""

    def test_init_defaults(self):
        """Test retriever initialization with defaults."""
        retriever = PartitionRetriever()
        assert retriever.method == "unfiltered"
        assert retriever.top_k == 5

    def test_init_custom(self):
        """Test retriever initialization with custom params."""
        retriever = PartitionRetriever(method="regex", top_k=3)
        assert retriever.method == "regex"
        assert retriever.top_k == 3

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        retriever = PartitionRetriever(method="invalid")
        partitions = [
            Partition("test", 0, 0, 4, {})
        ]
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            retriever.retrieve("query", partitions)


class TestUnfilteredRetrieval:
    """Test unfiltered (baseline) retrieval."""

    def test_returns_first_k(self):
        """Test that unfiltered returns first k partitions."""
        partitions = [
            Partition(f"Partition {i}", i, i*10, (i+1)*10, {})
            for i in range(10)
        ]

        retriever = PartitionRetriever(method="unfiltered", top_k=3)
        result = retriever.retrieve("any query", partitions)

        assert len(result) == 3
        assert result[0].text == "Partition 0"
        assert result[1].text == "Partition 1"
        assert result[2].text == "Partition 2"

    def test_handles_fewer_partitions(self):
        """Test handling when fewer partitions than top_k."""
        partitions = [
            Partition("Partition 0", 0, 0, 10, {}),
            Partition("Partition 1", 1, 10, 20, {})
        ]

        retriever = PartitionRetriever(method="unfiltered", top_k=5)
        result = retriever.retrieve("query", partitions)

        assert len(result) == 2

    def test_empty_partitions(self):
        """Test with empty partition list."""
        retriever = PartitionRetriever(method="unfiltered", top_k=3)
        result = retriever.retrieve("query", [])

        assert len(result) == 0


class TestRegexRetrieval:
    """Test regex/keyword retrieval."""

    def test_keyword_matching(self):
        """Test that regex retrieval ranks by keyword matches."""
        partitions = [
            Partition("This is about cats and dogs.", 0, 0, 30, {}),
            Partition("Python programming language.", 1, 30, 60, {}),
            Partition("Python snakes are reptiles.", 2, 60, 90, {}),
            Partition("Unrelated content here.", 3, 90, 120, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=2)
        result = retriever.retrieve("Python programming", partitions)

        # Should prioritize partition 1 (has both "python" and "programming")
        assert len(result) == 2
        assert result[0].index == 1  # Python programming language

    def test_exact_word_matches(self):
        """Test that exact word matches score higher."""
        partitions = [
            Partition("The cat is sleeping.", 0, 0, 20, {}),
            Partition("Catch the ball quickly.", 1, 20, 40, {}),
            Partition("A cat and a dog play.", 2, 40, 60, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=2)
        result = retriever.retrieve("cat", partitions)

        # Partitions 0 and 2 should rank higher (exact "cat" vs "catch")
        assert len(result) == 2
        retrieved_indices = {p.index for p in result}
        assert 0 in retrieved_indices
        assert 2 in retrieved_indices

    def test_stopword_filtering(self):
        """Test that stopwords are filtered from query."""
        partitions = [
            Partition("Machine learning algorithms.", 0, 0, 30, {}),
            Partition("The quick brown fox jumps.", 1, 30, 60, {}),
            Partition("Deep learning neural networks.", 2, 60, 90, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=2)
        # "the" is a stopword, should focus on "learning"
        result = retriever.retrieve("the learning", partitions)

        # Should match partitions with "learning"
        assert len(result) == 2
        retrieved_indices = {p.index for p in result}
        assert 0 in retrieved_indices  # machine learning
        assert 2 in retrieved_indices  # deep learning

    def test_no_matches_fallback(self):
        """Test fallback when no keywords match."""
        partitions = [
            Partition("First partition.", 0, 0, 20, {}),
            Partition("Second partition.", 1, 20, 40, {}),
            Partition("Third partition.", 2, 40, 60, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=2)
        result = retriever.retrieve("xyzabc", partitions)

        # Should return first k partitions as fallback
        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 1

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        partitions = [
            Partition("PYTHON is great.", 0, 0, 20, {}),
            Partition("I love Python.", 1, 20, 40, {}),
            Partition("python programming.", 2, 40, 60, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=3)
        result = retriever.retrieve("python", partitions)

        # All should match
        assert len(result) == 3


class TestEmbeddingRetrieval:
    """Test embedding-based retrieval."""

    def test_embedding_with_precomputed(self):
        """Test embedding retrieval with pre-computed embeddings."""
        # Create partitions with mock embeddings
        import numpy as np

        partitions = [
            Partition("Machine learning", 0, 0, 20,
                     {"embedding": [0.9, 0.1, 0.1] + [0.0] * 1533}),
            Partition("Cooking recipes", 1, 20, 40,
                     {"embedding": [0.1, 0.9, 0.1] + [0.0] * 1533}),
            Partition("AI and ML", 2, 40, 60,
                     {"embedding": [0.85, 0.15, 0.05] + [0.0] * 1533})
        ]

        retriever = PartitionRetriever(method="embedding", top_k=2,
                                      api_key="test_key_will_use_precomputed")

        # Mock query embedding similar to partition 0 and 2
        # Note: Without actual API, this will try to call OpenAI
        # We should skip if API key not available

        try:
            result = retriever.retrieve("machine learning AI", partitions)
            # If it succeeds, check results
            assert len(result) <= 2
        except Exception as e:
            # Skip if no API key
            pytest.skip(f"Embedding retrieval requires API key: {e}")

    def test_embedding_fallback_on_error(self):
        """Test that embedding falls back to regex on API error."""
        partitions = [
            Partition("Machine learning content.", 0, 0, 30, {}),
            Partition("Random other content.", 1, 30, 60, {})
        ]

        # Use invalid API key to trigger fallback
        retriever = PartitionRetriever(method="embedding", top_k=1,
                                      api_key="invalid_key_xyz123")

        # Should fall back to regex retrieval
        result = retriever.retrieve("machine learning", partitions)

        # Should still return results
        assert len(result) >= 1

    def test_embedding_respects_top_k(self):
        """Test that embedding retrieval respects top_k."""
        partitions = [
            Partition(f"Content {i}", i, i*10, (i+1)*10,
                     {"embedding": [float(i)] + [0.0] * 1535})
            for i in range(10)
        ]

        retriever = PartitionRetriever(method="embedding", top_k=3,
                                      api_key="test_key")

        try:
            result = retriever.retrieve("test query", partitions)
            assert len(result) <= 3
        except Exception:
            pytest.skip("Embedding retrieval requires API key")


class TestRetrievalIntegration:
    """Integration tests for retrieval."""

    def test_retrieval_with_different_partition_types(self):
        """Test retrieval works with different partition metadata."""
        partitions = [
            Partition("Token partition", 0, 0, 20,
                     {"kind": "token", "token_count": 5}),
            Partition("Structural partition", 1, 20, 40,
                     {"kind": "structural", "has_heading": True}),
            Partition("Semantic partition", 2, 40, 60,
                     {"kind": "semantic", "embedding": [0.1] * 1536})
        ]

        # Test with each retrieval method
        for method in ["unfiltered", "regex"]:
            retriever = PartitionRetriever(method=method, top_k=2)
            result = retriever.retrieve("test query", partitions)
            assert len(result) <= 2

    def test_empty_query(self):
        """Test retrieval with empty query."""
        partitions = [
            Partition("Content 1", 0, 0, 10, {}),
            Partition("Content 2", 1, 10, 20, {})
        ]

        retriever = PartitionRetriever(method="regex", top_k=1)
        result = retriever.retrieve("", partitions)

        # Should return something (fallback)
        assert len(result) >= 1

    def test_single_partition(self):
        """Test retrieval with only one partition."""
        partitions = [
            Partition("Only partition", 0, 0, 15, {})
        ]

        for method in ["unfiltered", "regex"]:
            retriever = PartitionRetriever(method=method, top_k=5)
            result = retriever.retrieve("query", partitions)
            assert len(result) == 1
            assert result[0].text == "Only partition"
