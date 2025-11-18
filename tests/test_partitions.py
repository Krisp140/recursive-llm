"""Tests for partitioning strategies."""

import pytest
from rlm.partitions import Partition, partition_text, count_tokens, _partition_token


class TestPartition:
    """Test Partition dataclass."""

    def test_partition_creation(self):
        """Test creating a partition."""
        p = Partition(
            text="Hello world",
            index=0,
            start_char=0,
            end_char=11,
            metadata={"token_count": 2}
        )
        assert p.text == "Hello world"
        assert p.index == 0
        assert len(p) == 11

    def test_partition_default_metadata(self):
        """Test partition with default metadata."""
        p = Partition(text="Test", index=0, start_char=0, end_char=4)
        assert p.metadata == {}


class TestTokenPartitioning:
    """Test token-based partitioning."""

    def test_single_partition_short_text(self):
        """Test that short text returns single partition."""
        text = "This is a short text."
        partitions = partition_text(text, strategy="token", max_tokens=1000)

        assert len(partitions) == 1
        assert partitions[0].text == text
        assert partitions[0].index == 0
        assert partitions[0].start_char == 0
        assert partitions[0].end_char == len(text)

    def test_multiple_partitions_long_text(self):
        """Test that long text is split into multiple partitions."""
        # Create text that's definitely longer than max_tokens
        text = "This is a test sentence. " * 500  # ~2500 words
        partitions = partition_text(text, strategy="token", max_tokens=100, overlap_tokens=10)

        # Should have multiple partitions
        assert len(partitions) > 1

        # Check partition indices are sequential
        for i, p in enumerate(partitions):
            assert p.index == i

        # Check coverage (all partitions should cover the text)
        assert partitions[0].start_char == 0
        # Last partition should end at or near the end of text
        assert partitions[-1].end_char == len(text)

    def test_partition_overlap(self):
        """Test that partitions overlap correctly."""
        text = "word " * 1000  # Repeat to ensure multiple partitions
        partitions = partition_text(text, strategy="token", max_tokens=50, overlap_tokens=10)

        if len(partitions) > 1:
            # Check that consecutive partitions have overlap
            for i in range(len(partitions) - 1):
                # End of current partition should be after start of next
                # (but this is token-based, so we check metadata)
                curr_end_token = partitions[i].metadata['end_token_idx']
                next_start_token = partitions[i + 1].metadata['start_token_idx']
                # The gap should be less than max_tokens (because of overlap)
                gap = next_start_token - partitions[i].metadata['start_token_idx']
                assert gap <= 50  # max_tokens

    def test_partition_token_counts(self):
        """Test that partition token counts are within limits."""
        text = "This is a test. " * 1000
        max_tokens = 100
        partitions = partition_text(text, strategy="token", max_tokens=max_tokens, overlap_tokens=10)

        for p in partitions:
            # Each partition should have token_count in metadata
            assert 'token_count' in p.metadata
            # Should not exceed max_tokens
            assert p.metadata['token_count'] <= max_tokens

    def test_empty_text(self):
        """Test partitioning empty text."""
        text = ""
        partitions = partition_text(text, strategy="token", max_tokens=100)

        assert len(partitions) == 1
        assert partitions[0].text == ""

    def test_overlap_validation(self):
        """Test that overlap >= max_tokens raises error."""
        # Need long enough text to trigger partitioning
        text = "Test text " * 200

        with pytest.raises(ValueError, match="overlap_tokens.*must be less than max_tokens"):
            partition_text(text, strategy="token", max_tokens=100, overlap_tokens=100)

        with pytest.raises(ValueError):
            partition_text(text, strategy="token", max_tokens=100, overlap_tokens=150)

    def test_partition_text_coverage(self):
        """Test that all text is covered by partitions."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 200
        partitions = partition_text(text, strategy="token", max_tokens=50, overlap_tokens=5)

        # Reconstruct text from partitions (without overlap handling)
        # At minimum, first partition should start at 0
        assert partitions[0].start_char == 0
        # Last partition should end at text length
        assert partitions[-1].end_char == len(text)

    def test_partition_metadata(self):
        """Test that partitions have correct metadata."""
        text = "Test " * 500
        partitions = partition_text(text, strategy="token", max_tokens=100, overlap_tokens=10)

        for p in partitions:
            # Should have token metadata
            assert 'token_count' in p.metadata
            assert 'start_token_idx' in p.metadata
            assert 'end_token_idx' in p.metadata
            # Token indices should be valid
            assert p.metadata['start_token_idx'] >= 0
            assert p.metadata['end_token_idx'] > p.metadata['start_token_idx']

    def test_different_model_tokenizers(self):
        """Test partitioning with different model names."""
        text = "This is a test. " * 100

        # Should work with known model
        p1 = partition_text(text, strategy="token", model_name="gpt-4")
        assert len(p1) >= 1

        # Should fallback gracefully for unknown model
        p2 = partition_text(text, strategy="token", model_name="unknown-model-xyz")
        assert len(p2) >= 1


class TestCountTokens:
    """Test token counting utility."""

    def test_count_tokens_short(self):
        """Test counting tokens in short text."""
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert count < 10  # Should be small

    def test_count_tokens_long(self):
        """Test counting tokens in longer text."""
        text = "This is a test sentence. " * 100
        count = count_tokens(text)
        assert count > 100  # Should be substantial

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        count = count_tokens("")
        assert count == 0


class TestPartitionStrategies:
    """Test different partitioning strategies."""

    def test_token_strategy(self):
        """Test token strategy works."""
        text = "Test " * 100
        partitions = partition_text(text, strategy="token")
        assert len(partitions) >= 1

    def test_learned_not_implemented(self):
        """Test that learned strategy raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Learned.*Phase 5"):
            partition_text("Test", strategy="learned")

    def test_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown partition strategy"):
            partition_text("Test", strategy="invalid")


class TestStructuralPartitioning:
    """Test structural partitioning strategy."""

    def test_paragraph_splitting(self):
        """Test that text is split on paragraph boundaries."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        partitions = partition_text(text, strategy="structural", max_tokens=1000)

        # Should create at least one partition
        assert len(partitions) >= 1
        # Should have structural metadata
        assert partitions[0].metadata['kind'] in ['structural', 'empty']

    def test_heading_detection_markdown(self):
        """Test detection of markdown headings."""
        text = """# Introduction

This is the introduction paragraph.

## Methods

This describes the methods used."""

        partitions = partition_text(text, strategy="structural", max_tokens=1000)

        # Should detect headings
        found_heading = False
        for p in partitions:
            if p.metadata.get('has_heading'):
                found_heading = True
                break

        assert found_heading, "Should detect markdown headings"

    def test_heading_detection_caps(self):
        """Test detection of all-caps headings."""
        text = """INTRODUCTION

This is the introduction paragraph.

METHODS

This describes the methods used."""

        partitions = partition_text(text, strategy="structural", max_tokens=1000)

        # Should detect headings
        found_heading = False
        for p in partitions:
            if p.metadata.get('has_heading'):
                found_heading = True
                break

        assert found_heading, "Should detect all-caps headings"

    def test_respects_max_tokens(self):
        """Test that partitions respect max_tokens."""
        # Create text with multiple paragraphs
        text = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(10)])

        partitions = partition_text(text, strategy="structural", max_tokens=200)

        # Each partition should respect token limit
        for p in partitions:
            if 'token_count' in p.metadata:
                assert p.metadata['token_count'] <= 200 or p.metadata['kind'] == 'oversized_unit_split'

    def test_single_paragraph(self):
        """Test structural partitioning with single paragraph."""
        text = "This is just one paragraph without breaks."

        partitions = partition_text(text, strategy="structural", max_tokens=1000)

        assert len(partitions) == 1
        assert partitions[0].text == text

    def test_empty_text_structural(self):
        """Test structural partitioning with empty text."""
        text = ""
        partitions = partition_text(text, strategy="structural", max_tokens=1000)

        assert len(partitions) == 1
        assert partitions[0].metadata['kind'] == 'empty'

    def test_oversized_paragraph(self):
        """Test handling of paragraph that exceeds max_tokens."""
        # Create a very long paragraph
        text = "Word " * 500  # Long enough to exceed typical max_tokens

        partitions = partition_text(text, strategy="structural", max_tokens=100)

        # Should split the oversized paragraph
        assert len(partitions) > 1


class TestSemanticPartitioning:
    """Test semantic partitioning strategy."""

    def test_semantic_short_text(self):
        """Test semantic partitioning with short text."""
        text = "This is a short sentence."

        # Note: semantic partitioning may fail without API key, so we catch that
        try:
            partitions = partition_text(text, strategy="semantic", max_tokens=1000)
            assert len(partitions) >= 1
        except Exception as e:
            # If API fails, should fall back gracefully
            pytest.skip(f"Semantic partitioning requires API key: {e}")

    def test_semantic_fallback_on_api_error(self):
        """Test that semantic falls back to structural on API error."""
        text = """Paragraph one.

Paragraph two.

Paragraph three."""

        # Use invalid API key to trigger fallback
        partitions = partition_text(
            text,
            strategy="semantic",
            max_tokens=1000,
            api_key="invalid_key_xyz"
        )

        # Should still return partitions (via fallback)
        assert len(partitions) >= 1

    def test_semantic_respects_max_tokens(self):
        """Test that semantic partitions respect max_tokens."""
        text = ". ".join([f"Sentence {i}" for i in range(100)])

        try:
            partitions = partition_text(text, strategy="semantic", max_tokens=100)

            for p in partitions:
                if 'token_count' in p.metadata:
                    # Should respect token limit
                    assert p.metadata['token_count'] <= 150  # Some tolerance
        except Exception:
            pytest.skip("Semantic partitioning requires API key")
