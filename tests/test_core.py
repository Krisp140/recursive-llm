"""Tests for core RLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rlm import RLM, MaxIterationsError, MaxDepthError


class MockResponse:
    """Mock LLM response."""
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]


@pytest.fixture
def mock_litellm():
    """Mock litellm.acompletion."""
    with patch('rlm.core.litellm.acompletion') as mock:
        yield mock


@pytest.mark.asyncio
async def test_simple_completion(mock_litellm):
    """Test simple completion with FINAL."""
    mock_litellm.return_value = MockResponse('FINAL("The answer")')

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("What is the answer?", "Some context")

    assert result == "The answer"
    assert mock_litellm.called


@pytest.mark.asyncio
async def test_multi_step_completion(mock_litellm):
    """Test multi-step completion."""
    responses = [
        MockResponse('x = context[:10]\nprint(x)'),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Hello World Test")

    assert result == "Done"
    assert mock_litellm.call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_error(mock_litellm):
    """Test max iterations exceeded."""
    mock_litellm.return_value = MockResponse('x = 1')  # Never returns FINAL

    rlm = RLM(model="test-model", max_iterations=3)

    with pytest.raises(MaxIterationsError):
        await rlm.acompletion("Test", "Context")


@pytest.mark.asyncio
async def test_max_depth_error(mock_litellm):
    """Test max depth exceeded."""
    rlm = RLM(model="test-model", max_depth=2, _current_depth=2)

    with pytest.raises(MaxDepthError):
        await rlm.acompletion("Test", "Context")


@pytest.mark.asyncio
async def test_final_var(mock_litellm):
    """Test FINAL_VAR extraction."""
    responses = [
        MockResponse('result = "Test Answer"\nprint(result)'),
        MockResponse('FINAL_VAR(result)'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Context")

    assert result == "Test Answer"


@pytest.mark.asyncio
async def test_repl_error_handling(mock_litellm):
    """Test REPL error handling."""
    responses = [
        MockResponse('x = 1 / 0'),  # This will cause error
        MockResponse('FINAL("Recovered")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Context")

    assert result == "Recovered"


@pytest.mark.asyncio
async def test_context_operations(mock_litellm):
    """Test context operations in REPL."""
    responses = [
        MockResponse('first_10 = context[:10]'),
        MockResponse('FINAL_VAR(first_10)'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Get first 10 chars", "Hello World Example")

    assert result == "Hello Worl"


def test_sync_completion():
    """Test sync wrapper."""
    with patch('rlm.core.litellm.acompletion') as mock:
        mock.return_value = MockResponse('FINAL("Sync result")')

        rlm = RLM(model="test-model")
        result = rlm.completion("Test", "Context")

        assert result == "Sync result"


@pytest.mark.asyncio
async def test_two_models(mock_litellm):
    """Test using different models for root and recursive."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(
        model="expensive-model",
        recursive_model="cheap-model",
        _current_depth=0
    )

    await rlm.acompletion("Test", "Context")

    # First call should use expensive model
    call_args = mock_litellm.call_args_list[0]
    assert call_args[1]['model'] == "expensive-model"


@pytest.mark.asyncio
async def test_stats(mock_litellm):
    """Test statistics tracking."""
    responses = [
        MockResponse('x = 1'),
        MockResponse('y = 2'),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    await rlm.acompletion("Test", "Context")

    stats = rlm.stats
    assert stats['llm_calls'] == 3
    assert stats['iterations'] == 3
    assert stats['depth'] == 0


@pytest.mark.asyncio
async def test_api_base_and_key(mock_litellm):
    """Test API base and key passing."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(
        model="test-model",
        api_base="http://localhost:8000",
        api_key="test-key"
    )

    await rlm.acompletion("Test", "Context")

    call_kwargs = mock_litellm.call_args[1]
    assert call_kwargs['api_base'] == "http://localhost:8000"
    assert call_kwargs['api_key'] == "test-key"


@pytest.mark.asyncio
async def test_parallel_subqueries(mock_litellm):
    """Test parallel sub-query execution produces same results as sequential."""
    # Mock responses for partitioned execution
    # First 3 calls: child RLM for partitions (with FINAL())
    # Fourth call: stitching call (direct text, no FINAL())
    child_responses = [
        MockResponse('FINAL("Answer from partition 1")'),
        MockResponse('FINAL("Answer from partition 2")'),
        MockResponse('FINAL("Answer from partition 3")'),
        MockResponse('Stitched final answer'),  # Stitching returns plain text
    ]
    mock_litellm.side_effect = child_responses

    # Create long context to trigger partitioning
    long_context = "Section 1: " + "content " * 1000 + "\n" + \
                   "Section 2: " + "content " * 1000 + "\n" + \
                   "Section 3: " + "content " * 1000

    # Test with parallel execution
    rlm_parallel = RLM(
        model="test-model",
        partition_strategy="token",
        max_partition_tokens=1000,
        parallel_subqueries=True,
        max_parallel_subqueries=3
    )

    result = await rlm_parallel.acompletion("Test query", long_context)

    assert result == "Stitched final answer"
    # Should have 4 LLM calls: 3 child calls + 1 stitching call
    assert mock_litellm.call_count == 4


@pytest.mark.asyncio
async def test_sequential_vs_parallel_same_results(mock_litellm):
    """Test that sequential and parallel execution produce the same results."""
    long_context = "Part 1 content " * 500 + "\n" + "Part 2 content " * 500

    # Sequential execution
    sequential_responses = [
        MockResponse('FINAL("Answer 1")'),
        MockResponse('FINAL("Answer 2")'),
        MockResponse('Combined answer'),  # Stitching call
    ]
    mock_litellm.side_effect = sequential_responses

    rlm_sequential = RLM(
        model="test-model",
        partition_strategy="token",
        max_partition_tokens=1000,
        parallel_subqueries=False,
        max_parallel_subqueries=2
    )

    result_sequential = await rlm_sequential.acompletion("Test", long_context)

    # Parallel execution
    parallel_responses = [
        MockResponse('FINAL("Answer 1")'),
        MockResponse('FINAL("Answer 2")'),
        MockResponse('Combined answer'),  # Stitching call
    ]
    mock_litellm.side_effect = parallel_responses

    rlm_parallel = RLM(
        model="test-model",
        partition_strategy="token",
        max_partition_tokens=1000,
        parallel_subqueries=True,
        max_parallel_subqueries=2
    )

    result_parallel = await rlm_parallel.acompletion("Test", long_context)

    # Both should produce the same result
    assert result_sequential == result_parallel
    assert result_parallel == "Combined answer"


@pytest.mark.asyncio
async def test_parallel_error_handling(mock_litellm):
    """Test error handling in parallel execution."""
    # First partition succeeds, second fails, third succeeds
    async def side_effect_with_error(*args, **kwargs):
        """Side effect that raises error on second call."""
        if not hasattr(side_effect_with_error, 'call_count'):
            side_effect_with_error.call_count = 0
        side_effect_with_error.call_count += 1

        if side_effect_with_error.call_count == 2:
            raise Exception("Simulated partition error")
        elif side_effect_with_error.call_count == 4:
            # Stitching call (returns plain text)
            return MockResponse('Stitched with errors')
        else:
            return MockResponse('FINAL("Success")')

    mock_litellm.side_effect = side_effect_with_error

    long_context = "A " * 1000 + "B " * 1000 + "C " * 1000

    rlm = RLM(
        model="test-model",
        partition_strategy="token",
        max_partition_tokens=1000,
        parallel_subqueries=True,
        max_parallel_subqueries=3
    )

    # Should not raise, but handle error gracefully
    result = await rlm.acompletion("Test", long_context)

    # Final result should still be returned
    assert result == "Stitched with errors"


@pytest.mark.asyncio
async def test_parallel_stats_tracking(mock_litellm):
    """Test that stats are tracked correctly in parallel mode."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    long_context = "Content " * 2000

    rlm = RLM(
        model="test-model",
        partition_strategy="token",
        max_partition_tokens=1000,
        parallel_subqueries=True,
        max_parallel_subqueries=3
    )

    await rlm.acompletion("Test", long_context)

    stats = rlm.stats
    # Should track child LLM calls
    assert stats['child_llm_calls'] > 0
    assert stats['depth'] == 0
