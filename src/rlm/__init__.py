"""Recursive Language Models for unbounded context processing."""

from .core import RLM, RLMError, MaxIterationsError, MaxDepthError
from .repl import REPLError
from .partitions import Partition, partition_text, count_tokens
from .retrieval import PartitionRetriever

__version__ = "0.1.0"

__all__ = [
    "RLM",
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "REPLError",
    "Partition",
    "partition_text",
    "count_tokens",
    "PartitionRetriever",
]
