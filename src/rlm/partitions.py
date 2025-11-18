"""Context partitioning strategies for RLM."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import tiktoken


@dataclass
class Partition:
    """A partition of the original context.

    Attributes:
        text: The text content of this partition
        index: Position in the partition sequence (0-indexed)
        start_char: Starting character index in original context
        end_char: Ending character index in original context
        metadata: Optional metadata (embeddings, headings, etc.)
    """
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return length of partition text."""
        return len(self.text)


PartitionStrategy = Literal["token", "structural", "semantic", "learned"]


def partition_text(
    text: str,
    strategy: PartitionStrategy = "token",
    max_tokens: int = 4000,
    overlap_tokens: int = 200,
    model_name: str = "gpt-4",
    **kwargs: Any
) -> List[Partition]:
    """
    Partition text into chunks using the specified strategy.

    Args:
        text: Text to partition
        strategy: Partitioning strategy to use
        max_tokens: Maximum tokens per partition
        overlap_tokens: Number of tokens to overlap between partitions
        model_name: Model name for tokenization
        **kwargs: Additional strategy-specific parameters

    Returns:
        List of Partition objects

    Examples:
        # Basic token-based partitioning
        partitions = partition_text(long_doc, strategy="token", max_tokens=2000)

        # With overlap for continuity
        partitions = partition_text(long_doc, strategy="token",
                                   max_tokens=2000, overlap_tokens=100)
    """
    if strategy == "token":
        return _partition_token(text, max_tokens, overlap_tokens, model_name)
    elif strategy == "structural":
        return _partition_structural(text, max_tokens, model_name, **kwargs)
    elif strategy == "semantic":
        return _partition_semantic(text, max_tokens, model_name, **kwargs)
    elif strategy == "learned":
        raise NotImplementedError("Learned partitioning not yet implemented (Phase 5)")
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")


def _partition_token(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    model_name: str
) -> List[Partition]:
    """
    Token-based partitioning (baseline strategy).

    Splits text into chunks of approximately max_tokens, with overlap
    between consecutive chunks for continuity.

    Args:
        text: Text to partition
        max_tokens: Maximum tokens per partition
        overlap_tokens: Overlap between partitions
        model_name: Model name for tokenizer

    Returns:
        List of Partition objects
    """
    # Get tokenizer
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    # Tokenize entire text
    tokens = encoding.encode(text)
    total_tokens = len(tokens)

    # If text fits in one partition, return it as-is
    if total_tokens <= max_tokens:
        return [Partition(
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
            metadata={"token_count": total_tokens}
        )]

    # Calculate stride (step size between partition starts)
    stride = max_tokens - overlap_tokens
    if stride <= 0:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be less than max_tokens ({max_tokens})"
        )

    partitions: List[Partition] = []
    partition_index = 0
    start_token_idx = 0

    while start_token_idx < total_tokens:
        # Calculate end token index for this partition
        end_token_idx = min(start_token_idx + max_tokens, total_tokens)

        # Extract token slice
        partition_tokens = tokens[start_token_idx:end_token_idx]

        # Decode back to text
        partition_text = encoding.decode(partition_tokens)

        # Find character positions in original text
        # For simplicity, we'll use approximate character positions
        # by decoding prefixes (more accurate than simple ratio)
        if start_token_idx == 0:
            start_char = 0
        else:
            # Decode up to start position to find character index
            prefix_text = encoding.decode(tokens[:start_token_idx])
            start_char = len(prefix_text)

        end_char = start_char + len(partition_text)

        # Create partition
        partition = Partition(
            text=partition_text,
            index=partition_index,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "token_count": len(partition_tokens),
                "start_token_idx": start_token_idx,
                "end_token_idx": end_token_idx,
            }
        )
        partitions.append(partition)

        # Move to next partition
        partition_index += 1
        start_token_idx += stride

        # Prevent infinite loop edge case
        if start_token_idx >= total_tokens:
            break

    return partitions


def _partition_structural(
    text: str,
    max_tokens: int,
    model_name: str,
    **kwargs: Any
) -> List[Partition]:
    """
    Structural partitioning strategy.

    Splits text on natural boundaries like paragraphs and headings,
    while respecting max_tokens constraints.

    Args:
        text: Text to partition
        max_tokens: Maximum tokens per partition
        model_name: Model name for tokenizer
        **kwargs: Additional options

    Returns:
        List of Partition objects with structural metadata
    """
    # Get tokenizer
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Split into structural units (paragraphs)
    # First, normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split on double newlines (paragraph boundaries)
    raw_units = text.split('\n\n')

    # Process each unit
    units: List[Dict[str, Any]] = []
    char_offset = 0

    for raw_unit in raw_units:
        if not raw_unit.strip():
            # Skip empty units, but track offset
            char_offset += len(raw_unit) + 2  # +2 for \n\n
            continue

        # Detect if this is a heading
        is_heading = False
        heading_text = None
        lines = raw_unit.split('\n')

        # Check for markdown-style headings (# Heading)
        if lines[0].strip().startswith('#'):
            is_heading = True
            heading_text = lines[0].strip().lstrip('#').strip()

        # Check for short lines that look like headings (all caps, short, etc.)
        elif len(lines[0]) < 100 and lines[0].isupper():
            is_heading = True
            heading_text = lines[0].strip()

        # Count tokens
        token_count = len(encoding.encode(raw_unit))

        units.append({
            'text': raw_unit,
            'start_char': char_offset,
            'end_char': char_offset + len(raw_unit),
            'token_count': token_count,
            'is_heading': is_heading,
            'heading_text': heading_text,
        })

        char_offset += len(raw_unit) + 2  # +2 for \n\n separator

    # Now merge units into partitions respecting max_tokens
    partitions: List[Partition] = []
    current_partition_units: List[Dict[str, Any]] = []
    current_token_count = 0
    partition_index = 0

    for unit in units:
        unit_tokens = unit['token_count']

        # If single unit exceeds max_tokens, split it with token-based partitioning
        if unit_tokens > max_tokens:
            # First, flush current partition if any
            if current_partition_units:
                partitions.append(_create_partition_from_units(
                    current_partition_units, partition_index
                ))
                partition_index += 1
                current_partition_units = []
                current_token_count = 0

            # Split the large unit using token-based partitioning
            large_unit_partitions = _partition_token(
                text=unit['text'],
                max_tokens=max_tokens,
                overlap_tokens=0,  # No overlap for structural units
                model_name=model_name
            )

            # Add each sub-partition
            for sub_partition in large_unit_partitions:
                partitions.append(Partition(
                    text=sub_partition.text,
                    index=partition_index,
                    start_char=unit['start_char'] + sub_partition.start_char,
                    end_char=unit['start_char'] + sub_partition.end_char,
                    metadata={
                        **sub_partition.metadata,
                        'kind': 'oversized_unit_split',
                    }
                ))
                partition_index += 1

            continue

        # Check if adding this unit would exceed max_tokens
        if current_token_count + unit_tokens > max_tokens and current_partition_units:
            # Create partition from accumulated units
            partitions.append(_create_partition_from_units(
                current_partition_units, partition_index
            ))
            partition_index += 1
            current_partition_units = []
            current_token_count = 0

        # Add unit to current partition
        current_partition_units.append(unit)
        current_token_count += unit_tokens

    # Don't forget the last partition
    if current_partition_units:
        partitions.append(_create_partition_from_units(
            current_partition_units, partition_index
        ))

    # Handle empty text case
    if not partitions:
        return [Partition(
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
            metadata={'kind': 'empty', 'token_count': 0}
        )]

    return partitions


def _create_partition_from_units(
    units: List[Dict[str, Any]],
    partition_index: int
) -> Partition:
    """
    Create a Partition from a list of structural units.

    Args:
        units: List of unit dictionaries
        partition_index: Index for this partition

    Returns:
        Partition object
    """
    # Reconstruct text with paragraph separators
    text = '\n\n'.join(unit['text'] for unit in units)

    # Calculate character range
    start_char = units[0]['start_char']
    end_char = units[-1]['end_char']

    # Calculate total tokens
    total_tokens = sum(unit['token_count'] for unit in units)

    # Gather metadata
    has_heading = any(unit['is_heading'] for unit in units)
    headings = [unit['heading_text'] for unit in units if unit['is_heading']]

    metadata = {
        'kind': 'structural',
        'token_count': total_tokens,
        'num_units': len(units),
        'has_heading': has_heading,
    }

    if headings:
        metadata['headings'] = headings

    return Partition(
        text=text,
        index=partition_index,
        start_char=start_char,
        end_char=end_char,
        metadata=metadata
    )


def _partition_semantic(
    text: str,
    max_tokens: int,
    model_name: str,
    **kwargs: Any
) -> List[Partition]:
    """
    Semantic partitioning strategy.

    Splits text based on topic shifts detected via embedding similarity.

    Args:
        text: Text to partition
        max_tokens: Maximum tokens per partition
        model_name: Model name for tokenizer
        **kwargs: Additional options (api_key, similarity_threshold, etc.)

    Returns:
        List of Partition objects with embedding metadata
    """
    import openai
    import numpy as np

    # Get API key from kwargs or environment
    api_key = kwargs.get('api_key')
    if api_key:
        openai.api_key = api_key

    # Get similarity threshold
    similarity_threshold = kwargs.get('similarity_threshold', 0.7)

    # Get tokenizer
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Split into sentences/spans
    # For simplicity, split on sentence boundaries (., !, ?)
    # followed by whitespace
    import re
    sentence_pattern = r'[.!?]+[\s\n]+'
    sentences = re.split(sentence_pattern, text)

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [Partition(
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
            metadata={'kind': 'empty', 'token_count': 0}
        )]

    # For very long documents, we can't embed every sentence
    # Group sentences into spans of ~100-200 tokens for embedding
    spans: List[Dict[str, Any]] = []
    current_span_sentences: List[str] = []
    current_span_tokens = 0
    char_offset = 0
    target_span_tokens = 150

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if current_span_tokens + sentence_tokens > target_span_tokens and current_span_sentences:
            # Save current span
            span_text = ' '.join(current_span_sentences)
            spans.append({
                'text': span_text,
                'start_char': char_offset,
                'end_char': char_offset + len(span_text),
                'token_count': current_span_tokens,
                'sentences': current_span_sentences.copy()
            })
            char_offset += len(span_text) + 2  # Approximate separator
            current_span_sentences = []
            current_span_tokens = 0

        current_span_sentences.append(sentence)
        current_span_tokens += sentence_tokens

    # Don't forget last span
    if current_span_sentences:
        span_text = ' '.join(current_span_sentences)
        spans.append({
            'text': span_text,
            'start_char': char_offset,
            'end_char': char_offset + len(span_text),
            'token_count': current_span_tokens,
            'sentences': current_span_sentences
        })

    # If only one span, return it as single partition
    if len(spans) == 1:
        return [Partition(
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
            metadata={'kind': 'semantic_single', 'token_count': spans[0]['token_count']}
        )]

    # Get embeddings for each span
    span_texts = [span['text'] for span in spans]

    try:
        # Use OpenAI embeddings API
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=span_texts
        )
        embeddings = [item.embedding for item in response.data]

        # Store embeddings in spans
        for span, embedding in zip(spans, embeddings):
            span['embedding'] = np.array(embedding)

    except Exception as e:
        # If embeddings fail, fall back to structural partitioning
        print(f"Warning: Embedding API failed ({e}), falling back to structural partitioning")
        return _partition_structural(text, max_tokens, model_name, **kwargs)

    # Compute cosine similarity between adjacent spans
    similarities: List[float] = []
    for i in range(len(embeddings) - 1):
        emb1 = np.array(embeddings[i])
        emb2 = np.array(embeddings[i + 1])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(float(similarity))

    # Find partition boundaries where similarity drops below threshold
    partition_boundaries = [0]  # Start of first partition
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            partition_boundaries.append(i + 1)  # Start new partition after this span

    partition_boundaries.append(len(spans))  # End of last partition

    # Create partitions from span groups
    partitions: List[Partition] = []

    for idx in range(len(partition_boundaries) - 1):
        start_idx = partition_boundaries[idx]
        end_idx = partition_boundaries[idx + 1]

        partition_spans = spans[start_idx:end_idx]

        # Merge spans into partition, respecting max_tokens
        current_partition_spans: List[Dict[str, Any]] = []
        current_tokens = 0

        for span in partition_spans:
            # If adding this span exceeds max_tokens, create a partition
            if current_tokens + span['token_count'] > max_tokens and current_partition_spans:
                partitions.append(_create_semantic_partition(
                    current_partition_spans, len(partitions)
                ))
                current_partition_spans = []
                current_tokens = 0

            current_partition_spans.append(span)
            current_tokens += span['token_count']

        # Add remaining spans as partition
        if current_partition_spans:
            partitions.append(_create_semantic_partition(
                current_partition_spans, len(partitions)
            ))

    return partitions if partitions else [Partition(
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
        metadata={'kind': 'semantic_fallback', 'token_count': count_tokens(text, model_name)}
    )]


def _create_semantic_partition(
    spans: List[Dict[str, Any]],
    partition_index: int
) -> Partition:
    """
    Create a Partition from semantic spans.

    Args:
        spans: List of span dictionaries
        partition_index: Index for this partition

    Returns:
        Partition object
    """
    # Reconstruct text
    text = ' '.join(span['text'] for span in spans)

    # Character range
    start_char = spans[0]['start_char']
    end_char = spans[-1]['end_char']

    # Total tokens
    total_tokens = sum(span['token_count'] for span in spans)

    # Average embedding (optional, for retrieval later)
    if 'embedding' in spans[0]:
        import numpy as np
        embeddings = [span['embedding'] for span in spans if 'embedding' in span]
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        metadata = {
            'kind': 'semantic',
            'token_count': total_tokens,
            'num_spans': len(spans),
            'embedding': avg_embedding,
        }
    else:
        metadata = {
            'kind': 'semantic',
            'token_count': total_tokens,
            'num_spans': len(spans),
        }

    return Partition(
        text=text,
        index=partition_index,
        start_char=start_char,
        end_char=end_char,
        metadata=metadata
    )


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))
