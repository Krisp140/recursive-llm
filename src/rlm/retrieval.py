"""Retrieval methods for selecting relevant partitions."""

import re
from typing import List, Optional, Any
import numpy as np

from .partitions import Partition


class PartitionRetriever:
    """Retrieves relevant partitions based on query."""

    def __init__(
        self,
        method: str = "unfiltered",
        top_k: int = 5,
        **kwargs: Any
    ):
        """
        Initialize partition retriever.

        Args:
            method: Retrieval method ("regex", "embedding", "unfiltered")
            top_k: Maximum number of partitions to retrieve
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.top_k = top_k
        self.kwargs = kwargs

    def retrieve(
        self,
        query: str,
        partitions: List[Partition]
    ) -> List[Partition]:
        """
        Retrieve relevant partitions for the query.

        Args:
            query: Query string
            partitions: List of all partitions

        Returns:
            List of selected partitions (up to top_k)
        """
        if self.method == "regex":
            return self._retrieve_regex(query, partitions)
        elif self.method == "embedding":
            return self._retrieve_embedding(query, partitions)
        elif self.method == "unfiltered":
            return self._retrieve_unfiltered(query, partitions)
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

    def _retrieve_regex(
        self,
        query: str,
        partitions: List[Partition]
    ) -> List[Partition]:
        """
        Regex/keyword-based retrieval.

        Scores partitions by keyword overlap and regex matches.

        Args:
            query: Query string
            partitions: List of all partitions

        Returns:
            Top-k partitions by match score
        """
        # Extract keywords from query (simple tokenization)
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'what', 'where', 'when', 'how', 'why', 'which', 'who'}

        query_words = query.lower().split()
        keywords = [w for w in query_words if w not in stopwords and len(w) > 2]

        # Score each partition
        scored_partitions: List[tuple[float, Partition]] = []

        for partition in partitions:
            score = 0.0
            text_lower = partition.text.lower()

            # Count keyword matches
            for keyword in keywords:
                # Exact matches (word boundaries)
                matches = re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower)
                score += len(matches)

                # Partial matches (substring)
                if keyword in text_lower and len(matches) == 0:
                    score += 0.5

            # Bonus for early position in document
            position_bonus = max(0, 1.0 - (partition.index / len(partitions)))
            score += position_bonus * 0.1

            scored_partitions.append((score, partition))

        # Sort by score (descending)
        scored_partitions.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        top_partitions = [p for score, p in scored_partitions[:self.top_k]]

        # If no partitions scored > 0, return first k partitions as fallback
        if not top_partitions or all(score == 0 for score, _ in scored_partitions[:self.top_k]):
            return partitions[:self.top_k]

        return top_partitions

    def _retrieve_embedding(
        self,
        query: str,
        partitions: List[Partition]
    ) -> List[Partition]:
        """
        Embedding-based retrieval.

        Scores partitions by cosine similarity with query embedding.

        Args:
            query: Query string
            partitions: List of all partitions

        Returns:
            Top-k partitions by similarity
        """
        import openai

        # Get API key from kwargs or environment
        api_key = self.kwargs.get('api_key')
        if api_key:
            openai.api_key = api_key

        # Check if partitions have embeddings
        # (they will if created with semantic partitioning)
        partitions_with_embeddings = [
            p for p in partitions if 'embedding' in p.metadata
        ]

        # If no partitions have embeddings, compute them now
        if not partitions_with_embeddings:
            partitions = self._compute_partition_embeddings(partitions, openai)
        else:
            # Some partitions might not have embeddings (mixed strategies)
            # Compute for those that don't
            for p in partitions:
                if 'embedding' not in p.metadata:
                    try:
                        response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=p.text[:8000]  # Truncate if needed
                        )
                        p.metadata['embedding'] = response.data[0].embedding
                    except Exception:
                        # If embedding fails, use zero vector
                        p.metadata['embedding'] = [0.0] * 1536

        # Get query embedding
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding)
        except Exception as e:
            # If query embedding fails, fall back to regex
            print(f"Warning: Query embedding failed ({e}), falling back to regex retrieval")
            return self._retrieve_regex(query, partitions)

        # Score each partition by cosine similarity
        scored_partitions: List[tuple[float, Partition]] = []

        for partition in partitions:
            partition_embedding = np.array(partition.metadata['embedding'])

            # Compute cosine similarity
            similarity = np.dot(query_embedding, partition_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(partition_embedding)
            )

            scored_partitions.append((float(similarity), partition))

        # Sort by similarity (descending)
        scored_partitions.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [p for score, p in scored_partitions[:self.top_k]]

    def _retrieve_unfiltered(
        self,
        query: str,
        partitions: List[Partition]
    ) -> List[Partition]:
        """
        Unfiltered retrieval (baseline).

        Returns first k partitions without filtering.

        Args:
            query: Query string (unused)
            partitions: List of all partitions

        Returns:
            First top-k partitions
        """
        return partitions[:self.top_k]

    def _compute_partition_embeddings(
        self,
        partitions: List[Partition],
        openai_client: Any
    ) -> List[Partition]:
        """
        Compute embeddings for partitions that don't have them.

        Args:
            partitions: List of partitions
            openai_client: OpenAI client

        Returns:
            Partitions with embeddings added to metadata
        """
        # Batch embed partitions
        texts = [p.text[:8000] for p in partitions]  # Truncate if needed

        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            # Add embeddings to partition metadata
            for partition, embedding_obj in zip(partitions, response.data):
                partition.metadata['embedding'] = embedding_obj.embedding

        except Exception as e:
            print(f"Warning: Batch embedding failed ({e}), using zero vectors")
            # Use zero vectors as fallback
            for partition in partitions:
                partition.metadata['embedding'] = [0.0] * 1536

        return partitions
