"""System prompt templates for RLM."""


def build_system_prompt(context_size: int, depth: int = 0) -> str:
    """
    Build system prompt for RLM.

    Args:
        context_size: Size of context in characters
        depth: Current recursion depth

    Returns:
        System prompt string
    """
    # Minimal prompt (paper-style)
    prompt = f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.

Available in environment:
- context: str (the document to analyze)
- query: str (the question: "{"{"}query{"}"}")
- recursive_llm(sub_query, sub_context) -> str (recursively process sub-context)
- re: already imported regex module (use re.findall, re.search, etc.)

Write Python code to answer the query. The last expression or print() output will be shown to you.

Examples:
- print(context[:100])  # See first 100 chars
- errors = re.findall(r'ERROR', context)  # Find all ERROR
- count = len(errors); print(count)  # Count and show

When you have the answer, use FINAL("answer") - this is NOT a function, just write it as text.

Depth: {depth}"""

    return prompt


def build_locodiff_prompt(context_size: int, depth: int = 0) -> str:
    """
    Build system prompt optimized for LoCoDiff file reconstruction from git history.

    Args:
        context_size: Size of context in characters
        depth: Current recursion depth

    Returns:
        System prompt string
    """
    
    return f"""Reconstruct a file from git history.

AVAILABLE:
- context: git log output ({context_size:,} chars) 
- llm_query(prompt): call sub-LLM
- strip_markdown(s): clean output
- FINAL_VAR(x): return result

DO THIS:
```repl
r = llm_query(f\"\"\"Reconstruct the file from git log. Output ONLY the file, no markdown.
{{context}}\"\"\")
result = strip_markdown(r)
```

```repl
FINAL_VAR(result)
```

Depth: {depth}"""


def build_user_prompt(query: str) -> str:
    """
    Build user prompt.

    Args:
        query: User's question

    Returns:
        User prompt string
    """
    return query


def build_iteration_prompt(query: str, iteration: int = 0) -> str:
    """
    Build iteration-aware user prompt for LoCoDiff.

    Args:
        query: User's question
        iteration: Current iteration number

    Returns:
        User prompt string
    """
    if iteration == 0:
        return f"Task: {query}\n\nUse the REPL to call llm_query with the full context. Execute the strategy shown above."
    else:
        return "Continue. If you have the result, use FINAL_VAR(result) to return it."
