"""Parse FINAL() and FINAL_VAR() statements from LLM responses."""

import re
from typing import Optional, Dict, Any


def extract_final(response: str) -> Optional[str]:
    """
    Extract answer from FINAL() statement.

    Args:
        response: LLM response text

    Returns:
        Extracted answer or None if not found
    """
    # Try triple quotes first (best for long multiline strings)
    patterns = [
        r'FINAL\s*\(\s*"""(.*)"""',  # FINAL("""answer""") - triple double quotes
        r"FINAL\s*\(\s*'''(.*)'''",  # FINAL('''answer''') - triple single quotes
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: Manual parsing for complex/long strings
    # Find FINAL( and then match quotes manually
    final_match = re.search(r'FINAL\s*\(\s*(["\'])', response)
    if final_match:
        quote_char = final_match.group(1)
        start_pos = final_match.end() - 1  # Position of opening quote

        # Find matching closing quote (handle escapes)
        i = start_pos + 1
        result = []
        while i < len(response):
            char = response[i]
            if char == '\\' and i + 1 < len(response):
                # Escaped character
                next_char = response[i + 1]
                if next_char == 'n':
                    result.append('\n')
                elif next_char == 't':
                    result.append('\t')
                elif next_char == 'r':
                    result.append('\r')
                elif next_char == '\\':
                    result.append('\\')
                elif next_char == quote_char:
                    result.append(quote_char)
                else:
                    result.append(next_char)
                i += 2
            elif char == quote_char:
                # Found closing quote
                return ''.join(result)
            else:
                result.append(char)
                i += 1

    return None


def extract_final_var(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Extract answer from FINAL_VAR() statement.

    Args:
        response: LLM response text
        env: REPL environment with variables

    Returns:
        Variable value as string or None if not found
    """
    # Look for FINAL_VAR(var_name)
    match = re.search(r'FINAL_VAR\s*\(\s*(\w+)\s*\)', response)
    if not match:
        return None

    var_name = match.group(1)

    # Get variable from environment
    if var_name in env:
        value = env[var_name]
        return str(value)

    return None


def is_final(response: str) -> bool:
    """
    Check if response contains FINAL() or FINAL_VAR().

    Args:
        response: LLM response text

    Returns:
        True if response contains final statement
    """
    return 'FINAL(' in response or 'FINAL_VAR(' in response


def parse_response(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Parse response for any final statement.

    Args:
        response: LLM response text
        env: REPL environment

    Returns:
        Final answer or None
    """
    # Try FINAL() first
    answer = extract_final(response)
    if answer is not None:
        return answer

    # Try FINAL_VAR()
    answer = extract_final_var(response, env)
    if answer is not None:
        return answer

    return None
