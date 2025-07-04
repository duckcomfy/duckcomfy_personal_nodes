import re
from typing import Dict, List, Tuple, Any
from comfy.sd import CLIP
from comfy.sd1_clip import SDTokenizer


def _parse_prompts(text: str) -> List[str]:
    """
    Parse the input text and split by existing BREAK markers.
    Returns a list of individual prompts.
    """
    # Split by BREAK but preserve the structure
    parts = text.split(' BREAK ')
    if len(parts) == 1:
        parts = text.split('BREAK')
    return parts


def get_clip_l_token_info(clip, text: str) -> str:
    """
    Return the input text with BREAK markers inserted where CLIP L chunks end/begin.

    Args:
        clip: The CLIP model instance with tokenize method
        text: The input text to analyze

    Returns:
        String with BREAK markers inserted at chunk boundaries
    """
    # Get the CLIP L tokenizer
    from comfy.sd1_clip import SDTokenizer
    tokenizers = [t for t in clip.tokenizer.__dict__.values() if isinstance(t, SDTokenizer)]
    clip_l_tokenizer = None
    for tokenizer in tokenizers:
        if tokenizer.embedding_key == "clip_l":
            clip_l_tokenizer = tokenizer
            break

    if not clip_l_tokenizer:
        return text  # Return original text if no CLIP L tokenizer found

    # Parse prompts to handle existing BREAK syntax
    prompts = _parse_prompts(text)
    result_parts = []

    for prompt_idx, prompt in enumerate(prompts):
        if len(prompt) == 0:
            result_parts.append(prompt)
            continue

        # Tokenize this prompt
        tokenizer_results = clip.tokenize(prompt)

        if "l" not in tokenizer_results:
            result_parts.append(prompt)
            continue

        chunks = tokenizer_results["l"]

        if len(chunks) <= 1:
            # Single chunk, no need for BREAK markers
            result_parts.append(prompt)
        else:
            # Multiple chunks - find boundaries using clean tokenization

            # Step 1: Create clean version without weights and comments
            clean_prompt = _create_clean_prompt(prompt)

            # Step 2: Tokenize clean version to find boundaries
            clean_tokenizer_results = clip.tokenize(clean_prompt)
            if "l" not in clean_tokenizer_results:
                result_parts.append(prompt)
                continue

            clean_chunks = clean_tokenizer_results["l"]

            # Step 3: Find boundary positions by untokenizing each chunk
            boundaries = []

            # Process each chunk boundary
            for chunk_idx in range(len(clean_chunks) - 1):
                # Get tokens up to and including this chunk
                tokens_so_far = []
                for i in range(chunk_idx + 1):
                    tokens_so_far.extend(clean_chunks[i])

                # Untokenize to see what text this represents
                token_ids = [t[0] for t in tokens_so_far]
                decoded_text = clip_l_tokenizer.tokenizer.decode(token_ids)
                decoded_text = decoded_text.replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip()

                # Now map this back to the original prompt
                # The decoded text shows us what content appears up to the boundary
                boundary_pos = _map_clean_boundary_to_original(prompt, clean_prompt, decoded_text)

                if boundary_pos > 0:
                    boundaries.append(boundary_pos)

            # Step 4: Insert BREAKs at boundary positions
            result = ""
            text_pos = 0

            for boundary_pos in boundaries:
                # Check if this boundary falls within a weight expression
                weight_split = _check_weight_boundary(prompt, text_pos, boundary_pos)

                if weight_split:
                    # Add text up to the weight
                    result += prompt[text_pos:weight_split['weight_start']]

                    # Add the split weight with BREAK
                    result += weight_split['first_weight']

                    # Insert BREAK with proper spacing
                    if result and not result[-1].isspace():
                        result += " "
                    result += "BREAK"
                    next_char = weight_split['second_weight'][0] if weight_split['second_weight'] else ""
                    if next_char and not next_char.isspace() and next_char != ",":
                        result += " "

                    result += weight_split['second_weight']

                    # Continue from after the weight
                    text_pos = weight_split['weight_end']
                else:
                    # Normal boundary - add text up to boundary
                    result += prompt[text_pos:boundary_pos]

                    # Insert BREAK with proper spacing
                    if result and not result[-1].isspace():
                        result += " "

                    result += "BREAK"

                    # Add space after BREAK if next char isn't already space or comma
                    next_char = prompt[boundary_pos] if boundary_pos < len(prompt) else ""
                    if next_char and not next_char.isspace() and next_char != ",":
                        result += " "

                    text_pos = boundary_pos

            # Add remaining text
            result += prompt[text_pos:]
            result_parts.append(result)

    # Join all prompt parts with BREAK
    return ' BREAK '.join(result_parts)


def _create_clean_prompt(prompt: str) -> str:
    """
    Create a clean version of the prompt without weights and comments.
    Removes (text:weight) but preserves escaped parentheses.
    """
    import re

    # Remove comment lines
    lines = prompt.split('\n')
    clean_lines = []
    for line in lines:
        if not line.strip().startswith('//'):
            clean_lines.append(line)
        else:
            # Keep empty line to preserve line structure
            clean_lines.append('')

    clean = '\n'.join(clean_lines)

    # Remove weight syntax: (text:number) but not escaped parentheses
    # This regex looks for ( not preceded by \, then captures until :number)
    # We need to be careful about nested parentheses

    # First, temporarily replace escaped parentheses
    clean = clean.replace('\\(', '\x00ESCAPED_OPEN\x00')
    clean = clean.replace('\\)', '\x00ESCAPED_CLOSE\x00')

    # Now remove weight syntax
    # Match (content:weight) and replace with just content
    pattern = r'\(([^():]+:[0-9.]+)\)'
    while re.search(pattern, clean):
        clean = re.sub(pattern, lambda m: m.group(1).split(':')[0], clean)

    # Also handle nested weights like ((text:0.5):0.8)
    pattern = r'\(([^()]+)\)'
    def replace_weight(match):
        content = match.group(1)
        if ':' in content and any(c.isdigit() for c in content.split(':')[-1]):
            # This is a weight, remove parentheses and weight
            return content.split(':')[0]
        else:
            # Not a weight, keep the parentheses
            return match.group(0)

    # Apply multiple times for nested cases
    for _ in range(5):  # Maximum nesting depth
        old_clean = clean
        clean = re.sub(pattern, replace_weight, clean)
        if clean == old_clean:
            break

    # Restore escaped parentheses
    clean = clean.replace('\x00ESCAPED_OPEN\x00', '\\(')
    clean = clean.replace('\x00ESCAPED_CLOSE\x00', '\\)')

    return clean


def _map_clean_boundary_to_original(original: str, clean: str, decoded_text: str) -> int:
    """
    Map a boundary position from the decoded/clean text back to the original.
    The decoded_text shows us what appears up to the chunk boundary.
    We need to find where this content ends in the original text.
    """
    # The decoded text might have different spacing than both original and clean
    # For example: "1girl" → "1 girl" in decoded, "hanging_light" → "hanging _ light"

    # Strategy: The decoded text represents the meaningful content that has been consumed
    # We need to find the position in the original text where this same amount of
    # meaningful content has been consumed, accounting for weight syntax differences

    # Remove all spacing from decoded to get pure content
    decoded_meaningful = ''.join(c for c in decoded_text if not c.isspace())
    if not decoded_meaningful:
        return 0

    # Universal approach: use the fallback algorithm for all cases
    # Count meaningful characters from decoded text and map to original

    # Use a more accurate character counting that matches tokenizer behavior
    # The tokenizer normalizes spacing, so we need to account for this difference

    # Based on analysis, the decoded text has ~6 fewer meaningful characters than expected
    # This suggests the tokenizer normalizes some characters differently
    # Let's use a target that accounts for this discrepancy

    target_meaningful = len(decoded_meaningful)
    meaningful_consumed = 0
    orig_pos = 0

    while orig_pos < len(original) and meaningful_consumed < target_meaningful:
        char = original[orig_pos]

        # Skip comments
        if orig_pos < len(original) - 1 and original[orig_pos:orig_pos+2] == '//':
            while orig_pos < len(original) and original[orig_pos] != '\n':
                orig_pos += 1
            if orig_pos < len(original):
                orig_pos += 1
            continue

        # Check for weight syntax
        if char == '(' and (orig_pos == 0 or original[orig_pos-1] != '\\'):
            paren_start = orig_pos
            paren_depth = 1
            temp_pos = orig_pos + 1

            while temp_pos < len(original) and paren_depth > 0:
                if original[temp_pos] == '(' and temp_pos > 0 and original[temp_pos-1] != '\\':
                    paren_depth += 1
                elif original[temp_pos] == ')' and temp_pos > 0 and original[temp_pos-1] != '\\':
                    paren_depth -= 1
                temp_pos += 1

            # Check if this is a weight
            if temp_pos > paren_start + 1:
                paren_content = original[paren_start+1:temp_pos-1]
                if ':' in paren_content and any(c.isdigit() for c in paren_content.split(':')[-1]):
                    # This is a weight - count only the content part
                    content_part = paren_content.split(':')[0]

                    # Special handling: if we're very close to the target and this weight
                    # could contain the boundary, be more precise about positioning
                    content_meaningful = sum(1 for c in content_part if not c.isspace())

                    if meaningful_consumed + content_meaningful >= target_meaningful:
                        # The boundary falls within this weight content
                        chars_needed = target_meaningful - meaningful_consumed

                        # Find the position within the weight content
                        weight_content_pos = 0
                        chars_in_content = 0
                        content_start = paren_start + 1

                        while weight_content_pos < len(content_part) and chars_in_content < chars_needed:
                            if not content_part[weight_content_pos].isspace():
                                chars_in_content += 1
                            weight_content_pos += 1

                        # Position after the consumed characters, accounting for token boundaries
                        result_pos = content_start + weight_content_pos

                        # If we're at a space, move to the next non-space character
                        # This ensures we position at token boundaries properly
                        while result_pos < len(original) and original[result_pos].isspace():
                            result_pos += 1

                        return result_pos

                    meaningful_consumed += content_meaningful
                    orig_pos = temp_pos
                    continue

        # Regular character - count all non-space characters
        if not char.isspace():
            meaningful_consumed += 1

        orig_pos += 1

    return orig_pos


def _check_weight_boundary(prompt: str, text_start: int, boundary_pos: int) -> dict:
    """
    Check if boundary falls within a weight expression and return split info.

    Returns dict with weight split info if boundary is within a weight, None otherwise.
    """
    import re

    # We need to look for weight expressions that might contain the boundary
    # The boundary might fall in the middle of a weight, so we need to search
    # in a reasonable area around the boundary position

    search_start = max(0, text_start)
    # Look for weights in an area around the boundary
    search_end = min(len(prompt), boundary_pos + 100)
    search_text = prompt[search_start:search_end]

    # Find weight expressions manually to handle nested parentheses
    # Look for pattern: (content:number) where content may contain escaped parentheses
    i = 0
    while i < len(search_text):
        if search_text[i] == '(' and (i == 0 or search_text[i-1] != '\\'):
            # Found opening paren, now find the matching closing paren
            paren_start = i
            paren_depth = 1
            j = i + 1

            while j < len(search_text) and paren_depth > 0:
                if search_text[j] == '(' and search_text[j-1] != '\\':
                    paren_depth += 1
                elif search_text[j] == ')' and search_text[j-1] != '\\':
                    paren_depth -= 1
                j += 1

            if paren_depth == 0:
                # Found matching closing paren
                paren_content = search_text[paren_start+1:j-1]

                # Check if this looks like a weight (ends with :number)
                weight_match = re.search(r':([0-9.]+)$', paren_content)
                if weight_match:
                    weight_value = weight_match.group(1)
                    content = paren_content[:weight_match.start()]

                    # Convert to absolute positions
                    weight_start_abs = search_start + paren_start
                    weight_end_abs = search_start + j
                    content_start_abs = weight_start_abs + 1  # +1 for opening paren
                    content_end_abs = weight_start_abs + 1 + len(content)  # content ends before :weight

                    # Check if boundary falls within this weight's content
                    if boundary_pos > content_start_abs and boundary_pos < content_end_abs:
                        # Boundary falls within the content of this weight
                        split_pos_in_content = boundary_pos - content_start_abs

                        first_part = content[:split_pos_in_content].rstrip()
                        second_part = content[split_pos_in_content:].lstrip()

                        # Only split if both parts have content
                        if first_part.strip() and second_part.strip():
                            return {
                                'weight_start': weight_start_abs,
                                'weight_end': weight_end_abs,
                                'first_weight': f'({first_part}:{weight_value})',
                                'second_weight': f'({second_part}:{weight_value})'
                            }

                i = j  # Continue after this weight
            else:
                i += 1
        else:
            i += 1

    return None



def get_special_tokens_map(clip: CLIP) -> dict[str, set[int]]:
    tokenizers: list[SDTokenizer] = [t for t in clip.tokenizer.__dict__.values() if isinstance(t, SDTokenizer)]
    special_tokens_map: dict[str, set[int]] = dict(
        (
            tokenizer.embedding_key.replace("clip_", ""),
            set([
                token
                for token in [tokenizer.start_token, tokenizer.end_token, tokenizer.pad_token]
                if isinstance(token, int)
            ]),
        )
        for tokenizer in tokenizers
    )
    return special_tokens_map

