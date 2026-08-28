"""Utilities for constrained function-calling generation with a small LLM."""

import json
import numpy as np
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from llm_sdk import Small_LLM_Model  # type: ignore


class PromptTest(BaseModel):
    """Validated prompt input used by test fixtures.

    Attributes:
        prompt: The user prompt to validate and process.
    """
    model_config = ConfigDict(extra='forbid', strict=True)
    prompt: StrictStr


class ParameterSchema(BaseModel):
    """Strict schema definition for a single function parameter.

    Attributes:
        type: The parameter type expected by the function schema.
        enum: Optional allowed values for the parameter.
    """
    model_config = ConfigDict(extra='forbid', strict=True)

    type: StrictStr
    enum: Optional[List[Any]] = None


class FunctionDefinition(BaseModel):
    """Function metadata and parameter schema consumed by the decoder.

    Attributes:
        name: The function name exposed to the model.
        description: Human-readable summary of the function.
        parameters: Mapping of parameter names to their schemas.
        returns: Schema information describing the function return value.
    """
    model_config = ConfigDict(extra='forbid', strict=True)

    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: Dict[str, str]


class FunctionCallResult(BaseModel):
    """Structured result produced after prompt processing.

    Attributes:
        prompt: The original prompt that was processed.
        name: The selected function name, or ``error`` when processing fails.
        parameters: The extracted argument values.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionCaller(BaseModel):
    """Constrained decoding engine for function-calling generation.

    The class loads the tokenizer vocabulary, restricts token selection to
    schema-valid outputs, and returns structured function call arguments.

    Attributes:
        function_definitions: Available functions the model may call.
        model: The small LLM wrapper used for encoding, decoding, and logits.
        vocab: Token-to-id vocabulary loaded from the model.
        reversed_vocab: Reverse lookup table from token id to token text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_definitions: List[FunctionDefinition]

    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    vocab: Dict[str, int] = Field(default_factory=dict)
    reversed_vocab: Dict[int, str] = Field(default_factory=dict)

    def _normalize_token_text(self, token_text: str) -> str:
        """Normalize token artifacts into regular text.

        Args:
            token_text: Raw token text from the vocabulary.

        Returns:
            The token text with tokenizer-specific markers converted.
        """
        return token_text.replace('Ġ', ' ').replace('Ċ', '\n')

    def _token_text(self, token_id: int) -> str:
        """Resolve a token id to normalized text.

        Args:
            token_id: The token identifier to look up.

        Returns:
            The normalized token string, or an empty string when missing.
        """
        raw_text = self.reversed_vocab.get(token_id, "")
        return self._normalize_token_text(raw_text)

    def _append_encoded_text(
        self,
        prompt_tokens: List[int],
        text: str,
    ) -> None:
        """Append encoded text to an existing token buffer.

        Args:
            prompt_tokens: Mutable token buffer to extend.
            text: Text to encode and append.
        """
        encoded_text = self.model.encode(text).flatten().tolist()
        prompt_tokens.extend(encoded_text)

    def model_post_init(self, __context: Any) -> None:
        """Load the model vocabulary and build reverse lookup tables.

        Args:
            __context: Pydantic initialization context.
        """
        print("loading model...")

        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print(f"model and vocab ({len(self.vocab)} tokens) ready")

        self.reversed_vocab = {
            int(value): key for key, value in self.vocab.items()
        }

    def _find_allowed_continuations(
        self,
        current_str: str,
        allowed_targets: List[str],
    ) -> List[int]:
        """Find token ids that keep the current string on a valid path.

        Args:
            current_str: The string generated so far.
            allowed_targets: The valid target strings to match against.

        Returns:
            Token ids that can still lead to one of the allowed targets.
        """
        legal_token_ids = []
        for token_id, token_chars in self.reversed_vocab.items():
            clean_token = self._normalize_token_text(token_chars)
            test_str = current_str + clean_token

            for target in allowed_targets:
                if target.startswith(test_str) or test_str == target:
                    legal_token_ids.append(token_id)
                    break
        return legal_token_ids

    def _drive_to_target_string(
        self,
        prompt_tokens: List[int],
        target_strings: List[str],
    ) -> str:
        """Greedily decode until one of the target strings is produced.

        Args:
            prompt_tokens: Token buffer representing the current context.
            target_strings: Valid strings that may be emitted.

        Returns:
            The generated string, or a partial prefix if decoding stalls.
        """
        accumulated_result = ""
        safety_limit = max((len(t) for t in target_strings), default=0) + 10

        for _ in range(safety_limit):
            valid_next_ids = self._find_allowed_continuations(
                accumulated_result,
                target_strings,
            )
            if not valid_next_ids:
                break

            raw_logits = self.model.get_logits_from_input_ids(prompt_tokens)
            masked_logits = np.full(len(raw_logits), -np.inf)

            for valid_id in valid_next_ids:
                if (valid_id < len(masked_logits) and
                        valid_id < len(raw_logits)):
                    masked_logits[valid_id] = raw_logits[valid_id]

            best_id = int(np.argmax(masked_logits))
            best_str = self._token_text(best_id)

            accumulated_result += best_str
            prompt_tokens.append(best_id)

            if accumulated_result in target_strings:
                break

        return accumulated_result

    def _extract_numeric_value(self, prompt_tokens: List[int],
                               prompt_len: int) -> str:
        """Generate a numeric literal from the model one token at a time.

        Args:
            prompt_tokens: Token buffer representing the current context.
            prompt_len: Original prompt length used to bound generation.

        Returns:
            A numeric string suitable for later conversion.
        """
        numeric_str = ""
        valid_chars = set("0123456789.-")
        halting_chars = set(",} \n\t")

        max_iters = max(50, prompt_len)
        for _ in range(max_iters):
            raw_logits = self.model.get_logits_from_input_ids(prompt_tokens)
            sorted_candidates = np.argsort(raw_logits)[::-1]

            selected_id = -1
            selected_char = ""

            for candidate_id in sorted_candidates:
                candidate_str = self._token_text(int(candidate_id))
                if not candidate_str:
                    continue

                if any(char in halting_chars for char in candidate_str):
                    if numeric_str and numeric_str not in ("-", ".", "-."):
                        return numeric_str
                    else:
                        continue

                if all(char in valid_chars for char in candidate_str):
                    test_str = numeric_str + candidate_str
                    is_valid = True

                    for i, char in enumerate(test_str):
                        if char == '-':
                            if i != 0:
                                is_valid = False
                                break
                        elif char == '.':
                            if test_str.count('.') > 1:
                                is_valid = False
                                break

                    if is_valid:
                        selected_id = candidate_id
                        selected_char = candidate_str
                        break

            if selected_id == -1:
                break
            numeric_str += selected_char
            prompt_tokens.append(int(selected_id))

        numeric_str = numeric_str.strip()

        if numeric_str in ("", "-", ".", "-."):
            return "0"

        return numeric_str

    def _extract_string_value(self, prompt_tokens: List[int],
                              prompt_len: int) -> str:
        """Generate a JSON string value until an unescaped quote is found.

        Args:
            prompt_tokens: Token buffer representing the current context.
            prompt_len: Original prompt length used to bound generation.

        Returns:
            The decoded string contents without the surrounding quotes.
        """
        string_token_ids: List[int] = []
        string_val = ""
        max_iters = max(100, prompt_len * 2)
        for _ in range(max_iters):
            raw_logits = self.model.get_logits_from_input_ids(prompt_tokens)
            sorted_ids = np.argsort(raw_logits)[::-1]

            selected_id = -1

            for token_id in sorted_ids:
                token_str = self._token_text(int(token_id))

                if not token_str:
                    continue
                if '\n' in token_str or '\r' in token_str:
                    continue

                if '"' in token_str:
                    unescaped_quote_idx = -1
                    for i, char in enumerate(token_str):
                        if char != '"':
                            continue

                        context_before = string_val + token_str[:i]
                        trailing_bs = (len(context_before) -
                                       len(context_before.rstrip('\\')))
                        is_escaped = trailing_bs % 2 == 1

                        if not is_escaped:
                            unescaped_quote_idx = i
                            break

                    if unescaped_quote_idx != -1:
                        sliced_str = token_str[:unescaped_quote_idx]
                        final_decoded_string = ""
                        if string_token_ids:
                            final_decoded_string = self.model.decode(
                                string_token_ids
                                )

                        final_decoded_string += sliced_str
                        if sliced_str:
                            self._append_encoded_text(prompt_tokens,
                                                      sliced_str)

                        return final_decoded_string

                selected_id = int(token_id)
                selected_char = token_str
                break

            if selected_id == -1:
                break

            string_val += selected_char
            string_token_ids.append(selected_id)
            prompt_tokens.append(selected_id)

        return str(self.model.decode(string_token_ids))

    def process_prompt(self, prompt: str) -> FunctionCallResult:
        """Convert a natural-language prompt into a structured function call.

        Args:
            prompt: The user input to analyze.

        Returns:
            A validated function call result containing the selected function
            name and extracted parameters, or an error result when decoding
            cannot proceed.
        """

        prompt = str(prompt) if prompt is not None else ""

        if not prompt or not prompt.strip():
            print("Error: Empty or whitespace-only prompt")
            return FunctionCallResult(
                prompt=prompt,
                name="error",
                parameters={"error": "Empty prompt"})

        if not self.function_definitions:
            print("model blocked, no valid token")
            return FunctionCallResult(
                prompt=prompt,
                name="error",
                parameters={})

        system_context = (
            "You are a helpful assistant. You have access to the "
            "following functions:\n"
        )
        for fn in self.function_definitions:
            system_context += f"- Function Name: {fn.name}\n"
            system_context += f"  Description: {fn.description}\n"

            safe_params = {k: v.model_dump(exclude_none=True)
                           for k, v in fn.parameters.items()}
            system_context += (
                f"  Parameters: {json.dumps(safe_params)}\n\n"
            )

        system_context += (
            "Choose the correct function based on "
            "the user's prompt.\n"
        )
        system_context += (
            "You must respond ONLY with a valid JSON object.\n\n"
        )

        full_prompt = (
            f"{system_context}User Prompt: {prompt}\n"
            f"Answer:"
        )

        prompt_tokens = self.model.encode(full_prompt).flatten().tolist()

        start_syntax = '{"name":"'
        final_json_string = start_syntax
        self._append_encoded_text(prompt_tokens, start_syntax)

        legal_function_names = [f'{fn.name}"'
                                for fn in self.function_definitions]
        inferred_fn_name_raw = self._drive_to_target_string(
            prompt_tokens,
            legal_function_names,
        )
        final_json_string += inferred_fn_name_raw
        inferred_fn_name = inferred_fn_name_raw.rstrip('"')

        selected_function_schema = next(
            (
                f
                for f in self.function_definitions
                if f.name == inferred_fn_name
            ),
            None,
        )
        if not selected_function_schema:
            return FunctionCallResult(prompt=prompt,
                                      name="error",
                                      parameters={})

        param_transition = ',"parameters":{'
        final_json_string += param_transition
        self._append_encoded_text(prompt_tokens, param_transition)

        extracted_arguments: Dict[str, Any] = {}
        required_keys = list(selected_function_schema.parameters.keys())

        prompt_length = len(prompt)
        # Generate data for each parameter sequentially
        for index, param_name in enumerate(required_keys):
            delimiter_handled = False
            key_syntax = f'"{param_name}":'
            final_json_string += key_syntax
            self._append_encoded_text(prompt_tokens, key_syntax)

            param_obj = selected_function_schema.parameters[param_name]
            param_type = param_obj.type
            param_options = param_obj.enum

            if param_options:
                if param_type == 'string':
                    allowed_choices = [f'"{opt}"' for opt in param_options]
                    generated_val = self._drive_to_target_string(
                        prompt_tokens,
                        allowed_choices,
                    )
                    final_json_string += generated_val
                    extracted_arguments[param_name] = generated_val.strip('"')
                else:
                    delimiter = ',' if index < len(required_keys) - 1 else '}'
                    allowed_choices = [
                        (str(opt).lower() if isinstance(opt, bool)
                         else str(opt)) + delimiter
                        for opt in param_options
                    ]
                    generated_val_raw = self._drive_to_target_string(
                        prompt_tokens,
                        allowed_choices,
                    )
                    final_json_string += generated_val_raw

                    generated_val = generated_val_raw.rstrip(delimiter)
                    delimiter_handled = True

                    if param_type in ('number', 'integer'):
                        try:
                            extracted_arguments[param_name] = (
                                float(generated_val)
                                if param_type == 'number'
                                else int(float(generated_val))
                            )
                        except (ValueError, TypeError):
                            extracted_arguments[param_name] = 0
                    elif param_type == 'boolean':
                        extracted_arguments[param_name] = (
                            generated_val == 'true'
                        )

            # Pure Numeric handling
            elif param_type in ('number', 'integer'):
                generated_val = self._extract_numeric_value(prompt_tokens,
                                                            prompt_length)
                final_json_string += generated_val
                try:
                    extracted_arguments[param_name] = (
                        float(generated_val)
                        if param_type == 'number'
                        else int(float(generated_val))
                    )
                except ValueError:
                    print(f"Warning: Failed to convert '{generated_val}' to "
                          f"{param_type}, defaulting to 0")
                    extracted_arguments[param_name] = 0

            # Boolean handling
            elif param_type == 'boolean':
                generated_val = self._drive_to_target_string(
                    prompt_tokens,
                    ['true', 'false'],
                )
                final_json_string += generated_val
                extracted_arguments[param_name] = (
                    generated_val == 'true'
                )

            # Pure String handling
            else:
                self._append_encoded_text(prompt_tokens, '"')
                final_json_string += '"'

                generated_val = self._extract_string_value(prompt_tokens,
                                                           prompt_length)
                final_json_string += generated_val + '"'

                self._append_encoded_text(prompt_tokens, '"')
                extracted_arguments[param_name] = generated_val

            if not delimiter_handled:
                if index < len(required_keys) - 1:
                    final_json_string += ','
                    self._append_encoded_text(prompt_tokens, ',')
                else:
                    final_json_string += '}'
                    self._append_encoded_text(prompt_tokens, '}')

        if not required_keys:
            final_json_string += '}'
            self._append_encoded_text(prompt_tokens, '}')

        final_json_string += '}'

        print(f"Answer: {final_json_string}")

        return FunctionCallResult(
            prompt=prompt,
            name=inferred_fn_name,
            parameters=extracted_arguments
        )
