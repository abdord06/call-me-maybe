import json
import numpy as np
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field
from llm_sdk import Small_LLM_Model  # type: ignore


class FunctionDefinition(BaseModel):
    """Pydantic model representing a function's definition and schema."""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, str]


class FunctionCallResult(BaseModel):
    """Pydantic model representing the final result of the LLM generation."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionCaller(BaseModel):
    """Engine responsible for constrained decoding and LLM interaction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_definitions: List[FunctionDefinition]

    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    vocab: Dict[str, int] = Field(default_factory=dict)
    reversed_vocab: Dict[int, str] = Field(default_factory=dict)

    def _normalize_token_text(self, token_text: str) -> str:
        return token_text.replace('Ġ', ' ').replace('Ċ', '\n')

    def _token_text(self, token_id: int) -> str:
        raw_text = self.reversed_vocab.get(token_id, "")
        return self._normalize_token_text(raw_text)

    def _append_encoded_text(
        self,
        prompt_tokens: List[int],
        text: str,
    ) -> None:
        encoded_text = self.model.encode(text).flatten().tolist()
        prompt_tokens.extend(encoded_text)

    def model_post_init(self, __context: Any) -> None:
        """Initializes the model vocabulary upon instantiation."""
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
        """Scan vocabulary tokens that extend the current string toward valid
        targets."""
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
        """Restrict output to a predefined list of valid strings."""
        accumulated_result = ""
        safety_limit = 25

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

    def _extract_numeric_value(self, prompt_tokens: List[int]) -> str:
        """Force the model to yield only valid characters for a number."""
        numeric_str = ""
        valid_chars = set("0123456789.-eE+")
        halting_chars = set(",} \n\t")

        for _ in range(30):
            raw_logits = self.model.get_logits_from_input_ids(prompt_tokens)
            sorted_candidates = np.argsort(raw_logits)[::-1]

            selected_id = -1
            selected_char = ""

            for candidate_id in sorted_candidates:
                candidate_str = self._token_text(int(candidate_id))
                if not candidate_str:
                    continue

                if any(char in halting_chars for char in candidate_str):
                    if numeric_str and numeric_str not in ("-", "+", ".",
                                                           "e", "E", "-.",
                                                           "+."):
                        return numeric_str
                    else:
                        continue

                if all(char in valid_chars for char in candidate_str):
                    test_str = numeric_str + candidate_str

                    e_count = test_str.count('e') + test_str.count('E')
                    is_valid = True

                    for i, char in enumerate(test_str):
                        if char in '-+':
                            if i != 0 and test_str[i-1] not in 'eE':
                                is_valid = False
                                break
                        elif char == '.':
                            if test_str.count('.') > 1:
                                is_valid = False
                                break
                            if 'e' in test_str[:i] or 'E' in test_str[:i]:
                                is_valid = False
                                break

                    if is_valid and e_count <= 1:
                        selected_id = candidate_id
                        selected_char = candidate_str
                        break

            if selected_id == -1:
                break
            numeric_str += selected_char
            prompt_tokens.append(int(selected_id))

        numeric_str = numeric_str.strip()
        if numeric_str in ("", "-", "+", ".", "e", "E", "-.", "+."):
            return "0"

        return numeric_str

    def _extract_string_value(self, prompt_tokens: List[int]) -> str:
        """Generate a string and stop when a closing quote appears."""
        string_token_ids: List[int] = []
        string_val = ""
        for _ in range(60):
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
        """Process a prompt using structured state machine decoding."""

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
            system_context += (
                f"  Parameters: {json.dumps(fn.parameters)}\n\n"
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

        legal_function_names = [fn.name for fn in self.function_definitions]
        inferred_fn_name = self._drive_to_target_string(
            prompt_tokens,
            legal_function_names,
        )
        final_json_string += inferred_fn_name

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

        param_transition = '","parameters":{'
        final_json_string += param_transition
        self._append_encoded_text(prompt_tokens, param_transition)

        extracted_arguments: Dict[str, Any] = {}
        required_keys = list(selected_function_schema.parameters.keys())

        # Generate data for each parameter sequentially
        for index, param_name in enumerate(required_keys):
            key_syntax = f'"{param_name}":'
            final_json_string += key_syntax
            self._append_encoded_text(prompt_tokens, key_syntax)

            param_schema = selected_function_schema.parameters.get(param_name,
                                                                   {})
            if not isinstance(param_schema, dict):
                print(f"Warning: Malformed schema for '{param_name}',"
                      f"defaulting to string.")
                param_schema = {}

            param_type = param_schema.get('type', 'string')
            param_options = param_schema.get('enum', None)

            if param_options is not None and not isinstance(param_options,
                                                            list):
                print(f"Warning: 'enum' for '{param_name}' is "
                      f"a {type(param_options).__name__}, not a list."
                      f" Ignoring enum.")
                param_options = None

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
                    allowed_choices = [
                        str(opt).lower() if isinstance(opt, bool) else str(opt)
                        for opt in param_options
                    ]
                    generated_val = self._drive_to_target_string(
                        prompt_tokens,
                        allowed_choices,
                    )
                    final_json_string += generated_val

                    if param_type in ('number', 'integer'):
                        try:
                            extracted_arguments[param_name] = (
                                float(generated_val)
                                if param_type == 'number'
                                else int(float(generated_val))
                            )
                        except (ValueError, TypeError):
                            print(
                                f"Warning: Failed to convert enum value "
                                f"'{generated_val}' to {param_type}, "
                                f"defaulting to 0"
                            )
                            extracted_arguments[param_name] = 0
                    elif param_type == 'boolean':
                        extracted_arguments[param_name] = (
                            generated_val == 'true'
                        )

            # Pure Numeric handling
            elif param_type in ('number', 'integer'):
                generated_val = self._extract_numeric_value(prompt_tokens)
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

                generated_val = self._extract_string_value(prompt_tokens)
                final_json_string += generated_val + '"'

                self._append_encoded_text(prompt_tokens, '"')
                extracted_arguments[param_name] = generated_val

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
