import json
import re
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

    def model_post_init(self, __context: Any) -> None:
        """Initializes the model vocabulary upon instantiation."""
        print("loading model...")

        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print(f"model and vocab ({len(self.vocab)} tokens) ready")
        self.reversed_vocab = {value: key for key, value in self.vocab.items()}

    def get_current_function(self, text: str) -> FunctionDefinition | None:
        """Extracts the currently generated function name from the text.

        Args:
            text (str): The currently generated JSON string.

        Returns:
            FunctionDefinition | None: The matched function definition.
        """
        target_start = '{"name":"'
        start_idx = text.find(target_start)

        if start_idx != -1:
            start_idx += len(target_start)
            end_index = text.find('"', start_idx)
            if end_index != -1:
                fn_name = text[start_idx:end_index]
                for fn in self.function_definitions:
                    if fn.name == fn_name:
                        return fn
        return None

    def get_active_parameter_enum(
        self, text: str, current_fn: FunctionDefinition
    ) -> list | None:
        """Extracts the 'enum' list for the current parameter if it exists."""
        params_idx = text.find('"parameters":')
        if params_idx == -1:
            return None

        params_section = text[params_idx + len('"parameters":'):]
        keys_found = re.findall(r'"([^"]+)"\s*:', params_section)

        if not keys_found:
            return None

        last_key = keys_found[-1]
        last_colon = params_section.rfind(':')
        last_comma = params_section.rfind(',')

        if last_comma > last_colon:
            return None

        param_def = current_fn.parameters.get(last_key, {})
        return param_def.get('enum')

    def get_active_parameter_type(
        self, text: str, current_fn: FunctionDefinition
    ) -> str:
        """Determines the expected type of the parameter currently being
        generated.

        Args:
            text (str): The current text generation state.
            current_fn (FunctionDefinition): The chosen function schema.

        Returns:
            str: The expected type (e.g., 'number', 'boolean', 'string').
        """
        params_idx = text.find('"parameters":')
        if params_idx == -1:
            return ""

        params_section = text[params_idx + len('"parameters":'):]
        keys_found = re.findall(r'"([^"]+)"\s*:', params_section)

        if not keys_found:
            return ""

        last_key = keys_found[-1]
        last_colon = params_section.rfind(':')
        last_comma = params_section.rfind(',')

        if last_comma > last_colon:
            return ""

        param_def = current_fn.parameters.get(last_key, {})
        return str(param_def.get('type', ''))

    def check_parmetre_key(self, cl_gen: str,
                           token_str: str,
                           current_fn: FunctionDefinition) -> bool:

        target_params = '"parameters":{'
        allowed_params = list(current_fn.parameters.keys())
        last_comma_idx = cl_gen.rfind(',')
        param_start_idx = (cl_gen.find(target_params) +
                           len(target_params) - 1)
        search_start = max(param_start_idx, last_comma_idx)
        current_chunk = cl_gen[search_start+1:] + token_str
        first_quote_idx = current_chunk.find('"')
        if first_quote_idx == -1:
            if current_chunk != "":
                return False
            return True

        second_quote_idx = current_chunk.find('"', first_quote_idx + 1)

        if second_quote_idx == -1:
            current_key = current_chunk[first_quote_idx + 1:]
            if current_key and not any(k.startswith(current_key)
                                       for k in allowed_params):
                return False
        else:
            current_key = current_chunk[first_quote_idx + 1:second_quote_idx]

            if current_key not in allowed_params:
                return False

            after_quote = current_chunk[second_quote_idx + 1:]
            if after_quote:
                if not after_quote.startswith(':'):
                    return False
                if after_quote.count(':') > 1:
                    return False

        return True

    def is_valid_json(self, generated_text: str, token_str: str) -> bool:
        """Validates if the new token maintains valid JSON and adheres to the
        schema.

        Args:
            generated_text (str): The text generated so far.
            token_str (str): The new token candidate to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        token_str = token_str.replace('Ġ', ' ')
        text = generated_text + token_str
        text = text.replace(' ', '').replace('\n',
                                             '').replace('\r',
                                                         '').replace('\t', '')

        if not text.startswith('{'):
            return False

        target_start = '{"name":"'
        if len(text) <= len(target_start):
            return target_start.startswith(text)

        if not text.startswith(target_start):
            return False

        allowed_functions = [fn.name for fn in self.function_definitions]
        end_quote_idx = text.find('"', len(target_start))

        if end_quote_idx == -1:
            current_fn_name = text[len(target_start):]
            return any(fn.startswith(current_fn_name)
                       for fn in allowed_functions)

        current_fn_name = text[len(target_start):end_quote_idx]
        if current_fn_name not in allowed_functions:
            return False

        target_params = f'{target_start}{current_fn_name}","parameters":{{'

        if len(text) <= len(target_params):
            return target_params.startswith(text)

        if not text.startswith(target_params):
            return False

        cl_gen = generated_text.replace(' ', '').replace('\n', '')
        cl_gen = cl_gen.replace('\r', '').replace('\t', '')
        target_params = '"parameters":{'

        if target_params in cl_gen:
            current_fn = self.get_current_function(cl_gen)
            if current_fn:
                if '}' in token_str:
                    for required_key in current_fn.parameters.keys():
                        if f'"{required_key}":' not in text:
                            return False

                if not self.check_parmetre_key(cl_gen,
                                               token_str,
                                               current_fn):
                    return False

                expected_type = self.get_active_parameter_type(
                    cl_gen, current_fn
                )
                expected_enum = self.get_active_parameter_enum(
                    cl_gen, current_fn
                )

                if expected_enum:
                    last_colon_idx = cl_gen.rfind(':')
                    value_chunk = cl_gen[last_colon_idx + 1:] + token_str
                    value_chunk = value_chunk.lstrip(' \t\n\r')

                    if value_chunk:
                        if expected_type == 'string':
                            if not value_chunk.startswith('"'):
                                return False
                            clean_val = value_chunk[1:]
                            if '"' in clean_val:
                                actual_val = clean_val.split('"')[0]
                                if actual_val not in expected_enum:
                                    return False
                            else:
                                if (clean_val and not
                                    any(opt.startswith(clean_val)
                                        for opt in expected_enum)):
                                    return False

                        else:
                            str_enums = [str(opt).lower()
                                         if isinstance(opt, bool)
                                         else str(opt)
                                         for opt in expected_enum]

                            if any(c in value_chunk for c in ',} \t\n\r'):
                                actual_val = (re.split(r'[,}\s]',
                                              value_chunk)[0])
                                if actual_val not in str_enums:
                                    return False
                            else:
                                if not any(opt.startswith(value_chunk)
                                           for opt in str_enums):
                                    return False

                elif expected_type == 'number':
                    if not all(c in '0123456789.- ,}\n\r\t'
                               for c in token_str):
                        return False
                elif expected_type == 'boolean':
                    if not all(c in 'truefals ,}\n\t\r'
                               for c in token_str):
                        return False
        return True

    def process_prompt(self, prompt: str) -> FunctionCallResult:
        """Processes a natural language prompt using constrained decoding.

        Args:
            prompt (str): The user's input request.

        Returns:
            FunctionCallResult: The strictly formatted function call.
        """
        system_context = (
            "You are a helpful assistant. You have access to the "
            "following functions:\n"
        )
        for fn in self.function_definitions:
            system_context += f"- Function Name: {fn.name}\n"
            system_context += f"  Description: {fn.description}\n"
            system_context += f"  Parameters: {json.dumps(fn.parameters)}\n\n"

        system_context += ("Choose the correct function based on "
                           "the user's prompt.\n")
        system_context += ("You must respond ONLY with a valid JSON object "
                           "in this format: {\"name\": \"function_name\", \""
                           "parameters\": {\"key1\": value1, \"key2\": "
                           "value2}}\n\n")

        # seed_text = '{"name":"'
        full_prompt = (f"{system_context}User Prompt: {prompt}\n"
                       f"Answer:")

        raw_input = self.model.encode(full_prompt)
        input_ids = raw_input.flatten().tolist()

        generated_text = ""
        max_tokens = 150

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)

            if not generated_text.strip().endswith('}'):
                for special_token in [151643, 151644, 151645]:
                    logits[special_token] = float('-inf')

            logits_array = np.array(logits)
            sorted_token_ids = np.argsort(logits_array)[::-1]

            next_token_id = -1

            for token_id_np in sorted_token_ids:
                token_id = int(token_id_np)

                if token_id >= len(self.vocab):
                    continue

                token_str = self.reversed_vocab.get(token_id, "")
                if not token_str:
                    continue

                if self.is_valid_json(generated_text, token_str):
                    next_token_id = token_id
                    break

            if next_token_id == -1:
                print("model blocked, no valid token")
                break

            token_str = self.reversed_vocab.get(next_token_id, "")
            token_str = token_str.replace('Ġ', ' ')
            generated_text += token_str
            input_ids.append(next_token_id)

            open_count = generated_text.count('{')
            closed_count = generated_text.count('}')

            if open_count > 0 and open_count == closed_count:
                break

        last_brace_idx = generated_text.rfind('}')
        if last_brace_idx != -1:
            generated_text = generated_text[:last_brace_idx]

        clean_text = generated_text.strip()
        open_braces = clean_text.count('{')
        close_braces = clean_text.count('}')

        if open_braces > close_braces:
            clean_text += '}' * (open_braces - close_braces)

        clean_text = clean_text.replace('""', '"')
        clean_text = clean_text.replace(',}', '}')
        clean_text = clean_text.replace('\\"}}', '\\""}}')
        clean_text = re.sub(r'\\\\|\\(?![/"\\bfnrtu])', r'\\\\', clean_text)

        print(f"Answer: {clean_text}")

        try:
            parsed_json = json.loads(clean_text)
            fn_name = parsed_json.get("name", "")
            fn_parmtr = parsed_json.get("parameters", {})
        except json.JSONDecodeError as e:
            print(f"something in json format went wrong: {e}")
            fn_name = "error"
            fn_parmtr = {}

        return FunctionCallResult(
            prompt=prompt,
            name=fn_name,
            parameters=fn_parmtr
        )
