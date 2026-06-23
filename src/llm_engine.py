import json
import re
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, str]


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionCaller(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_definitions: List[FunctionDefinition]

    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    vocab: Dict[str, int] = Field(default_factory=dict)
    reversed_vocab: Dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        print("loading model...")

        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print(f"model and vocab ({len(self.vocab)} tokens) ready")
        self.reversed_vocab = {value: key
                               for key, value in self.vocab.items()}

    def get_current_function(self, text: str) -> FunctionDefinition | None:
        target_start = '{"name":"'
        if target_start in text:
            start_idx = len(target_start)
            end_index = text.find('"', start_idx)
            if end_index != -1:
                fn_name = text[start_idx:end_index]
                for fn in self.function_definitions:
                    if fn.name == fn_name:
                        return fn
        return None

    def get_active_parameter_type(self, text: str,
                                  current_fn: FunctionDefinition) -> str:
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
        return param_def.get('type', '')

    def is_valid_json(self, generated_text: str, token_str: str) -> bool:
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
                expected_type = self.get_active_parameter_type(
                    cl_gen, current_fn
                    )
                if expected_type == 'number':
                    if not all(c in '0123456789.- ,}\n\r\t'
                               for c in token_str):
                        return False
                elif expected_type == 'boolean':
                    if not all(c in 'truefals ,}\n\t\r'
                               for c in token_str):
                        return False
        return True

    def process_prompt(self, prompt: str) -> FunctionCallResult:
        system_context = ("You are a helpful assistant. You have "
                          "access to the following functions:\n")
        for fn in self.function_definitions:
            system_context += f"- Function Name: {fn.name}\n"
            system_context += f"  Description: {fn.description}\n"
            system_context += f"  Parameters: {json.dumps(fn.parameters)}\n\n"

        system_context += ("Choose the correct function based on "
                           "the user's prompt.\n")
        system_context += ("You must respond ONLY with a valid JSON object "
                           "in this format: {\"name\": \"function_name\", "
                           "\"parameters\": {\"param_name\": value}}\n\n")

        full_prompt = f"{system_context}User Prompt: {prompt}\nAnswer:"

        raw_input = self.model.encode(full_prompt)
        input_ids = raw_input.flatten().tolist()

        generated_text = ""
        max_tokens = 150

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)

            if not generated_text.strip().endswith('}'):
                for special_token in [151643, 151644, 151645]:
                    logits[special_token] = float('-inf')
            for ghost_token in range(len(self.vocab), len(logits)):
                logits[ghost_token] = float('-inf')

            for token_str, token_id in self.vocab.items():
                if not token_str:
                    logits[token_id] = float('-inf')
                    continue
                # test_json = generated_text + token_str

                if not self.is_valid_json(generated_text, token_str):
                    logits[token_id] = float('-inf')

            max_score = max(logits)
            if max_score == float('-inf'):
                print("model blocked, no valid token")
                break

            next_token_id = logits.index(max_score)

            token_str = self.reversed_vocab.get(next_token_id, "")
            token_str = token_str.replace('Ġ', ' ')
            # print(token_str, end="", flush=True)
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
