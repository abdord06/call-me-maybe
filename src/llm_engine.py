import json
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

    def is_valid_json(self, text: str) -> bool:
        # text = text.replace(' ', '').replace('\n', '')

        # if not text.startswith('{'):
        #     return False

        # target_start = '{"name":"'
        # if len(text) <= len(target_start):
        #     return target_start.startswith(text)

        # if not text.startswith(target_start):
        #     return False

        # allowed_functions = [fn.name for fn in self.function_definitions]
        # end_quote_idx = text.find('"', len(target_start))

        # if end_quote_idx == -1:
        #     current_fn_name = text[len(target_start):]
        #     return any(fn.startswith(current_fn_name)
        #                for fn in allowed_functions)

        # current_fn_name = text[len(target_start):end_quote_idx]
        # if current_fn_name not in allowed_functions:
        #     return False

        # target_params = f'{target_start}{current_fn_name}","parameters":{{'

        # if len(text) <= len(target_params):
        #     return target_params.startswith(text)

        # if not text.startswith(target_params):
        #     return False

        return True

    def process_prompt(self, prompt: str) -> FunctionCallResult:
        raw_input = self.model.encode(prompt)
        input_ids = raw_input.flatten().tolist()

        generated_text = ""
        max_tokens = 70

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)

            if not generated_text.strip().endswith('}'):
                for special_token in [151643, 151644, 151645]:
                    logits[special_token] = float('-inf')
            for ghost_token in range(len(self.vocab), len(logits)):
                logits[ghost_token] = float('-inf')

            for token_str, token_id in self.vocab.items():
                # token_str = token_str.replace('Ġ', '')
                if not token_str:
                    logits[token_id] = float('-inf')
                    continue
                test_json = generated_text + token_str

                if not self.is_valid_json(test_json):
                    logits[token_id] = float('-inf')

            max_score = max(logits)
            if max_score == float('-inf'):
                print("model blocked, no valid token")
                break

            next_token_id = logits.index(max_score)

            token_str = self.reversed_vocab.get(next_token_id, "")
            token_str = token_str.replace('Ġ', ' ')
            # print(next_token_id)
            # print(token_str)
            generated_text += token_str
            input_ids.append(next_token_id)
            if (generated_text.strip().endswith('}')
                    or next_token_id in [151645, 151643]):
                break

        print(f"Answer: {generated_text}")
        clean_text = generated_text.strip()
        open_braces = clean_text.count('{')
        close_braces = clean_text.count('}')

        if open_braces > close_braces:
            clean_text += '}' * (open_braces - close_braces)

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
