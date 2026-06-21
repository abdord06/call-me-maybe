import argparse
import json
import sys
import os
from typing import Any
from src.llm_engine import FunctionCaller, FunctionDefinition


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM, call me maybe")

    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/function_definitions.json",
                        help="path to the function definitions Json file"
                        )

    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path to the input JSON file containing prompts"
                        )

    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calling_results.json",
                        help="Path to save the output JSON file"
                        )
    args = parser.parse_args()

    try:
        with open(args.functions_definition, 'r', encoding='utf-8') as f:
            function_defs: list[dict[str, Any]] = json.load(f)
        with open(args.input, 'r', encoding='utf-8') as f:
            input_tests: list[dict[str, Any]] = json.load(f)

        print("Success in parsing input files")
        print(f"Output will be registered in: {args.output}")

        pydantic_functions = [FunctionDefinition(**f_dict)
                              for f_dict in function_defs]

        engine = FunctionCaller(function_definitions=pydantic_functions)

        results = []
        for test in input_tests:
            prompt_txt = test["prompt"]
            print(f"Answering: {prompt_txt}")

            result_obj = engine.process_prompt(prompt_txt)

            result_dict = result_obj.model_dump()

            result_dict["prompt"] = prompt_txt
            results.append(result_dict)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        print(f"Done! check file {args.output}")

    except FileNotFoundError as e:
        print(f"path of file not correct or file doesn't exist {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error in Json file {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
