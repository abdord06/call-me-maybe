"""Command-line entry point for the function-calling demo application."""

import argparse
import json
import sys
import os
from typing import Any, List, Dict
from src.llm_engine import FunctionCaller, FunctionDefinition, PromptTest
from pydantic import ValidationError


def main() -> None:
    """Run the CLI workflow for validating inputs and generating outputs.

    The command-line interface loads function definitions and prompt tests,
    validates them with Pydantic, invokes the function-calling engine, and
    writes the resulting structured calls to disk.
    """
    parser = argparse.ArgumentParser(description="LLM, call me maybe")

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="path to the function definitions Json file"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing prompts"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to save the output JSON file"
    )
    args = parser.parse_args()

    try:
        with open(args.functions_definition, 'r', encoding='utf-8') as f:
            function_defs: List[Dict[str, Any]] = json.load(f)

        with open(args.input, 'r', encoding='utf-8') as f:
            input_tests: List[Dict[str, Any]] = json.load(f)

        if not isinstance(input_tests, list):
            print("Error: 'function_calling_tests.json'"
                  " must be a JSON Array [].")
            sys.exit(1)

        if not isinstance(function_defs, list):
            print("Error: 'functions_definition.json'"
                  " must be a JSON Array [].")
            sys.exit(1)

        pydantic_functions = []
        seen_names = set()

        for i, f_dict in enumerate(function_defs):
            try:
                func_def = FunctionDefinition(**f_dict)
            except ValidationError as e:
                print(f"Error: Invalid format in functions_definition.json"
                      f" at index {i}:\n{e}")
                sys.exit(1)

            if func_def.name in seen_names:
                print(f"Error: Duplicate function name '{func_def.name}'")
                sys.exit(1)

            seen_names.add(func_def.name)
            pydantic_functions.append(func_def)

        validated_prompts = []
        for i, test_dict in enumerate(input_tests):
            try:
                test_obj = PromptTest(**test_dict)
                validated_prompts.append(test_obj.prompt)
            except ValidationError as e:
                print(f"Error: Invalid format in function_calling_tests.json"
                      f" at index {i}:\n{e}")
                sys.exit(1)

        print("Success in parsing input files (Strictly validated)")
        print(f"Output will be registered in: {args.output}")

        engine = FunctionCaller(function_definitions=pydantic_functions)

        results: List[Dict[str, Any]] = []
        for prompt_txt in validated_prompts:
            print(f"\nAnswering: {prompt_txt}")

            try:
                result_obj = engine.process_prompt(prompt_txt)
                result_dict = result_obj.model_dump()
                results.append(result_dict)
            except Exception as e:
                print(f"Error processing prompt '{prompt_txt[:20]}...': {e}")
                results.append({
                    "prompt": prompt_txt,
                    "name": "error",
                    "parameters": {}
                }
                )
                continue

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        print(f"\nDone! check file {args.output}")

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
