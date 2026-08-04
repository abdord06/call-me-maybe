*This project has been created as part of the 42 curriculum by aredouan.*

## Description

**Call Me Maybe** is a function calling engine that translates natural language prompts into structured function calls using constrained decoding with a small LLM model. The project demonstrates how to build a reliable interface between user queries and predefined functions by constraining the LLM's output to valid, executable function schemas.

The goal is to enable users to invoke specific functions through natural language while ensuring the LLM respects strict parameter types, enumerated values, and function signatures. This is achieved through a state-machine-based constrained decoding approach that guides token generation toward syntactically valid JSON representations of function calls.

## Algorithm Explanation

### Constrained Decoding Approach

The core innovation of this project is the **state-machine constrained decoding** algorithm, which guides the LLM's token generation to produce valid function calls without relying on fine-tuning or reinforcement learning.

#### High-Level Process

1. **Prompt Construction**: User input is wrapped in a system context that lists all available functions with their schemas.

2. **Token Generation with Masking**: At each step, the algorithm:
   - Generates raw logits from the LLM
   - Identifies valid token continuations based on the current output
   - Masks logits to restrict selection to only legal tokens
   - Selects the token with the highest probability among valid options

3. **State-Specific Extraction**: Different parameter types have specialized extraction strategies:
   - **Strings**: Generate until a closing quote, handling escape sequences
   - **Numbers**: Extract valid numeric characters (digits, dots, minus signs)
   - **Booleans**: Constrain to "true" or "false"
   - **Enums**: Restrict to predefined enum values

#### Specific Algorithms

**String Matching (`_drive_to_target_string`)**:
- Maintains an accumulated result string
- For each valid continuation, finds tokens that extend the current string toward target strings
- Iterates up to a safety limit (25 iterations) to avoid infinite loops
- Stops when the accumulated result matches a target or no valid tokens exist

**Numeric Extraction (`_extract_numeric_value`)**:
- Iterates through token selections, accepting only valid numeric characters (0-9, -, .)
- Prevents multiple decimal points
- Halts on structural characters (comma, brace, newline) to mark end of number
- Safety limit of 20 iterations

**String Extraction (`_extract_string_value`)**:
- Iterates selecting top-probability tokens until a closing quote
- Handles escaped quotes by checking preceding backslash
- Prevents premature termination on escaped quotes
- Safety limit of 60 iterations

**Vocabulary Scanning (`_find_allowed_continuations`)**:
- For each vocabulary token, computes the normalized form (replacing Ġ with space, Ċ with newline)
- Tests if appending this token would keep us on track toward valid target strings
- Returns IDs of tokens that can legally extend the current string

## Design Decisions

### 1. Pydantic Models for Type Safety
Used Pydantic `BaseModel` for `FunctionDefinition`, `FunctionCallResult`, and `FunctionCaller` to ensure type validation and provide clear interfaces.

### 2. Vocabulary-Driven Constraint
Rather than post-processing or rejection sampling, the algorithm operates at the token level by scanning the model's vocabulary. This provides hard constraints and immediate feedback.

### 3. State Machine Architecture
Each parameter type has dedicated extraction logic, allowing specialized strategies (e.g., quoted strings vs. bare numbers) rather than a single generic approach.

### 4. Safety Limits on Iteration
To prevent infinite loops or excessive token generation, each extraction method has a safety limit (20-60 iterations). This balances correctness with predictable resource usage.

### 5. Reversed Vocabulary Cache
The vocabulary is loaded once and stored as both forward (token→ID) and reversed (ID→token) mappings to enable efficient lookups during constraint generation.

### 6. JSON Structure Preservation
The algorithm builds the output JSON incrementally, ensuring closing braces and commas are properly placed to produce valid JSON even if generation terminates early.

## Performance Analysis

### Accuracy
- The constrained decoding approach guarantees syntactically valid JSON output, eliminating parse errors that plague unconstrained LLM generation.
- Enum, boolean, and numeric constraints ensure type correctness for all parameters.
- The algorithm respects function signatures, preventing hallucinated function names or unexpected parameters.

### Speed
- Vocabulary scanning (`_find_allowed_continuations`) is O(V) per token, where V is vocabulary size (~50k tokens typical).
- With safety limits capping iterations per parameter, the total time per function call is predictable and scales linearly with parameter count.
- For 5-10 parameters, expect sub-second latency on modern hardware.

### Reliability
- Hard constraints eliminate generation failures caused by format non-compliance.
- Safety limits prevent infinite loops and runaway generation.
- Empty/whitespace prompts and malformed schemas are gracefully handled with error returns.
- Robust handling of escape sequences in strings and edge cases (negative numbers, floats, large integers).

## Challenges Faced

### 1. Vocabulary Token Normalization
**Challenge**: Tokenizers represent spaces as `Ġ` and newlines as `Ċ`, requiring normalization to match strings correctly.

**Solution**: Implemented `_normalize_token_text()` to consistently replace these special characters before string comparison.

### 2. Floating-Point Precision
**Challenge**: String-based numeric extraction can produce invalid floats or multiple decimal points.

**Solution**: Added validation in `_extract_numeric_value()` to prevent multiple dots and use try-catch for float conversion with sensible defaults (0).

### 3. Enum Type Diversity
**Challenge**: Enums can be strings, booleans, or numbers, each requiring different handling.

**Solution**: Check parameter type alongside enum values; convert selected values to the appropriate type before returning.

### 4. Escape Sequence Handling
**Challenge**: Strings may contain escaped quotes (`\"`), which must not terminate string extraction prematurely.

**Solution**: In `_extract_string_value()`, check if a quote is preceded by a backslash before treating it as terminator.

### 5. JSON Closing Braces
**Challenge**: Ensuring JSON validity even if generation is cut short by safety limits.

**Solution**: Explicitly append closing braces and commas in the correct places during parameter iteration, building valid JSON incrementally.

## Testing Strategy

### Test Coverage
Comprehensive test cases were created to validate edge cases and crash scenarios:

**Null/Empty Cases**:
- `null` prompts
- Empty string prompts
- Whitespace-only prompts

**Numeric Edge Cases**:
- Negative numbers (-10, -5, -1)
- Large numbers (999...999)
- Floating-point precision (3.14159, 0.123456789012345)
- Zero values
- Negative square root (invalid math)

**Type Handling**:
- Integer conversion from floats
- Enum value violations
- Boolean true/false
- Mixed numeric enum values

**Functions with Edge Cases**:
- Functions with no parameters
- Functions with string enums (4 values)
- Functions with boolean and numeric enums
- Functions with multiple parameters

### Validation Method
All test results are serialized to JSON and can be compared against expected outputs. The test suite covers both happy paths and failure modes to ensure robust error handling.

## Instructions

### Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone <repository-url>
cd call-me-maybe
make install
```

Or manually:

```bash
uv sync
```

### Execution

Run the main program with default input/output paths:

```bash
make run
```

Or specify custom paths:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json \
                      --input data/input/function_calling_tests.json \
                      --output data/output/function_calling_results.json
```

### Debug Mode

To run with Python debugger (pdb):

```bash
make debug
```

### Linting and Type Checking

Run code quality checks:

```bash
make lint
```

For strict type checking:

```bash
make lint-strict
```

Clean temporary files:

```bash
make clean
```

## Example Usage

### Input: Function Definitions
Define functions with schemas in `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_enum_priority",
    "description": "Select a priority level for a task.",
    "parameters": {
      "priority": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"]
      }
    },
    "returns": { "type": "string" }
  }
]
```

### Input: Test Prompts
Provide test cases in `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "Add 5 and 10",},
  { "prompt": "Greet Alice",},
  { "prompt": "Set priority to high"}
]
```

### Execution
```bash
make run
```

### Output
Results are saved to `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "Add 5 and 10",
    "name": "fn_add_numbers",
    "parameters": { "a": 5.0, "b": 10.0 }
  },
  {
    "prompt": "Greet Alice",
    "name": "fn_greet",
    "parameters": { "name": "Alice" }
  },
  {
    "prompt": "Set priority to high",
    "name": "fn_enum_priority",
    "parameters": { "priority": "high" }
  }
]
```

## Resources

### Documentation
- [Pydantic Documentation](https://docs.pydantic.dev/) - Type validation and model configuration
- [Python JSON Module](https://docs.python.org/3/library/json.html) - JSON parsing and serialization
- [NumPy Documentation](https://numpy.org/doc/) - Numerical operations and array handling

### References on Function Calling and Constrained Decoding
- [Outlines: Constrained Text Generation](https://github.com/outlines-ai/outlines) - Framework for constrained LLM decoding
- [GBNF: GGML-based Backus-Naur Form](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) - Grammar-based constraint approach
- [Guidance: Language Models as Programmable Controllers](https://arxiv.org/abs/2211.08493) - Theoretical foundation for constrained generation

### Tutorials
- [Tokenization Basics](https://platform.openai.com/docs/guides/tokens) - Understanding token normalization
- [JSON Schema Specification](https://json-schema.org/) - Formal parameter schema definition

### AI Usage in This Project

**AI assistance was used for the following tasks:**

1. **Code Structure & Architecture**: AI provided guidance on organizing the constrained decoding logic into modular methods and suggested the state-machine approach.

2. **Algorithm Optimization**: AI helped identify performance bottlenecks in vocabulary scanning and suggested caching the reversed vocabulary for faster lookups.

3. **Edge Case Testing**: AI suggested comprehensive test cases for Unicode, escape sequences, type conversion edge cases, and null/empty input handling.

4. **Documentation**: AI assisted in writing clear explanations of the algorithm, design rationale, and usage instructions.

5. **Error Handling**: AI recommended defensive programming practices for handling malformed schemas, type conversion failures, and safety limits.

**Parts of the project benefiting most from AI assistance**:
- Test case generation (`data/input/`)
- Documentation and README structure
- Type hint annotations and static analysis guidance
