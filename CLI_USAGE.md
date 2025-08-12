# CLI Usage Guide

This CLI tool provides a convenient interface for running text-to-SQL agent evaluations with different workflow configurations.

## Installation

Make sure you have the dependencies installed. If using `uv`:

```bash
uv sync
```

## Basic Usage

Run the CLI using `uv run`:

```bash
uv run python cli.py eval [OPTIONS]
```

or activate virtual environment `source .venv/bin/activate` and simply run:

```bash
python cli.py eval [OPTIONS]
```

## Available Workflows

- **1.1**: Basic advanced workflow with standard response synthesis
- **1.2**: Advanced workflow with restricted context (only uses SQL response info)
- **2.1**: Advanced workflow with vector indexing for enhanced table retrieval

## Command Options

### Required/Main Options
- `--workflow [1.1|1.2|2.1]`: Choose which workflow version to run (default: 1.1)

### Evaluation Options
- `--ragchecker / --no-ragchecker`: Run RAGChecker evaluation after workflow execution
- `--jllm / --no-jllm`: Run JLLM evaluation after workflow execution

### Output Options
- `--output TEXT`: Specify custom output filename (default: workflow{version}_results.json)

### Model Configuration
- `--model TEXT`: OpenAI model to use (default: gpt-4o-mini)
- `--temperature FLOAT`: Temperature for the LLM (default: 0.5)

### Other Options
- `--verbose / --no-verbose`: Enable verbose output
- `--help`: Show help message

## Examples

### Run Basic Workflow
```bash
# Run workflow 1.1 with default settings
uv run python cli.py eval --workflow 1.1
```

### Run with Evaluations
```bash
# Run workflow 2.1 with both RAGChecker and JLLM evaluations
uv run python cli.py eval --workflow 2.1 --ragchecker --jllm
```

### Custom Output and Model
```bash
# Run workflow 1.2 with custom output file and different model
uv run python cli.py eval --workflow 1.2 --output my_results.json --model gpt-4 --ragchecker
```

### Verbose Output
```bash
# Run with verbose output to see detailed progress and results
uv run python cli.py eval --workflow 1.1 --ragchecker --jllm --verbose
```

## Output Files

The CLI generates several output files:

1. **Main Results**: `workflow{version}_results.json` (or custom filename)
   - Contains query results, responses, and retrieved context
   - Used as input for evaluation tools

2. **RAGChecker Results**: `{main_file}_eval.json` (when --ragchecker is used)
   - Contains RAGChecker evaluation metrics and detailed results

3. **JLLM Results**: Displayed in terminal output
   - Shows which queries the agent claims it cannot answer

## Workflow Details

### Workflow 1.1
- Basic advanced workflow with query-time table retrieval
- Uses standard response synthesis prompt
- Good baseline for comparisons

### Workflow 1.2  
- Same as 1.1 but with restricted response synthesis
- Only uses information from SQL response (no external knowledge)
- Tests pure SQL-based answering capability

### Workflow 2.1
- Enhanced with vector indexing for table content
- Better table retrieval based on semantic similarity
- Most advanced workflow option

## Evaluation Metrics

### RAGChecker Metrics
- **Overall**: Precision, Recall, F1
- **Retriever**: Claim recall, Context precision  
- **Generator**: Context utilization, Noise sensitivity, Hallucination, Faithfulness

### JLLM Evaluation
- Identifies queries where the agent claims it cannot find the answer
- Useful for measuring answer confidence and completeness
