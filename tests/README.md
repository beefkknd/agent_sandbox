# LangChain Agent Testing Guide

This directory contains tests for the refactored, class-based LangChain agent implementation.

## Quick Start

### 1. Install Test Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"

# Or with uv:
uv pip install -e ".[test]"
```

### 2. Run Your First Test

The most important test to get working first is `test_orchestrator_creates_first_task`:

```bash
pytest tests/unit/test_orchestrator.py::test_orchestrator_creates_first_task -v -s
```

This test demonstrates:
- How to create an `OrchestratorNode` instance
- How to set up initial state
- How to call the node's `run()` method
- How to verify the output

**Expected output**: The test should pass and you'll see the orchestrator create a task based on the query.

### 3. Run All Orchestrator Tests

```bash
# Run all orchestrator tests (including LLM calls)
pytest tests/unit/test_orchestrator.py -v -s

# Run only fast tests (no LLM calls)
pytest tests/unit/test_orchestrator.py -m "unit and not llm" -v
```

## Test Structure

```
tests/
├── unit/                           # Unit tests for individual nodes
│   ├── test_orchestrator.py        # OrchestratorNode tests (START HERE!)
│   ├── test_assistant.py           # AssistantNode tests (to be created)
│   └── ...
├── integration/                    # Integration tests for workflows
│   └── test_full_workflow.py      # Full graph tests (to be created)
├── fixtures/
│   └── states/                     # Saved state snapshots for debugging
├── utils/
│   ├── test_utils.py              # State dump/load utilities
│   └── mock_helpers.py            # Mock factories (to be created)
└── README.md                       # This file
```

## Running Tests

### By Marker

Tests are marked with pytest markers for selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only LLM tests (real API calls)
pytest -m llm

# Skip expensive LLM tests
pytest -m "not llm"

# Run integration tests
pytest -m integration
```

### By File

```bash
# Run specific test file
pytest tests/unit/test_orchestrator.py -v

# Run specific test function
pytest tests/unit/test_orchestrator.py::test_orchestrator_creates_first_task -v
```

### All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with output capture disabled (see print statements)
pytest -s
```

## Understanding the Tests

### Prompt Rendering Tests (Fast, No LLM)

These tests verify that prompts are correctly formatted with state context WITHOUT calling the LLM:

```python
@pytest.mark.unit
def test_orchestrator_prompt_rendering_initial_state(orchestrator_node):
    state = create_minimal_state("Find vessel location")

    prompt_text = orchestrator_node.render_prompt(state)

    assert "Find vessel location" in prompt_text
    assert "No tasks completed yet" in prompt_text
```

**Use case**: Verify prompt quality and context interpolation quickly.

### Node Execution Tests (With Real LLM)

These tests run the actual node with real LLM calls:

```python
@pytest.mark.llm
@pytest.mark.unit
@pytest.mark.asyncio
async def test_orchestrator_creates_first_task(orchestrator_node):
    state = create_minimal_state("Find vessel ADAMAS III")

    result = await orchestrator_node.run(state)

    assert len(result["tasks"]) == 1
    assert result["active_task_id"] == 1
```

**Use case**: Test actual node behavior with real LLM responses.

## Debugging Workflow

When you see unexpected behavior during a live agent run:

### Step 1: Capture the State

Add this to your agent code temporarily:

```python
from tests.utils.test_utils import dump_agent_state

# After a node produces unexpected output
dump_agent_state(current_state, "debug_states/weird_output.json")
```

### Step 2: Create a Test with Captured State

```python
from tests.utils.test_utils import load_agent_state

@pytest.mark.llm
@pytest.mark.asyncio
async def test_debug_weird_output(orchestrator_node):
    # Load the exact state that caused the issue
    state = load_agent_state("debug_states/weird_output.json")

    # First, check what prompt was sent
    prompt = orchestrator_node.render_prompt(state)
    print(prompt)  # Inspect the prompt

    # Then run the node
    result = await orchestrator_node.run(state)

    # Add assertions to prevent regression
    assert "weird_thing" not in result["tasks"][0]["question"]
```

### Step 3: Run and Debug

```bash
pytest tests/unit/test_orchestrator.py::test_debug_weird_output -v -s
```

### Step 4: Fix and Verify

1. Modify the prompt in `agent_app/config.py`
2. Re-run the test
3. Verify the fix works
4. Keep the test to prevent regression

## Test Utilities

### `create_minimal_state(query)`

Creates a basic initial state for testing:

```python
from tests.utils.test_utils import create_minimal_state

state = create_minimal_state("Find vessel location")
```

### `dump_agent_state(state, filepath)`

Saves state to JSON file:

```python
from tests.utils.test_utils import dump_agent_state

dump_agent_state(current_state, "debug_states/state1.json")
```

### `load_agent_state(filepath)`

Loads state from JSON file:

```python
from tests.utils.test_utils import load_agent_state

state = load_agent_state("debug_states/state1.json")
```

### `print_agent_state(state)`

Pretty-prints state to console:

```python
from tests.utils.test_utils import print_agent_state

print_agent_state(current_state)
```

## Writing New Tests

### For a New Node

1. Create `tests/unit/test_[node_name].py`
2. Add fixtures for the node:

```python
@pytest.fixture
def my_node(llm, config):
    return MyNode(llm, config)
```

3. Add prompt rendering tests (fast):

```python
@pytest.mark.unit
def test_my_node_prompt_rendering(my_node):
    state = {...}
    prompt = my_node.render_prompt(state)
    assert "expected content" in prompt
```

4. Add execution tests (with LLM):

```python
@pytest.mark.llm
@pytest.mark.unit
@pytest.mark.asyncio
async def test_my_node_execution(my_node):
    state = {...}
    result = await my_node.run(state)
    assert result["expected_key"] == "expected_value"
```

## Next Steps

After getting `test_orchestrator.py` working:

1. **Test other nodes**: Create `test_assistant.py`, `test_consolidator.py`, etc.
2. **Test routing**: Create `test_routing.py` for routing functions
3. **Integration tests**: Create `test_full_workflow.py` for multi-node flows
4. **Add more test data**: Create fixture files with captured states

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running pytest from the project root:

```bash
cd /Users/yingzhou/work/agent_sanbox
pytest tests/unit/test_orchestrator.py -v
```

### LLM API Errors

If you get API errors, check:
- Environment variables are set (`.env` file loaded)
- API keys are valid
- You have credits/quota available

### Async Test Errors

Make sure:
- Tests use `async def` for async functions
- Tests are marked with `@pytest.mark.asyncio`
- `pytest-asyncio` is installed

## Cost Considerations

Tests marked with `@pytest.mark.llm` make real API calls and may incur costs. To avoid costs during development:

```bash
# Skip LLM tests
pytest -m "not llm"

# Or run only fast unit tests
pytest -m "unit and not llm"
```

For CI/CD, you may want to mock LLMs or use a separate test budget.
