import asyncio
import json
from typing import List, Dict, Annotated, TypedDict
from functools import partial

import backoff
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient

from llm_factory import LLMFactory


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_ITERATIONS = 9  # Max rounds of the orchestrator


# ============================================================================
# LLM INVOCATION WITH BACKOFF
# ============================================================================

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def invoke_llm_with_backoff(chain, params):
    """Invokes the LLM chain with exponential backoff on ResourceExhausted errors.

    Includes a 5-second delay before each call to prevent hitting rate limits.
    """
    # Add delay before each API call to avoid rate limiting
    await asyncio.sleep(5)
    return await chain.ainvoke(params)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
ORCHESTRATOR_PROMPT = """
You are a master orchestrator agent. Your job is to methodically break down a user's complex request into a sequence of concise, actionable tasks for a team of worker agents.

**User's Goal:**
{input}

**Established Facts (From completed tasks):**
{tasks_completed}

**Your Role:**
1. **Analyze Goal & Facts**: Deeply understand the user's goal and compare it against the "Established Facts".
1. **Evaluate Sufficiency:** Ask yourself: "Do the Established Facts provide a direct and useful answer to the user's Likely Intent?" Avoid chasing unnecessary details. If the main question is already answered, it is time to stop.
2. **Identify the Next Step:** If and only if the goal is NOT sufficiently met, Identify the single most critical piece of information needed next.
3. **Formulate Task:** State the next task as a clear, direct command for the worker agent.

**Output Format:**
* **If more work is needed:** Provide ONLY the single, imperative command for the next task. Do not include explanations, reasoning, or conversational text.
  * *Bad:* "Okay, so to find the customer order history, we first need to identify the customer by their email."
  * *Good:* "Find the user's email address from avaiation records."
* **If the goal is met:** Respond with ONLY the word "DONE".

Example (Next Task Command or "DONE"):
"""

MY_ASSISTANT_PROMPT = """
You are acting as a **Principal Engineer**, a seasoned expert in solving complex technical problems. Your primary function is to achieve the current task by strategically utilizing the available tools. You are methodical, analytical, and always choose the most efficient tools for the job.

**Overall User Goal:**
{input}

**Your Current Task:**
{task_question}

**Your Scratchpad (Recent Tool History):**
{scratchpad}

**Intelligent Analysis:**

* **Think beyond the literal:** Your role is to be a "smart" agent, not just literal interpreter.
* **Derive and Infer:** If a user's query requires information that is not directly available as a field in the data, you must formulate a plan to derive or infer that information from the available data.
* **Example of Derivation:** A person's birth day can infer their zodiac sign and age. A latitude/longitude can infer a city or country.

**Your Available Tools:**
{tool_names}

**Instructions:**
You are in a loop to solve the current task. You must continue to execute actions until the task is fully resolved.
1. **Thought:** Adopt the mindset of a Principal Engineer. Analyze the task, your scratchpad, and available tools. Formulate a clear, step-by-step plan for your next action.
2. **Action:** Based on your thought process, decide on one of the following possible actions:
    a. **Call Tool(s):** If you determine that more information is required, call one or more tools. The tool name(s) you MUST use be from the "Available Tools" list. Be precise in your input to the tool(s).
    b. **Provide Final Answer:** Once you have all the necessary data and are confident in your conclusion, provide the final answer as a clear, comprehensive text response. If you have successfully found the information and the task is complete, start your response with "Final Answer:". If you could not find the information and are providing suggestions or a partial answer, start your response with "Suggestion:". Do NOT call any more tools.
"""

SUMMARIZE_ANSWER_PROMPT = """You are a final report generator. Your job is to provide a concise, final answer to the user's original question based on the work done by an agent.

**User's Original Question:**
{input}

**Completed Sub-Tasks and Their Answers:**
{tasks_completed}

**Final Answer:**
"""

TASK_SUMMARY_PROMPT = """You are a summarization agent. Your job is to create a concise summary of a task's execution for the master orchestrator. The orchestrator needs to understand what was done, what was found, and if anything failed.

**Task Question:**
{task_question}

**Execution History (Tool Calls & Results):**
{execution_history}

**Assistant's Final Conclusion:**
{assistant_conclusion}

**Your Summary:**
Based on the execution, provide a bulleted list of key findings and outcomes. Be factual and concise. Mention any tool failures.
This summary will inform the orchestrator's next decision.

*   **Outcome:** [Briefly state the main outcome of the task]
*   **Key Findings:**
    *   [Finding 1]
    *   [Finding 2]
    *   ...
*   **Failures:**
    *   [If any tools failed, describe the failure, e.g., "Tool 'x' failed with error: ..."]
    *   [If no failures, state "None"]
"""


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
class Task(TypedDict):
    """Represents a single sub-task created by the orchestrator.

    Attributes:
        id: Unique identifier for the task
        question: The task question/command from orchestrator
        history: Task-specific conversation history containing:
                 - HumanMessage (initial task question)
                 - AIMessage (with tool_calls if tools were invoked)
                 - ToolMessage (results from tool execution)
                 This allows the orchestrator to review what it asked for each task.
        answer: Final answer produced by my_assistant for this task
    """
    id: int
    question: str
    history: Annotated[List[BaseMessage], add_messages]
    answer: str

class AgentState(TypedDict):
    """Global state shared across the entire agent workflow.

    Attributes:
        messages: Top-level conversation (user's original question + final summary)
        active_task_id: ID of the task currently being worked on
        tasks: List of all tasks (completed and in-progress)
        iteration: Orchestrator iteration counter
        scratchpad: Global history of ALL tool calls across ALL tasks.
                    This helps my_assistant avoid redundant tool calls
                    (e.g., calling get_es_mapping multiple times).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    active_task_id: int
    tasks: List[Task]
    iteration: int
    scratchpad: Annotated[List[ToolMessage], add_messages]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _format_scratchpad_for_prompt(scratchpad: List[ToolMessage]) -> str:
    """Format the last 5 tool messages from the scratchpad into a
    de-duplicated string for the LLM prompt.

    The scratchpad contains ALL tool calls from ALL tasks, allowing
    my_assistant to see what tools have already been called and avoid
    redundant operations (e.g., calling get_es_mapping multiple times).
    """
    if not scratchpad:
        return "No tools used yet."

    # Get the last 5 tools
    recent_tools = scratchpad[-5:]

    run_text_lines = []
    for tool_msg in recent_tools:
        run_text_lines.append(f"Tool: {tool_msg.name}")
        run_text_lines.append(f"Input: {tool_msg.additional_kwargs.get('tool_input', 'N/A')}")
        run_text_lines.append(f"Result: {tool_msg.content}")

    # FIXME: De-duplicating the scratchpad might cause loss
    # of important context if a tool is called multiple times to check state.
    # This can be commented out if it causes issues.
    unique_lines = list(dict.fromkeys(run_text_lines))
    return "\n".join(unique_lines)


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

# --- Orchestrator Node ---
async def orchestrator(state: AgentState, llm: BaseChatModel):
    """The master agent that breaks down the problem and manages tasks.
    """
    print(f"\n======================================================\n"
          f"--- Orchestrator Iteration: {state['iteration']} / {MAX_ITERATIONS}\n"
          f"======================================================")
    iteration = state['iteration'] + 1

    if not state['tasks'] and iteration > MAX_ITERATIONS:
        tasks_completed = "No tasks completed yet."
    else:
        tasks_completed_list = [f"Task: {t['question']}\nOutcome: {t['answer']}" for t in state['tasks'] if t["answer"]]
        tasks_completed = "\n\n".join(sorted(tasks_completed_list)) if tasks_completed_list else "No tasks completed yet."

    prompt = ChatPromptTemplate.from_template(ORCHESTRATOR_PROMPT)
    chain = prompt | llm

    response = await invoke_llm_with_backoff(chain, {
        "input": state['messages'][0].content,
        "tasks_completed": tasks_completed,
    })

    decision = response.content

    if "DONE" in decision or not decision:
        print("[Orchestrator]: All tasks are complete. Processing to summary.")
        return {"active_task_id": None}

    new_task_id = len(state['tasks']) + 1
    new_task = Task(id=new_task_id, question=decision, history=[HumanMessage(content=decision)], answer="")
    print(f"[Orchestrator]: New Task Created: {new_task['question']}")

    return {
        "tasks": state['tasks'] + [new_task],
        "iteration": iteration,
        "active_task_id": new_task_id,
    }


# --- Assistant Node ---
async def my_assistant(state: AgentState, llm_with_tools: BaseChatModel, tools: List) -> Dict:
    """The worker agent that uses tools to solve a single task in a loop.

    This agent receives:
    1. The current task question from the orchestrator
    2. Task-specific history (for conversation continuity within this task)
    3. Global scratchpad (to avoid redundant tool calls across all tasks)
    """
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return state

    print(f"\n   [Assistant] Task: \"{active_task['question']}\"")

    tool_names = ", ".join([t.name for t in tools])

    # Get scratchpad: global tool call history across ALL tasks
    scratchpad_content = _format_scratchpad_for_prompt(state.get('scratchpad', []))

    prompt = ChatPromptTemplate.from_messages([
        ("system", MY_ASSISTANT_PROMPT),
        MessagesPlaceholder(variable_name="history"),
    ])

    chain = prompt | llm_with_tools

    # IMPORTANT: 'history' is task-specific, containing only the conversation
    # for THIS task (HumanMessage + AIMessages + ToolMessages).
    # This allows the orchestrator to review what it asked in each task.
    # The LLM (Gemini) understands this format natively - no conversion needed!
    history = active_task.get('history', [])

    response_message = await invoke_llm_with_backoff(chain, {
        "input": state['messages'][0].content,
        "task_question": active_task['question'],
        "history": history,  # Task-specific conversation
        "tool_names": tool_names,
        "scratchpad": scratchpad_content,  # Global tool call history
    })

    if response_message.tool_calls:
        tool_name = ", ".join([tc['name'] for tc in response_message.tool_calls])
        print(f"   [Assistant]: Calling tool(s): {tool_name}")
    else:
        print(f"   [Assistant]: Reaching a final answer.")

    # Append response to task-specific history
    active_task['history'].append(response_message)
    return {"tasks": state['tasks']}


# --- Tool Execution Nodes ---
async def execute_tool_call(tool_call, tools_map) -> ToolMessage:
    """Executes tools called by the assistant in parallel and appends results.
    """
    tool_name = tool_call['name']
    try:
        tool_to_call = tools_map[tool_name]
        output = await tool_to_call.ainvoke(tool_call['args'])
        return ToolMessage(
            content=str(output),
            tool_call_id=tool_call['id'],
            name=tool_name,
            additional_kwargs={"tool_input": tool_call['args']}
        )
    except Exception as e:
        error_payload = json.dumps({
            "error": f"Tool Execution Error: '{type(e).__name__}'",
            "details": str(e),
            "tool_name": tool_name,
            "tool_args": tool_call['args'],
        })
        return ToolMessage(
            content=error_payload,
            tool_call_id=tool_call['id'],
            name=tool_name,
            additional_kwargs={"tool_input": tool_call['args']}
        )


async def tool_execution(state: AgentState, tools: List) -> Dict:
    """Execute all tool calls from the last assistant message.

    Results are stored in TWO places:
    1. Task-specific history (for conversation continuity within the task)
    2. Global scratchpad (to prevent redundant tool calls across all tasks)
    """
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return {}

    last_message = active_task['history'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_map = {tool.name: tool for tool in tools}
    tasks = [execute_tool_call(tc, tool_map) for tc in last_message.tool_calls]
    tool_results = await asyncio.gather(*tasks)

    # Log each tool execution with deeper indentation
    for tool_result in tool_results:
        print(f"      [Tool: {tool_result.name}] Executed")

    # Add tool results to task-specific history (for this task's conversation)
    active_task['history'].extend(tool_results)

    # Add tool results to global scratchpad (to avoid redundant calls later)
    return {
        "tasks": state['tasks'],
        "scratchpad": state['scratchpad'] + tool_results,
    }


# --- Consolidator Node ---
async def tool_result_consolidator(state: AgentState) -> Dict:
    """Consolidates tool results into a summary for the orchestrator."""
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return {}

    print(f"   [Consolidator] Task: \"{active_task['question']}\"")
    print(f"   [Consolidator]: storing final answer...")

    # Extract final answer from history
    final_answer_message = active_task['history'][-1]

    if isinstance(final_answer_message, AIMessage) and not final_answer_message.tool_calls:
        final_answer = final_answer_message.content
        active_task['answer'] = final_answer
        print(f"   [Orchestrator]: Task reviewed task answer: \"{final_answer}\"")
    else:
        active_task['answer'] = "Error: No final answer provided by assistant."    

    return {
        "tasks": state['tasks'],
        "active_task_id": None,
    }


# --- Summary Node ---
async def summarize_answer(state: AgentState, llm: BaseChatModel) -> Dict:
    """Generate final summary answer from all completed tasks."""
    print(f"\n----------------------------------------\n"
          f"--Summarizing Final Answer\n"
          f"----------------------------------------")
    task_completed = "\n".join([f"Task: {t['question']}\nAnswer: {t['answer']}" for t in state['tasks'] if t['answer']])
    prompt = ChatPromptTemplate.from_template(SUMMARIZE_ANSWER_PROMPT)
    chain = prompt | llm
    final_response = await invoke_llm_with_backoff(chain, {
        "input": state['messages'][0].content,
        "tasks_completed": task_completed,
    })

    print(f"\n[Final Summary]: {final_response.content}\n")

    return {"messages": state['messages'] + [final_response]}


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_from_orchestrator(state: AgentState) -> str:
    """Routes from the orchestrator to the assistant or to the end.
    """
    if state['iteration'] >= MAX_ITERATIONS:
        print("[Orchestrator]: Reached maximum iterations. Ending process.")
        return "summarize_answer"

    return "my_assistant" if state['active_task_id'] else "summarize_answer"


def route_from_assistant(state: AgentState) -> str:
    """Routing logic to the tools or back to the orchestrator.
    """
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task or not active_task['history']:
        return "orchestrator"

    last_message = active_task['history'][-1]
    # FIXED: Check if it's an AIMessage AND has tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Continue to tool execution if the last message has tool calls
        return "tool_execution"
    else:
        # Break the loop and go back to orchestrator
        return "tool_result_consolidator"


# ============================================================================
# GRAPH INITIALIZATION
# ============================================================================

def create_agent_graph(
    orchestrator_llm: BaseChatModel,
    assistant_llm: BaseChatModel,
    summarizer_llm: BaseChatModel,
    tools: List,
) -> StateGraph:
    """Creates and compiles the agent workflow graph.

    Args:
        orchestrator_llm: The LLM for the orchestrator node.
        assistant_llm: The LLM for the assistant node.
        summarizer_llm: The LLM for the summarizer node.
        tools: List of MCP tools available to the agent.

    Returns:
        Compiled StateGraph ready for execution.
    """
    assistant_llm_with_tools = assistant_llm.bind_tools(tools)

    # Define the Graph
    workflow = StateGraph(AgentState)

    # Add nodes to workflow
    workflow.add_node("orchestrator", partial(orchestrator, llm=orchestrator_llm))
    workflow.add_node(
        "my_assistant",
        partial(my_assistant, llm_with_tools=assistant_llm_with_tools, tools=tools),
    )
    workflow.add_node("tool_execution", partial(tool_execution, tools=tools))
    workflow.add_node("tool_result_consolidator", tool_result_consolidator)
    workflow.add_node("summarize_answer", partial(summarize_answer, llm=summarizer_llm))

    # Define edges
    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "my_assistant": "my_assistant",
            "summarize_answer": "summarize_answer",
        },
    )
    workflow.add_conditional_edges(
        "my_assistant",
        route_from_assistant,
        {
            "tool_execution": "tool_execution",
            "tool_result_consolidator": "tool_result_consolidator",
        },
    )
    workflow.add_edge("tool_execution", "my_assistant")
    workflow.add_edge("tool_result_consolidator", "orchestrator")
    workflow.add_edge("summarize_answer", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def main():
    """Main entry point for the agent application."""
    # Load environment variables
    load_dotenv()

    # Define the models for each role. Change these to use different models.
    # Example for Ollama: "ollama/llama3.1:8b"
    GEMINI_MODEL = 'gemini-2.5-flash'
    model = GEMINI_MODEL
    ORCHESTRATOR_MODEL = model
    ASSISTANT_MODEL = model
    SUMMARIZER_MODEL = model

    # Use the factory to create LLM instances
    llm_factory = LLMFactory()

    try:
        print(f"Initializing models...\n- Orchestrator: {ORCHESTRATOR_MODEL}\n- Assistant: {ASSISTANT_MODEL}\n- Summarizer: {SUMMARIZER_MODEL}")
        orchestrator_llm = llm_factory.create_llm(
            ORCHESTRATOR_MODEL, temperature=0
        )
        assistant_llm = llm_factory.create_llm(
            ASSISTANT_MODEL, temperature=0
        )
        summarizer_llm = llm_factory.create_llm(
            SUMMARIZER_MODEL, temperature=0
        )

    except ValueError as e:
        print(f"ERROR: Could not create LLM. {e}")
        return

    # Setup MCP tools
    print("\n--- Setting up tools ---\n")
    mcp_server_connection = {
        "my_server": {
            "url": "http://127.0.0.1:8000/sse",
            "transport": "sse",
        }
    }
    mcp_client = MultiServerMCPClient(mcp_server_connection)

    try:
        tools = await mcp_client.get_tools()
        print(f"Successfully connected to MCP server and retrieved {len(tools)} tools.")
    except Exception as e:
        print(f"ERROR: Failed to connect to MCP server or retrieve tools: {e}")
        return

    # Create the agent graph
    app = create_agent_graph(orchestrator_llm, assistant_llm, summarizer_llm, tools)

    # Run query
    user_input = (
        "Are there a vessel last visited port of Miami. Like within 10km radius?"
    )
    # Alternative query: "Where is the location of vessel named 'ADAMAS III' last visited port of Miami."
    print(f"\n--- Asking the graph: {user_input} ---")

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "tasks": [],
        "active_task_id": None,
        "iteration": 0,
        "scratchpad": [],
    }

    # Execute the agent workflow
    final_event = None
    async for event in app.astream(initial_state, config={"recursion_limit": 30}):
        final_event = event

    # Display final answer
    if final_event and "__end__" in final_event:
        final_state = final_event["__end__"]
        messages = final_state.get("messages", [])
        if len(messages) > 1:  # Ensure there is more than the initial user input
            final_answer = messages[-1]
            print(f"\n{'='*60}")
            print(f"FINAL ANSWER:")
            print(f"{'='*60}")
            print(final_answer.content)
            print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
