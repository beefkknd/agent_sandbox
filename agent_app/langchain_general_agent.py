import sys
import asyncio
import operator
import json
from typing import List, Dict, Annotated, Any, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configuration
MAX_ITERATIONS = 9  # Max rounds of the orchestrator

# Prompt Templates
ORCHESTRATOR_PROMPT = """
You are a master orchestrator agent. Your job is to methodically break down a user's complex request into a sequence of concise, actionable tasks for a team of worker agents.

User's Goal:
{input}

Established Facts (From completed tasks):
{tasks_completed}

Your Role & Goal & Facts: Deeply understand the user's goal and compare it against the "Established Facts".
1. Evaluate Sufficiency: Ask yourself: "Do the Established Facts provide a direct and useful answer to the user's Likely Intent?" Avoid chasing unnecessary details.
2. Identify the Next Step: If and only if the goal is NOT sufficiently met, Identify the single most critical piece of information needed next.
3. Formulate Task: State the next task as a clear, direct command for the worker agent.

**Output Format:**
* If more work is needed: Provide ONLY the single, imperative command for the next task. Do not include explanations, reasoning, or conversational text.
  - Bad: "Okay, so to find the weather, we need to find the location first..."
  - Good: "Find the user's current location."
* If the goal is met: Respond with ONLY the word "DONE".

Example (Next Task Command or "DONE"):
"""

MY_ASSISTANT_PROMPT = """
You are acting as a =Principal Engineer, a seasoned expert in solving complex technical problems. Your primary function is to achieve the

input}
User Goal:
{task}

Current Task:
{task_question}

Your Scratchpad (Recent Tool History):
{scratchpad}

Intelligent Analysis:
* Think beyond the literal: Your role is to be a "smart" agent, not just literal interpreter.
* Derive and Infer: If a user's query requires inference that is not directly stated, fold in the data, you must formulate a plan to derive or infer.
  * Example of Derivation: To find a vessel's "last visited port", you can infer it by finding the nearest known port to the vessel's current coordinates using

Available Tools:
{tool_names}

Instructions:
You are in a loop to solve the current task. You must continue to execute actions until the task is fully resolved.
1. *Thought*: Adopt the mindset of a Principal Engineer. Analyze the task, your scratchpad, and available tools. Formulate a clear, step-by-step plan for your next action.
2. a. *Tool Call(s)*: If you determine that more information is required, call one or more tools. The tool name(s) you MUST use be from the "Available Tools" list.
    b. *Provide Final Answer*: Once you have all the necessary data and are confident in your conclusion, provide the final answer as a clear, comprehensive text.
"""

SUMMARIZE_ANSWER_PROMPT = """
You are a final report generator. Your job is to provide a concise, final answer to the user's original question based on the work done by an agent.

User's Original Question:
{input}

Completed Sub-Tasks and Their Answers:
{tasks_completed}

Final Answer:
"""

# Usage Management
class Task(TypedDict):
    id: int
    question: str
    history: Annotated[List[BaseMessage], add_messages]
    answer: str

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_task_id: int
    tasks: List[Task]
    iteration: int
    scratchpad: Annotated[List[ToolMessage], add_messages]

# Global context for my assistant's tool calls
scratchpad: Annotated[List[ToolMessage], add_messages]

def _format_scratchpad_for_prompt(scratchpad: List[ToolMessage]) -> str:
    """Format the last 5 tool messages from the scratchpad into a
    de-duplicated string for the LLM prompt.
    """
    if not scratchpad:
        return "No tools used yet."

    # 1. Get the last tool
    recent_tools = scratchpad[-5:]

    run_text_lines = []
    for tool_msg in recent_tools:
        run_text_lines.append(f"Tool: {tool_msg.name}")
        run_text_lines.append(f"Input: {tool_msg.tool_input}")
        run_text_lines.append(f"Result: {tool_msg.content}")

    # A FIXME: The lines to de-duplicating the scratchpad night cause loss
    # of important context if a tool is called multiple times to check state.
    # This can be commented out if it causes issues.
    unique_lines = list(dict.fromkeys(run_text_lines))
    return "\n".join(unique_lines)

# --- Node Functions ---
async def orchestrator(state: AgentState, llm: ChatOpenAI):
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

    response = await chain.ainvoke({
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

import sys
from langchain_core.callbacks import BaseCallbackHandler


async def my_assistant(state: AgentState, llm_with_tools: ChatOpenAI, tools: List) -> Dict:
    """The worker agent that uses tools to solve a single task in a loop.
    """
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return state

    print(f"\n   [Assistant] Task: \"{active_task['question']}\"")

    tool_names = ", ".join([t.name for t in tools])

    scratchpad_content = _format_scratchpad_for_prompt(state['scratchpad'], [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", MY_ASSISTANT_PROMPT),
        MessagesPlaceholder(variable_name="history"),
    ])

    chain = prompt | llm_with_tools

    history = active_task['history'] + state['messages'][1:]  # Exclude the initial user input message

    response_message = await chain.ainvoke({
        "input": state['messages'][0].content,
        "task_question": active_task['question'],
        "history": history,
        "tool_names": tool_names,
        "scratchpad": scratchpad_content,
    })

    if response_message.tool_calls:
        tool_name = ", ".join([tc['name'] for tc in response_message.tool_calls])
        print(f"   [Assistant]: Calling tool(s): {tool_name}")
    else:
        print(f"   [Assistant]: Reaching a final answer.")
        
    active_task['history'].append(response_message)
    return{"tasks": state['tasks']}


async def execute_tool_call(tool_call, tools_map) -> Dict:
    """Executes tools called by the assistant in parallel and appends results.
    """
    tool_name = tool_call['name']
    try:
        tool_to_call = tools_map[tool_name]
        output = await tool_to_call.ainvoke(tool_call['args'])
        return ToolMessage(content=str(output), tool_call_id=tool_call['id'], name=tool_name)
    except Exception as e:
        error_payload = json.dumps({
            "error": f"Tool Execution Error: '{type(e).__name__}'",
            "details": str(e),
            "tool_name": tool_name,
            "tool_args": tool_call['args'],
        })
        return ToolMessage(content=error_payload, tool_call_id=tool_call['id'], name=tool_name)


async def tool_execution(state: AgentState, tools: List) -> Dict:
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return {}

    last_message = active_task['history'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_map = {tool.name: tool for tool in tools}
    tasks = [execute_tool_call(tc, tool_map) for tc in last_message.tool_calls]
    tool_results = await asyncio.gather(*tasks)
    active_task['history'].extend(tool_results)

    return {
        "tasks": state['tasks'],
        "scratchpad": state['scratchpad'] + tool_results,
    }


async def tool_result_consolidator(state: AgentState) -> Dict:
    """Consolidates tool results into the assistant's history.
    """
    active_task = next((t for t in state['tasks'] if t['id'] == state['active_task_id']), None)
    if not active_task:
        return {}
    
    print(f"   [Consolidator] Task: \"{active_task['question']}\"")

    final_answer_message = active_task['history'][-1]
    if isinstance(final_answer_message, AIMessage) and not final_answer_message.tool_calls:
        final_answer = final_answer_message.content
        active_task['answer'] = final_answer
        print(f"   [Orchestrator]: Review Task answered with: \"{final_answer}\"")
    else:
        active_task['answer'] = "Error: Assistant loop ended without final answer provided."

    return {
        "tasks": state['tasks'],
        "active_task_id": None,
    }

async def summarize_answer(state: AgentState, llm: ChatOpenAI) -> Dict:
    print(f"\n----------------------------------------\n" 
          f"--Summarizing Final Answer\n"
          f"----------------------------------------")
    task_completed = "\n".join([f"Task: {t['question']}\nAnswer: {t['answer']}" for t in state['tasks'] if t['answer']])
    prompt = ChatPromptTemplate.from_template(SUMMARIZE_ANSWER_PROMPT)
    chain = prompt | llm
    final_response = await chain.ainvoke({
        "input": state['messages'][0].content,
        "tasks_completed": task_completed,  
    })
    return {"messages": state['messages'] + [final_response], "summary_answer": final_response.content}



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
    if not isinstance(last_message, AIMessage) and last_message.tool_calls:
        # continue to tool execution if the last message has tool calls
        return "tool_execution"
    else:
        # break the loop and go back to orchestrator
        return "tool_result_consolidator"

   

# --- Invoke the graph ---
async def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    print("\n--- Setting up tools ---\n")
    mcp_server_connection = {"my_server": "http://127.0.0.1:8100/sse", "transport": "sse"}
    mcp_client = MultiServerMCPClient(mcp_server_connection)
    try:
        await mcp_client.get_tools()
        tools = mcp_client.tools
        print(f"Successfully connected to MCP server and retrieved {len(tools)} tools.")
    except Exception as e:
        print(f"Failed to connect to MCP server or retrieve tools: {e}")
        return
    
    llm_with_tools = llm.with_tools(mcp_client.tools, verbose=False) 

    # --- Define the Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("orchestrator", lambda state: asyncio.run(orchestrator(state, llm)))
    workflow.add_node("my_assistant", lambda state: asyncio.run(my_assistant(state, llm_with_tools, tools)))
    workflow.add_node("tool_execution", lambda state: asyncio.run(tool_execution(state, tools)))
    workflow.add_node("tool_result_consolidator", lambda state: asyncio.run(tool_result_consolidator(state)))
    workflow.add_node("summarize_answer", lambda state: asyncio.run(summarize_answer(state, llm)))

    # --- Define Edges ---
    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edge("orchestrator", route_from_orchestrator, {
        "my_assistant": "my_assistant",
        "summarize_answer": "summarize_answer",
    })
    workflow.add_conditional_edge("my_assistant", route_from_assistant, {
        "tool_execution": "tool_execution",
        "tool_result_consolidator": "tool_result_consolidator", 
    })
    workflow.add_conditional_edge("tool_execution", "my_assistant")
    workflow.add_edge("summarize_answer", END)

    app = workflow.compile()

    # "Do I have ala data for vessel named 'ADAMAS III' or 'LARRY B WHIPPLE'?"
    user_input = "Are there a vessel named last visited port of Miami. Like within 10km radius?"
    user_input = "Where is the location of vessel named 'ADAMAS III' last visited port of Miami."
    print("\n--- Asking the graph: (user_input) ---")

    initial_state = {"messages": [HumanMessage(content=user_input)], "tasks": [], "active_task_id": None, "iteration": 0, "scratchpad": []}

    # The event stream is no longer needed for logging, but we still need to consume it.
    # The last event from an astream on a compiled graph is the final state.
    # This final state has the summary_answer.
    final_event = None
    async for event in app.astream(initial_state, config={"recursion_limit": 30}):
        final_event = event

    print(f"\nThe last event from an astream on a compiled graph is the final state.")
    print(final_event['summary_answer']['messages'][0].content)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error occurred: {e}")