import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from sample_tools.tools.elastic_tool import (
    get_all_elastic_indices,
    get_elastic_index_mapping,
    run_elastic_query,
)

app = FastAPI(
    title="Gemini Tool Server",
    description="A server for exposing tools to Gemini.",
    version="0.1.0",
)

mcp = FastMCP("combined tools")

mcp.add_tool(get_all_elastic_indices)
mcp.add_tool(get_elastic_index_mapping)
mcp.add_tool(run_elastic_query)

app.mount("/", mcp.sse_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)