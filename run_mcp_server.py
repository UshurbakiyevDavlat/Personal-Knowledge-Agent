import sys
import os
from agent_server.server import mcp

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mcp.run(transport="sse", host="0.0.0.0", port=8000)
