#!/usr/bin/env python3
"""
A minimal tool-using AI agent.

The model doesn't just answer: it decides which tool to call, the program
executes it, the result goes back to the model, and this repeats until the
task is done (the "agent loop").

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key
    python agent.py                       # interactive
    python agent.py "Make a file primes.py that prints the first 20 primes, then run it"

Add your own tool by writing a function and decorating it with @tool(...).
"""

import ast
import json
import operator
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "./workspace")).resolve()
MAX_STEPS = 15          # safety cap on loop iterations
AUTO_APPROVE = False    # True = never ask before writing files / running code

WORKSPACE.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------
TOOLS = {}  # name -> {"fn": callable, "schema": dict, "risky": bool}


def tool(name, description, properties, required, risky=False):
    """Register a function as a tool the model can call."""
    def wrap(fn):
        TOOLS[name] = {
            "fn": fn,
            "risky": risky,
            "schema": {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return fn
    return wrap


def safe_path(rel):
    """Resolve a path and make sure it stays inside the workspace."""
    p = (WORKSPACE / rel).resolve()
    if p != WORKSPACE and WORKSPACE not in p.parents:
        raise ValueError(f"Path escapes workspace: {rel}")
    return p


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool("calculator", "Evaluate an arithmetic expression exactly, e.g. '(12.5*4)/3'.",
      {"expression": {"type": "string"}}, ["expression"])
def calculator(expression):
    return str(_eval(ast.parse(expression, mode="eval").body))


@tool("current_time", "Get the current local date and time.", {}, [])
def current_time():
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


@tool("list_files", "List files in the workspace directory (or a subfolder).",
      {"path": {"type": "string", "description": "Relative folder, default '.'"}}, [])
def list_files(path="."):
    p = safe_path(path)
    items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(items) or "(empty)"


@tool("read_file", "Read a text file from the workspace.",
      {"path": {"type": "string"}}, ["path"])
def read_file(path):
    return safe_path(path).read_text()[:20000]


@tool("write_file", "Create or overwrite a text file in the workspace.",
      {"path": {"type": "string"}, "content": {"type": "string"}},
      ["path", "content"], risky=True)
def write_file(path, content):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


@tool("run_python", "Run a Python script that exists in the workspace and return its output.",
      {"path": {"type": "string"}}, ["path"], risky=True)
def run_python(path):
    p = safe_path(path)
    r = subprocess.run([sys.executable, str(p)], cwd=WORKSPACE,
                       capture_output=True, text=True, timeout=30)
    out = (r.stdout + r.stderr).strip()
    return f"exit code {r.returncode}\n{out[:8000]}"


@tool("http_get", "Fetch the text content of a URL (first 8000 chars).",
      {"url": {"type": "string"}}, ["url"])
def http_get(url):
    if not url.startswith(("http://", "https://")):
        raise ValueError("Only http(s) URLs allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "tool-agent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")[:8000]


# --------------------------------------------------------------------------
# Tool execution (with approval gate)
# --------------------------------------------------------------------------
def execute(name, args):
    """Run a tool. Returns (text, is_error)."""
    spec = TOOLS.get(name)
    if not spec:
        return f"Unknown tool: {name}", True
    if spec["risky"] and not AUTO_APPROVE:
        print(f"\n  ⚠  Agent wants to run {name}({json.dumps(args)[:300]})")
        if input("  Allow? [y/N] ").strip().lower() != "y":
            return "User denied this action.", True
    try:
        return str(spec["fn"](**args)), False
    except Exception as e:  # report errors to the model so it can recover
        return f"{type(e).__name__}: {e}", True


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------
SYSTEM = (
    "You are a capable assistant that completes tasks by using tools. "
    "Plan briefly, call tools to take real actions or get real data, check "
    "results, fix errors, and stop when the task is done. Never claim you did "
    "something you did not do with a tool. Work only inside the workspace."
)


def run_agent(client, messages, verbose=True):
    for step in range(MAX_STEPS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=[t["schema"] for t in TOOLS.values()],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        for block in resp.content:
            if block.type == "text" and block.text.strip() and verbose:
                print(f"\n🤖 {block.text}")

        if resp.stop_reason != "tool_use":
            return messages  # model is finished

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"  🔧 {block.name}({json.dumps(block.input)[:200]})")
                out, is_err = execute(block.name, block.input)
                if verbose:
                    print(f"     → {out[:200]}{'…' if len(out) > 200 else ''}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})

    print("\n(stopped: reached max steps)")
    return messages


def main():
    import anthropic
    client = anthropic.Anthropic()
    messages = []
    print(f"Agent ready. Workspace: {WORKSPACE}\nTools: {', '.join(TOOLS)}")

    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        run_agent(client, messages)
        return

    while True:
        try:
            task = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in {"exit", "quit", ""}:
            break
        messages.append({"role": "user", "content": task})
        run_agent(client, messages)


if __name__ == "__main__":
    main()
