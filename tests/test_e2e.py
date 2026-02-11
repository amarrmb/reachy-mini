#!/usr/bin/env python3
"""
E2E test: natural language → vLLM (prompt-based tool calling) → reachy_tools → robot moves.

Matches the ACTUAL VoiceAssistant pipeline:
1. Tool descriptions go in the system prompt (not OpenAI tools API)
2. LLM responds with {"tool": "name", "args": {...}} as text
3. _try_parse_tool_call() extracts and executes

Usage (on Jetson, with vLLM on :8001 and reachy-mini-daemon on laptop):
    REACHY_HOST=<daemon-ip> PYTHONPATH=./bot python3 tests/test_e2e.py
"""

import json
import re
import sys
import time

from jetson_assistant.assistant.tools import ToolRegistry
from jetson_assistant.assistant.llm import create_llm, ToolCallResult

# ── Build the system prompt exactly as VoiceAssistant does ──

TOOL_SYSTEM_PROMPT_HEADER = (
    "You are a helpful voice assistant with tools. "
    "You MUST use your tools when they are relevant.\n\n"
    "To call a tool, respond with ONLY a JSON object like this:\n"
    '{"tool": "tool_name", "args": {"param": "value"}}\n\n'
    "RULES:\n"
    "- If a tool can answer the question, respond with ONLY the JSON tool call.\n"
    "- Do NOT add any text before or after the JSON.\n"
    "- Call ONLY ONE tool per response. If the user wants multiple things, "
    "pick the most important one.\n"
    "- If no tool is needed, just answer normally in 1-2 sentences.\n"
    "- No markdown or formatting.\n"
)

print("Setting up LLM + tools...", file=sys.stderr)

# Load tools
tools = ToolRegistry()
import reachy_tools
reachy_tools.register_tools(tools, context={"llm": None, "camera_pool": None})

# Build tool prompt (same logic as _apply_tool_system_prompt)
tool_lines = ["\nAvailable tools:"]
for defn in tools.definitions():
    func = defn["function"]
    name = func["name"]
    desc = func["description"]
    params = func["parameters"].get("properties", {})
    required = func["parameters"].get("required", [])

    if params:
        param_parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"{pname}: {ptype}{req} - {pdesc}")
        tool_lines.append(f"- {name}: {desc}")
        for pp in param_parts:
            tool_lines.append(f"    {pp}")
    else:
        tool_lines.append(f"- {name}: {desc}")

tool_lines.append(
    '\nExamples:\n'
    'User: "look left" → {"tool": "look", "args": {"direction": "left"}}\n'
    'User: "show me you\'re happy" → {"tool": "express", "args": {"emotion": "happy"}}\n'
    'User: "do a dance" → {"tool": "dance", "args": {"name": "random"}}\n'
    'User: "nod yes" → {"tool": "nod", "args": {"response": "yes"}}\n'
    'User: "go to sleep" → {"tool": "reachy_power", "args": {"action": "sleep"}}\n'
    'User: "Hello, how are you?" → Just reply normally, no tool needed.'
)

system_prompt = TOOL_SYSTEM_PROMPT_HEADER + "\n".join(tool_lines)

# Create LLM with the tool system prompt
llm = create_llm(
    backend="vllm",
    model="nvidia/Qwen2.5-VL-7B-Instruct-NVFP4",
    host="http://localhost:8001/v1",
    system_prompt=system_prompt,
)

tool_names = [d["function"]["name"] for d in tools.definitions()]
print(f"Tools: {tool_names}", file=sys.stderr)
print(f"System prompt length: {len(system_prompt)} chars", file=sys.stderr)


def try_parse_tool_call(text):
    """Parse tool call from LLM text output (same as core.py _try_parse_tool_call)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = text[start:j + 1]
                        try:
                            data = json.loads(json_str)
                            tool_name = data.get("tool") or data.get("name")
                            if tool_name:
                                args = data.get("args") or data.get("arguments") or {}
                                return tool_name, args
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return None, None


# ── Test cases ──

TEST_CASES = [
    ("look to the left", "look"),
    ("can you look right please?", "look"),
    ("look up at the ceiling", "look"),
    ("show me you're happy!", "express"),
    ("do a little dance for me", "dance"),
    ("nod yes", "nod"),
    ("shake your head no", "nod"),
    ("go to sleep now", "reachy_power"),
    ("wake up!", "reachy_power"),
    ("look back at me", "look"),
]

# ── Run ──

print(f"\n{'='*60}")
print(f"E2E: {len(TEST_CASES)} natural language → vLLM → tool → robot")
print(f"{'='*60}\n")

results = []
for query, expected_tool in TEST_CASES:
    print(f'USER: "{query}"')

    response = llm.generate(query, context=[])
    response_text = response.text.strip()
    print(f"  LLM: {response_text[:150]}")

    tool_name, tool_args = try_parse_tool_call(response_text)

    tool_result = None
    if tool_name:
        print(f"  Parsed tool: {tool_name}({tool_args})")
        try:
            tc = ToolCallResult(name=tool_name, arguments=tool_args)
            tool_result = tools.execute(tc)
            print(f"  Robot: {tool_result}")
        except Exception as e:
            tool_result = f"ERROR: {e}"
            print(f"  Error: {e}")
    else:
        print(f"  No tool call found in response")

    success = tool_name == expected_tool
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] expected={expected_tool} got={tool_name or 'NONE'}\n")

    results.append({
        "query": query,
        "expected": expected_tool,
        "called": tool_name,
        "result": tool_result,
        "llm_text": response_text[:120],
        "success": success,
    })

    time.sleep(1.5)

# ── Summary ──

print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
passed = sum(1 for r in results if r["success"])
total = len(results)
for r in results:
    mark = "PASS" if r["success"] else "FAIL"
    called = r["called"] or "NONE"
    print(f"  [{mark}] \"{r['query']}\" → {called} (expected: {r['expected']})")
    if not r["success"]:
        print(f"        LLM said: {r['llm_text']}")

print(f"\n{passed}/{total} passed")

reachy_tools.cleanup()
sys.exit(0 if passed == total else 1)
