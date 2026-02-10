# Live Tool Creation Demo — "Add a Tool in 60 Seconds"

> **GTC Act 5: Show that extensibility is real, not marketing.**

## Setup (Before Demo)

1. Have `example_tool_weather.py` ready but NOT loaded
2. Have the demo running normally with just `reachy_tools`
3. Have a text editor open with `example_tool_weather.py` visible

## Demo Script (60 seconds)

### Step 1: Show the Problem (10s)

> "Ask Reachy: 'What's the weather in San Jose?'"
>
> Reachy: "I don't have a weather tool, but I can tell you..."
> (It will hallucinate or say it can't check weather)

### Step 2: Write the Tool (30s)

Show the screen. Open `example_tool_weather.py`:

```python
def register_tools(registry, context=None):
    @registry.register(
        "Get the current weather for a city."
    )
    def weather(city: str = "San Jose") -> str:
        forecasts = {
            "san jose": "San Jose: 68°F, sunny.",
            "tokyo": "Tokyo: 52°F, partly cloudy.",
        }
        return forecasts.get(city.lower(), f"{city}: 65°F, clear.")
```

> "That's it. One function, one decorator, one return value.
> No framework rewrites. No pipeline changes. No redeployment."

### Step 3: Add It (10s)

Edit `config-gtc-demo.yaml`:

```yaml
external_tools:
  - reachy_tools
  - example_tool_weather    # ← add this line
```

Restart the assistant (Ctrl+C → re-run `./run-gtc-demo.sh`).

### Step 4: Use It (10s)

> "Ask Reachy: 'What's the weather in San Jose?'"
>
> Reachy: "San Jose: 68°F, sunny, light breeze from the west."

> "Same robot, new capability, 60 seconds. Now imagine this is
> your robot and the tool calls your API."

## Quick-Restart Command

```bash
# Ctrl+C the running demo, then:
./run-gtc-demo.sh
```

The script auto-detects vLLM and Reachy daemon — no need to restart them.

## Key Talking Points

1. **"One function, one decorator."** — `@registry.register("description")`
2. **"Type hints become the schema."** — `city: str` → LLM knows the parameter
3. **"No redeployment."** — Restart the assistant, not the model server
4. **"Works with any robot."** — Same plugin system, different `reachy_tools`
5. **"In production, that weather function calls your API."** — Swap the mock for real logic
