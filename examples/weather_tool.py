"""
Example external tool — Weather report.

This is the "add a tool in 60 seconds" demo for GTC.
10 lines of actual code. Drop it in, restart, use it.

Usage:
    jetson-assistant assistant --external-tools reachy_tools,example_tool_weather ...

Or in config.yaml:
    external_tools:
      - reachy_tools
      - example_tool_weather
"""

from typing import Annotated


def register_tools(registry, context=None):
    @registry.register(
        "Get the current weather for a city. "
        "Use when the user asks about weather or temperature."
    )
    def weather(
        city: Annotated[str, "City name (e.g., 'San Jose', 'Tokyo')"] = "San Jose",
    ) -> str:
        # In production, call a real API. For the demo, return realistic data.
        forecasts = {
            "san jose": "San Jose: 68°F, sunny, light breeze from the west.",
            "tokyo": "Tokyo: 52°F, partly cloudy, chance of rain this evening.",
            "london": "London: 45°F, overcast, typical.",
            "paris": "Paris: 50°F, clear skies, beautiful day.",
        }
        return forecasts.get(city.lower().strip(), f"{city}: 65°F, clear skies.")
