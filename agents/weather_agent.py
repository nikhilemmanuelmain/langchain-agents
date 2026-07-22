"""A first LangChain agent based on the official weather quickstart."""

import os

from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # This is the deliberately fake weather tool used by the quickstart.
    return f"It's always sunny in {city}!"


def create_weather_agent():
    """Create an agent that can call the weather tool."""
    return create_agent(
        model="openai:gpt-5.5",
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )


def main() -> None:
    """Ask the example agent a weather question and print its answer."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running this agent."
        )

    agent = create_weather_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Luxembourg?",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
