import asyncio
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract two numbers and returns the difference"""
    return a - b

def divide(a: float, b: float) -> float:
    """Divide two numbers and returns the quotient"""
    return a / b

def power(a: float, b: float) -> float:
    """Raise a to the power of b and return the result"""
    return a ** b



# Agent
def create_agent(tools: list[Callable], system_prompt: str):
    llm = OpenAI(model="gpt-4.1", temperature=.5)
    workflow = FunctionAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True,
    )

    return workflow

# Run
async def run_agent(agent: FunctionAgent, message: str):
    response = await agent.run(user_msg=message)
    print("="*100)
    print("Response:")
    print(response)
    print("="*100)


if __name__ == "__main__":
    workflow = create_agent(
        tools=[multiply, add, subtract, divide, power],
        system_prompt="You are an expert in math and you are given a task to solve a math problem. You are also given a list of tools to use to solve the problem. You are to use the tools to solve the problem and return the answer.",
    )
    asyncio.run(run_agent(workflow, "What is 100 * 100?"))
    asyncio.run(run_agent(workflow, "What is 100 + 100?"))
    asyncio.run(run_agent(workflow, "What is 100 - 100?"))
    asyncio.run(run_agent(workflow, "What is 100 / 100?"))
    asyncio.run(run_agent(workflow, "What is 100 ** 2?"))
    asyncio.run(run_agent(workflow, "What is 20+(2*4)?"))