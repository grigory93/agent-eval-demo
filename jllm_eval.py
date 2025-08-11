import json
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage

judge_llm_prompt_template = """
    You are a helpful and unbiased judge. Analyze the following answer to a user question and determine if the answer contains claim that the answer can not be found, at least partially. Respond with True if the answer contains such claim, otherwise respond with False. Also include the phrase from the answer that contains the claim.

    Question: {query}
    Answer: {response}

    Response:
"""

class JudgeLLMResponse(BaseModel):
    '''Judge LLM Response'''
    query: str = Field(description="The user question")
    response: str = Field(description="The LLM answer")
    eval_result: bool = Field(description="Whether the LLM answer contains a claim that the answer can not be found, at least partially")
    phrase: str = Field(description="The phrase that the LLM used to make the claim")
    
def run_jllm_eval(queries, responses):
    results = []

    llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
    jllm = llm.as_structured_llm(JudgeLLMResponse)
    for query, response in zip(queries, responses):
        input_msg = ChatMessage.from_str(judge_llm_prompt_template.format(query=query, response=response.message.content))
        jllm_response = jllm.chat([input_msg])
        result_json = jllm_response.message.content
        result_obj = JudgeLLMResponse.model_validate(json.loads(result_json))
        results.append(result_obj)
    return results

def main():
    query = "What was the year that The Notorious BIG was signed to Bad Boy?"
    class Response:
        def __init__(self, message_content):
            class Message:
                def __init__(self, content):
                    self.content = content
            self.message = Message(message_content)
    response = Response("The query did not return any results regarding the year that The Notorious BIG was signed to Bad Boy Records. However, it is widely known that he was signed in 1993.")
    results = run_jllm_eval([query], [response])
    # Extract the first JudgeLLMResponse object from the results
    if results and len(results) > 0:
        print(json.dumps(results[0].model_dump(), indent=2, ensure_ascii=False))
    else:
        print("No results returned from run_jllm_eval.")

if __name__ == "__main__":
    main()