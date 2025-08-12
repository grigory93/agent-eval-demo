import asyncio
import json
from dotenv import load_dotenv
from pathlib import Path
from llama_index.llms.openai import OpenAI

from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate

from llama_index.core.workflow import (
    WorkflowTimeoutError,
)

from jllm_eval import run_jllm_eval 
from ragchecker_eval import run_ragchecker_eval
from text_to_sql_setup import index_all_tables, setup_data_index
from text_to_sql_workflows import TextToSQLWorkflow1, create_advanced_1_workflow, create_advanced_2_workflow



load_dotenv()


# eval dataset
EVAL_DATASET = {
    "B.I.G": {
        "query": "What was the year that The Notorious B.I.G was signed to Bad Boy?",
        "gt_answer": "The Notorious B.I.G was signed to Bad Boy Records in 1993.",
    },
    "BIG": {
        "query": "What was the year that The Notorious BIG was signed to Bad Boy?",
        "gt_answer": "The Notorious BIG was signed to Bad Boy Records in 1993.",
    },
    "best director": {
        "query": "Who won best director in the 1972 academy awards",
        "gt_answer": "William Friedkin won the Best Director award at the 1972 Academy Awards.",
    },
    "Preziosa": {
        "query": "What was the term of Pasquale Preziosa?",
        "gt_answer": "Pasquale Preziosa has been serving since 25 February 2013 and is currently in office as incumbent.",
    },
}

# list of eval queries
EVAL_QUERIES = [sample["query"] for sample in EVAL_DATASET.values()]

# gt answer for each query
def get_gt_answer(query):
    for key, value in EVAL_DATASET.items():
        if key.lower() in query.lower():
            return value["gt_answer"]
    else:
        raise ValueError(f"No GT answer found for query: {query}")


# Run
async def run_agent(agent: TextToSQLWorkflow1, message: str):
    try:
        response = await agent.run(query=message)
        return response
    except WorkflowTimeoutError:
        print("="*100)
        print("Workflow execution timed out!")
        print("="*100)
        return None


def run_workflow_evaluation(queries, workflow, output_file="workflow_results.json", save_to_file=True):
    """
    Run all queries through a workflow and assemble results into JSON format.
    
    Args:
        queries: List of query strings to run
        workflow: The TextToSQLWorkflow instance to use
        output_file: Path to save the JSON results
    
    Returns:
        Dict containing the assembled results
    """
    results = []
    responses = []
    retrieved_contexts = []
    
    for query_id, query in enumerate(queries):
        print(f"Processing query {query_id}: {query}")
        
        try:
            # Run the workflow
            result = asyncio.run(run_agent(workflow, query))
            
            if result:
                response = result.get("response")
                context = result.get("context", [])
                
                # Build retrieved_context array
                retrieved_context = []
                for doc_id, ctx in enumerate(context):
                    retrieved_context.append({
                        "doc_id": str(doc_id),
                        "text": ctx
                    })
                
                # Build the query result object
                query_result = {
                    "query_id": str(query_id),
                    "query": query,
                    "gt_answer": get_gt_answer(query),  # Leave empty as requested
                    "response": response.message.content if response else "",
                    "retrieved_context": retrieved_context
                }
            else:
                # Handle timeout or error case
                query_result = {
                    "query_id": str(query_id),
                    "query": query,
                    "gt_answer": "",
                    "response": "",
                    "retrieved_context": []
                }
                response = None
                
            results.append(query_result)
            responses.append(response)
            retrieved_contexts.append(retrieved_context)
        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            # Add error case to results
            query_result = {
                "query_id": str(query_id),
                "query": query,
                "gt_answer": "",
                "response": f"Error: {str(e)}",
                "retrieved_context": []
            }
            results.append(query_result)
            responses.append(None)
            retrieved_contexts.append(None)
    # Assemble final result
    final_result = {
        "results": results
    }
    
    # Save to JSON file
    if save_to_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    return final_result, responses, retrieved_contexts


def main():
    
    llm = OpenAI(model="gpt-4o-mini", temperature=0.5)
    engine, sql_database, obj_retriever, sql_retriever = setup_data_index(llm)
    
    # Response Synthesis Prompt
    response_synthesis_prompt_template1 = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt1 = PromptTemplate(
        response_synthesis_prompt_template1,
    )
    response_synthesis_prompt_template2 = (
        "Given an input question, synthesize a response from the query results. Only use the information from the SQL Response.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt2 = PromptTemplate(
        response_synthesis_prompt_template2,
    )

    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
        dialect=engine.dialect.name
    )

    # Create workflow with 30-second timeout
    workflow1_1 = create_advanced_1_workflow(sql_database, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt1, llm, timeout=3000.0, verbose=False)

    workflow1_2 = create_advanced_1_workflow(sql_database, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt2, llm, timeout=3000.0, verbose=False)

    vector_index_dict = index_all_tables(sql_database)
    workflow2_1 = create_advanced_2_workflow(sql_database, vector_index_dict, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt2, llm, timeout=3000.0, verbose=False)

    
    # Option 1: Use the evaluation function to run all queries and save to JSON
    print("Running workflow evaluation...")
    eval_file_name = "workflow1_1_results.json"
    workflow = workflow1_1
    results, responses, retrieved_contexts = run_workflow_evaluation(EVAL_QUERIES, workflow, eval_file_name, save_to_file=True)

    rag_results, eval_file = run_ragchecker_eval(eval_file_name)
    print(rag_results.metrics)
    jllm_results = run_jllm_eval(EVAL_QUERIES, responses)
    for eval_result in jllm_results:
        print(eval_result.query)
        print(eval_result.response)
        print(eval_result.eval_result)
        print(eval_result.phrase)
        print("-"*100)


if __name__ == "__main__":
    main()