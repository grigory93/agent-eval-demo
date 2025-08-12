#!/usr/bin/env python3
"""
CLI interface for running text-to-SQL agent evaluations.
"""

import click
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate

from text_to_sql_setup import index_all_tables, setup_data_index
from text_to_sql_workflows import create_advanced_1_workflow, create_advanced_2_workflow
from text_to_sql_agent import run_workflow_evaluation, EVAL_QUERIES
from jllm_eval import run_jllm_eval
from ragchecker_eval import run_ragchecker_eval

# Load environment variables
load_dotenv()


def setup_workflows(llm):
    """Setup and return all available workflows."""
    engine, sql_database, obj_retriever, sql_retriever = setup_data_index(llm)
    
    # Response Synthesis Prompts
    response_synthesis_prompt_template1 = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt1 = PromptTemplate(response_synthesis_prompt_template1)
    
    response_synthesis_prompt_template2 = (
        "Given an input question, synthesize a response from the query results. "
        "Only use the information from the SQL Response.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt2 = PromptTemplate(response_synthesis_prompt_template2)

    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
        dialect=engine.dialect.name
    )

    # Create workflows
    workflows = {}
    
    # Workflow 1.1 - Basic advanced workflow
    workflows["1.1"] = create_advanced_1_workflow(
        sql_database, obj_retriever, text2sql_prompt, sql_retriever, 
        response_synthesis_prompt1, llm, timeout=3000.0, verbose=False
    )
    
    # Workflow 1.2 - Advanced workflow with restricted context
    workflows["1.2"] = create_advanced_1_workflow(
        sql_database, obj_retriever, text2sql_prompt, sql_retriever, 
        response_synthesis_prompt2, llm, timeout=3000.0, verbose=False
    )
    
    # Workflow 2.1 - Advanced workflow with vector indexing
    vector_index_dict = index_all_tables(sql_database)
    workflows["2.1"] = create_advanced_2_workflow(
        sql_database, vector_index_dict, obj_retriever, text2sql_prompt, 
        sql_retriever, response_synthesis_prompt2, llm, timeout=3000.0, verbose=False
    )
    
    return workflows


@click.command()
@click.option(
    '--workflow', 
    type=click.Choice(['1.1', '1.2', '2.1']),
    default='1.1',
    help='Version of the workflow to run (default: 1.1)'
)
@click.option(
    '--ragchecker/--no-ragchecker',
    default=False,
    help='Run RAGChecker evaluation after workflow execution'
)
@click.option(
    '--jllm/--no-jllm',
    default=False,
    help='Run JLLM evaluation after workflow execution'
)
@click.option(
    '--output',
    type=str,
    help='Output filename for results (default: workflow{version}_results.json)'
)
@click.option(
    '--model',
    type=str,
    default='gpt-4o-mini',
    help='OpenAI model to use (default: gpt-4o-mini)'
)
@click.option(
    '--temperature',
    type=float,
    default=0.5,
    help='Temperature for the LLM (default: 0.5)'
)
@click.option(
    '--verbose/--no-verbose',
    default=False,
    help='Enable verbose output'
)
def eval(workflow, ragchecker, jllm, output, model, temperature, verbose):
    """
    Run text-to-SQL agent evaluations.
    
    This command runs the specified workflow version on the evaluation dataset
    and optionally performs RAGChecker and JLLM evaluations on the results.
    
    Examples:
    
    \b
    # Run workflow 1.1 with default settings
    python cli.py eval --workflow 1.1
    
    \b
    # Run workflow 2.1 with both evaluations
    python cli.py eval --workflow 2.1 --ragchecker --jllm
    
    \b
    # Run workflow 1.2 with custom output file
    python cli.py eval --workflow 1.2 --output my_results.json --ragchecker
    """
    click.echo(f"🚀 Starting evaluation with workflow {workflow}")
    
    # Initialize LLM
    llm = OpenAI(model=model, temperature=temperature)
    
    # Setup workflows
    if verbose:
        click.echo("Setting up workflows...")
    workflows = setup_workflows(llm)
    
    # Get selected workflow
    selected_workflow = workflows[workflow]
    
    # Determine output filename
    if output is None:
        output = f"workflow{workflow.replace('.', '_')}_results.json"
    
    click.echo(f"📝 Output will be saved to: {output}")
    
    # Run workflow evaluation
    click.echo(f"🔄 Running workflow {workflow} evaluation...")
    results, responses, retrieved_contexts = run_workflow_evaluation(
        EVAL_QUERIES, 
        selected_workflow, 
        output, 
        save_to_file=True
    )
    
    click.echo(f"✅ Workflow evaluation completed. Results saved to {output}")
    
    # Run evaluations if requested
    eval_results = {}
    
    if ragchecker:
        click.echo("🔍 Running RAGChecker evaluation...")
        try:
            rag_results, eval_file = run_ragchecker_eval(output)
            eval_results['ragchecker'] = {
                'metrics': rag_results.metrics,
                'eval_file': eval_file
            }
            click.echo(f"✅ RAGChecker evaluation completed. Results saved to {eval_file}")
            
            # Display RAGChecker metrics
            click.echo("\n📊 RAGChecker Metrics:")
            for category, metrics in rag_results.metrics.items():
                click.echo(f"  {category}:")
                for metric, value in metrics.items():
                    click.echo(f"    {metric}: {value}")
                    
        except Exception as e:
            click.echo(f"❌ RAGChecker evaluation failed: {str(e)}", err=True)
    
    if jllm:
        click.echo("🤖 Running JLLM evaluation...")
        try:
            jllm_results = run_jllm_eval(EVAL_QUERIES, responses)
            eval_results['jllm'] = [result.model_dump() for result in jllm_results]
            
            click.echo("✅ JLLM evaluation completed.")
            
            # Display JLLM results summary
            click.echo("\n📊 JLLM Evaluation Results:")
            cannot_answer_count = sum(1 for result in jllm_results if result.eval_result)
            click.echo(f"  Queries where answer claimed 'cannot be found': {cannot_answer_count}/{len(jllm_results)}")
            
            if verbose:
                click.echo("\n📋 Detailed JLLM Results:")
                for i, result in enumerate(jllm_results):
                    click.echo(f"  Query {i+1}: {result.query}")
                    click.echo(f"    Claims cannot answer: {result.eval_result}")
                    if result.eval_result:
                        click.echo(f"    Phrase: {result.phrase}")
                    click.echo()
                    
        except Exception as e:
            click.echo(f"❌ JLLM evaluation failed: {str(e)}", err=True)
    
    # Summary
    click.echo(f"\n🎉 Evaluation complete!")
    click.echo(f"  Workflow: {workflow}")
    click.echo(f"  Queries processed: {len(EVAL_QUERIES)}")
    click.echo(f"  Results file: {output}")
    
    if ragchecker and 'ragchecker' in eval_results:
        click.echo(f"  RAGChecker evaluation: {eval_results['ragchecker']['eval_file']}")
    
    if jllm and 'jllm' in eval_results:
        cannot_answer = sum(1 for result in eval_results['jllm'] if result['eval_result'])
        click.echo(f"  JLLM evaluation: {cannot_answer}/{len(eval_results['jllm'])} queries claimed unanswerable")


@click.group()
def cli():
    """
    Text-to-SQL Agent Evaluation CLI
    
    This tool provides a command-line interface for running and evaluating
    text-to-SQL agents with different workflow configurations.
    """
    pass


cli.add_command(eval)


if __name__ == '__main__':
    cli()
