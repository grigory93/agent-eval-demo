import asyncio
import json
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from llama_index.core.retrievers import SQLRetriever

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
import re

from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import SQLDatabase, VectorStoreIndex

from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate

from llama_index.core.workflow import (
    WorkflowTimeoutError,
)
from sqlalchemy import text
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.retrievers import SQLRetriever
import os
from pathlib import Path
from typing import Dict

from jllm_eval import run_jllm_eval 
from ragchecker_eval import run_ragchecker_eval
from text_to_sql_workflows import TextToSQLWorkflow1, create_advanced_1_workflow, create_advanced_2_workflow



load_dotenv()

class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )

prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise. 
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """
prompt_tmpl = ChatPromptTemplate(
    message_templates=[ChatMessage.from_str(prompt_str, role="user")]
)



def read_csv_files(data_dir: Path):
    csv_files = sorted([f for f in data_dir.glob("*.csv")])
    dfs = []
    for csv_file in csv_files:
        print(f"processing file: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {csv_file}: {str(e)}")
    return dfs




def _get_tableinfo_with_index(tableinfo_dir: Path, idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )


def get_table_infos(dfs: list[pd.DataFrame], tableinfo_dir: Path, llm: OpenAI):
    table_names = set()
    table_infos = []
    for idx, df in enumerate(dfs):
        table_info = _get_tableinfo_with_index(tableinfo_dir, idx)
        if table_info:
            table_infos.append(table_info)
        else:
            while True:
                df_str = df.head(10).to_csv()
                table_info = llm.structured_predict(
                    TableInfo,
                    prompt_tmpl,
                    table_str=df_str,
                    exclude_table_name_list=str(list(table_names)),
                )
                table_name = table_info.table_name
                print(f"Processed table: {table_name}")
                if table_name not in table_names:
                    table_names.add(table_name)
                    break
                else:
                    # try again
                    print(f"Table name {table_name} already exists, trying again.")
                    pass

            out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
            json.dump(table_info.dict(), open(out_file, "w"))
        table_infos.append(table_info)

    return table_infos


# put data into sqlite db
# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


def create_sql_database(dfs: list[pd.DataFrame], tableinfo_dir: Path, llm: OpenAI, db_path: Path):
    # engine = create_engine("sqlite:///:memory:")
    engine = create_engine(f"sqlite:///{db_path}")
    metadata_obj = MetaData()
    for idx, df in enumerate(dfs):
        tableinfo = _get_tableinfo_with_index(tableinfo_dir, idx)
        print(f"Creating table: {tableinfo.table_name}")
        create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)
    return engine






# Advanced Capability 2: Text-to-SQL with Query-Time Row Retrieval (along with Table Retrieval)
def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "data/table_index_dir"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    table_index_exceptions = ["french_airports_usage_summary", "norwegian_club_performance_statistics","filmography_of_diane","kodachrome_film_types_and_dates","missing_persons_case_summary","bbc_radio_service_costs_comparison","binary_encoding_probabilities",]
    vector_index_dict = {}
    engine = sql_database.engine
    for table_name in sql_database.get_usable_table_names():
        if table_name in table_index_exceptions:
            print(f"Skipping table: {table_name}")
            continue
        print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            # get all rows from table
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            # index each row, put into vector store index
            nodes = [TextNode(text=str(t)) for t in row_tups]

            # put into vector store index (use OpenAIEmbeddings by default)
            index = VectorStoreIndex(nodes)

            # save index
            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            # load index
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )
        vector_index_dict[table_name] = index

    return vector_index_dict



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


def preprocess_response(response):
    
    if "query did not return any results" in response.lower():
        # INSERT_YOUR_CODE
        # Remove the last sentence from the string
        import re
        # Split into sentences using regex, keep punctuation
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 1:
            return ' '.join(sentences[:-1])
        else:
            return response
    else:
        return response

GT_ANSWER_LIST = {
    "B.I.G": "The Notorious B.I.G was signed to Bad Boy Records in 1993.",
    "BIG": "The Notorious BIG was signed to Bad Boy Records in 1993.",
    "best director": "William Friedkin won the Best Director award at the 1972 Academy Awards.",
    "Preziosa": "Pasquale Preziosa has been serving since 25 February 2013 and is currently in office as incumbent.",
}

def get_gt_answer(query):
    for key, value in GT_ANSWER_LIST.items():
        if key.lower() in query.lower():
            return value
    else:
        raise ValueError(f"No GT answer found for query: {query}")

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


if __name__ == "__main__":
    # Extract Table Name and Summary from each Table
    data_dir = Path("./WikiTableQuestions/csv/200-csv")
    tableinfo_dir = "WikiTableQuestions_TableInfo"
    tableinfo_dir = Path(tableinfo_dir)
    tableinfo_dir.mkdir(exist_ok=True)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.5)

    dfs = read_csv_files(data_dir)
    table_infos = get_table_infos(dfs, tableinfo_dir, llm)

    # Put Data in SQL Database¶
    db_path = "wiki_table_questions.db"
    engine = create_sql_database(dfs, tableinfo_dir, llm, db_path)

    # Object index, retriever, SQLDatabase
    sql_database = SQLDatabase(engine)

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
    ]  # add a SQLTableSchema for each table

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    # SQLRetriever + Table Parser
    sql_retriever = SQLRetriever(sql_database)

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
    print(text2sql_prompt.template)

    # Create workflow with 30-second timeout
    workflow1_1 = create_advanced_1_workflow(sql_database, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt1, llm, timeout=3000.0, verbose=False)

    workflow1_2 = create_advanced_1_workflow(sql_database, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt2, llm, timeout=3000.0, verbose=False)

    vector_index_dict = index_all_tables(sql_database)
    workflow2_1 = create_advanced_2_workflow(sql_database, vector_index_dict, obj_retriever, text2sql_prompt, sql_retriever, response_synthesis_prompt2, llm, timeout=3000.0, verbose=False)

    queries = [
        "What was the year that The Notorious B.I.G was signed to Bad Boy?",
        "What was the year that The Notorious BIG was signed to Bad Boy?",
        "Who won best director in the 1972 academy awards",
        "What was the term of Pasquale Preziosa?",
    ]

    
    # Option 1: Use the evaluation function to run all queries and save to JSON
    print("Running workflow evaluation...")
    eval_file_name = "workflow1_1_results.json"
    workflow = workflow1_1
    results, responses, retrieved_contexts = run_workflow_evaluation(queries, workflow, eval_file_name, save_to_file=False)

    # rag_results, eval_file = run_ragchecker_eval(eval_file_name)
    # print(rag_results.metrics)
    jllm_results = run_jllm_eval(queries, responses)
    for eval_result in jllm_results:
        print(eval_result.query)
        print(eval_result.response)
        print(eval_result.eval_result)
        print(eval_result.phrase)
        print("-"*100)

    
    # Option 2: Or if you want to run manually and still see individual outputs:
    # for query in queries:
    #     print("="*100)
    #     print(f"Running {workflow.__class__.__name__} with query: {query}")
    #     result = asyncio.run(run_agent(workflow, query))
    #     response = result.get("response")
    #     context = result.get("context")
    #     if response:
    #         print("="*100)
    #         print(f"Query: {query}")
    #         print(f"Response: {response.message.content}")
    #         print(f"Context: {context}")
    #     else:
    #         print("Response: None")
    #     print("="*100)
    
    # You can also run evaluations for different workflows:
    # results_1_2 = run_workflow_evaluation(workflow1_2, queries, "workflow1_2_results.json")
    # results_2_1 = run_workflow_evaluation(workflow2_1, queries, "workflow2_1_results.json")

    

