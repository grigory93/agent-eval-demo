import re
from typing import Dict
import os
from pathlib import Path
import pandas as pd
import json

from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI

from llama_index.core.llms import ChatMessage
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from sqlalchemy import text
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.retrievers import SQLRetriever
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)


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


def setup_data_index(llm):
    # Extract Table Name and Summary from each Table
    data_dir = Path("./WikiTableQuestions/csv/200-csv")
    tableinfo_dir = Path("WikiTableQuestions_TableInfo")
    tableinfo_dir.mkdir(exist_ok=True)

    dfs = read_csv_files(data_dir)
    table_infos = get_table_infos(dfs, tableinfo_dir, llm)

    # Put Data in SQL Database¶
    db_path = "wiki_table_questions.db"
    engine = create_sql_database(dfs, tableinfo_dir, llm, db_path)

    # Object index, retriever, SQLDatabase

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

    return engine, sql_database, obj_retriever, sql_retriever