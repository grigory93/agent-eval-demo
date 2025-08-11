# Multi-Agentic App and Eval

## Overview

We construct an agent that supports clients or users via chat (or other protocol like REST API) by utilizing Text-to-SQL workflow to answer questions. 
The agent interacts with clients in natural language by converting text to SQL, executing SQL queries, then translating query results to text. 
All takes place on arbitrary structured data sources.
The chat and API interface are not part of the agent, rather the agent can fit into layered application based on the requirements and use cases.

## Frameworks

 * LlamaIndex for constructing agent workflow, SQL database interface, SQL retriever, and LLM capabilities
 * Pydantic for structured data extraction, structured LLM output
 * sqlalchemy for data loading
 * sqlite for in-memory database
 * RAGChecker for comprehensive evaluation of RAG pipeline
 * LLM as a judge for custom evaluation


## Data

 * We use the [WikiTableQuestions dataset](https://ppasupat.github.io/WikiTableQuestions/) (Pasupat and Liang 2015) as our knowledge base.
 * Using `sqlalchemy` and `sqlite` we create SQL Database and load data.
 * Using `llama_index` SQL objects and SQL retrievers we connect and parse databse schema and data.

## Workflow

With LlamaIndex we define an agent as a workflow of the events:

![Text-to-SQL Agent Workflow](img/text-to-sql-agent-workflow.png)




Regression


## References

 * [LlamaIndex Agent Workflow](https://www.llamaindex.ai/blog/introducing-agentworkflow-a-powerful-system-for-building-ai-agent-systems)
 * [LlamaIndex Text-to-SQL Guide](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/)
 * [Structured Data Extraction in LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/structured_outputs/structured_outputs/)
 * [RAGChecker Framework](https://github.com/amazon-science/RAGChecker)
 * 