import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# Load your environement variables
# Make sure to create a .env file with your API keys
# and set the variables GROQ_API_KEY and GOOGLE_API_KEY
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY') # type: ignore
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY') # type: ignore

# Load the table description from a file
with open("table_description.txt", "r") as f:
    table_description = f.read()

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        sql_query: sql_query
        grade_query: Is the sql query correct
        invoke_higher_power: Should we call Google uncle for help
        retries: number of retries before we give up
    """
    question: str
    sql_query: str
    statement: str
    grade_query: str
    invoke_higher_power: bool
    retries: int

#NODES 
def get_sql_query(state):
    ''' Converts the natural language to SQL Query using one of the two LLMs'''

    print(""" GENERATING SQL_QUERY""")

    system = """ You are an expert at generating SQL Queries.
    You have been given a SQL table with the following schema:
    {table_description} 
    Convert the following natural language request into a valid SQL query for the contributions table. 
    Ensure the SQL query is optimized and follows best practices.
    Do not assume extra information about the tables other than the one given to you.
    Reply only with the relevant SQL Query and nothing else"""

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
    )

    if state["invoke_higher_power"] == False:
        llm = ChatGroq(model="llama3-70b-8192")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",max_retries=2)
        
    sql_query_generator = prompt | llm | StrOutputParser()
    question = state["question"]
    sql_query = sql_query_generator.invoke({"table_description":table_description,"question": question})

    retries = state["retries"]
    
    return {"sql_query":sql_query,"retries":retries+1,"invoke_higer_power":True}
    
def describe_sql_query(state):
    ''' Describes the SQL query using LLM.
    This is used to check if the SQL query is correct and if it answers the question.'''

    print(""" DESCRIBING SQL_QUERY""")
    system = """State what the sql query does on the given table."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "sql_query: {sql_query} \n table: {table_description}"),
        ]
    )
    llm = ChatGroq(model="llama3-70b-8192")
    describer_llm = prompt | llm 
    statement = describer_llm.invoke({"sql_query":state["sql_query"],"table_description":table_description})

    return {"statement":statement}

def grade_sql_query(state):
    print(""" GRADING SQL_QUERY""")
    system = """ You are an expert in determining if the steps given is sufficient and correct to answer the question.
    You are given the following schema \n
    Table schema: {table_description} \n
    Is the folllowing steps correct to answer the given question ? \n
    Answer only 'yes' or 'no' """

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "steps: {steps} \n question: {question}"),
    ])
    llm = ChatGroq(model="llama3-70b-8192")
    prover = prompt | llm |StrOutputParser()
    grade_query = prover.invoke({"table_description":table_description,"steps":state["statement"],"question":state["question"]})
    return {"grade_query":grade_query}

def final_query(state):
    print("Here is your final query")
    print(state["sql_query"])
    
#EDGES - Conditional edge to check if we should invoke the higher power
def should_invoke(state):
    retries = state["retries"]
    grade = state["grade_query"]
    if grade == 'yes' or retries >= 3:
        print(""" ANSWER LOOKS FINE """)
        return 'final_query'
    else:
        print(""" ANSWER IS NOT GOOD, DECIDING TO INVOKE BETTER LLM """)
        return 'get_sql_query'

memory = MemorySaver()

workflow = StateGraph(GraphState)
workflow.add_node('get_sql_query',get_sql_query)
workflow.add_node('describe_sql_query',describe_sql_query)
workflow.add_node('grade_sql_query',grade_sql_query)
workflow.add_node('final_query',final_query)

workflow.add_edge(START,"get_sql_query")
workflow.add_edge("get_sql_query","describe_sql_query")
workflow.add_edge("describe_sql_query","grade_sql_query")
workflow.add_conditional_edges("grade_sql_query",should_invoke,['final_query','get_sql_query'])
workflow.add_edge('final_query',END)

app = workflow.compile(checkpointer=memory)

with open("question.txt", "r") as f:
    question = f.read()

config = {"configurable": {"thread_id": "1"}}

input = {"question": question, "retries": 0, "invoke_higher_power":False}

for output in app.stream(input, config, stream_mode="values"):
    print("\n---\n")