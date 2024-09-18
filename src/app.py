import ast
import os
import random
from typing import Any, Dict, List, Literal, Annotated, TypedDict

import dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from IPython.display import Image
from langchain.agents.agent import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import tool, BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from opentelemetry import trace, trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from langchain_core.documents import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI agentic bot that can interact with a database"
)

st.title("💬 AI agentic RAG")
st.caption("🚀 A Bot that can use an agent to retrieve, augment, generate, validate and iterate")

@st.cache_resource
def setup_tracing():
    exporter = AzureMonitorTraceExporter.from_connection_string(
        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
    trace.get_tracer_provider().add_span_processor(span_processor)
    LangchainInstrumentor().instrument()
    return tracer

@st.cache_resource
def create_session(st: st) -> None:
    if "session_id" not in st.session_state:
        id = random.randint(0, 1000000)
        st.session_state["session_id"] = "00000000-0000-0000-0000-" + str(id).zfill(12)
        print("started new session: " + st.session_state["session_id"])
        st.write("You are running in session: " + st.session_state["session_id"])

tracer = setup_tracing()
create_session(st)

def num_tokens_from_messages(messages: List[str]) -> int:
    '''
    Calculate the number of tokens in a list of messages. This is a somewhat naive implementation that simply concatenates 
    the messages and counts the tokens in the resulting string. A more accurate implementation would take into account the 
    fact that the messages are separate and should be counted as separate sequences.
    If available, the token count should be taken directly from the model response.
    '''
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = 0
    content = ' '.join(messages)
    num_tokens += len(encoding.encode(content))

    return num_tokens

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class TokenCounterCallback(BaseCallbackHandler):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.completion_tokens += 1

callback = TokenCounterCallback()

def measure_prompt_tokens(messages: List[BaseMessage]) -> List[BaseMessage]:
    for message in messages:
        callback.prompt_tokens += num_tokens_from_messages([message.content])
    return messages

llm: AzureChatOpenAI = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True,
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

@st.cache_resource
def create_search_index() -> AzureSearch:
    index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
        index_name=index_name,
        embedding_function=embeddings_model.embed_query,
    )

search_index = create_search_index()

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

#-----------------------------------------------------------------------------------------------

@tool
def index_search_tool(query: str) -> List[str]:
    """
    Search for relevant schema information in the vector index based on the user's query.

    Args:
        query (str): The input query. This is used to search the vector index.

    Returns:
        List[str]: The resulting list of schema information in the form Table;Column;DataType.

    """

    results = search_index.similarity_search(
        query=query,
        search_type="hybrid",
    )

    return [result.page_content for result in results]

#-----------------------------------------------------------------------------------------------

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = {
        "messages": [
            ("user", human_query),
        ]
    }

    with st.chat_message("Human"):
        st.markdown(human_query)

    with tracer.start_as_current_span("graph-chain") as span:
        
        #add ui stream output code here

        span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
        span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
        span.set_attribute("gen_ai.response.total_tokens", callback.completion_tokens + callback.prompt_tokens)