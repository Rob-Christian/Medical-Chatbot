from flask import Flask, render_template, request, jsonify
import os
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

# Flask app setup
app = Flask(__name__)

# Environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Medical Chatbot"

# LangChain Setup
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name="medical-chatbot")

def initialize_graph():
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(state: MessagesState):
        """Generate answer"""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. Use the following context. "
            "If you don't know, say that. Use three sentence maximum and keep it concise."
            "\n\n" + docs_content
        )
        prompt = [SystemMessage(system_message_content)] + state["messages"]
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools := ToolNode([retrieve]))
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile(checkpointer=MemorySaver())

new_thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": new_thread_id}}
graph = initialize_graph()

@app.route("/")
def index():
    """Render the main interface for the chatbot."""
    return render_template("index.html")

@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Start a new chat by resetting the thread ID."""
    new_thread_id = str(uuid.uuid4())
    config["configurable"]["thread_id"] = new_thread_id
    return jsonify({"message": "New chat started", "thread_id": config["configurable"]["thread_id"]})

@app.route("/chat_input", methods=["POST"])
def chat_input():
    """Handle user input and generate a chatbot response."""
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    input_message = HumanMessage(content=user_input)
    response = []
    for chunk in graph.stream({"messages": [input_message]}, config):
        chunk
    if "generate" in chunk:
        response.append(chunk["generate"]["messages"][0].content)
    elif "query_or_respond" in chunk:
        response.append(chunk["query_or_respond"]["messages"][0].content)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
