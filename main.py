from langchain_community.document_loaders import DirectoryLoader
from graph.graph import create_graph
from graph.graph_state import GraphState
from utils.setup import setup

# Call setup to initialize environment
setup()

path = "data/"
loader = DirectoryLoader(path, glob="*.html")
docs = loader.load()

workflow = create_graph()
state = GraphState(docs=docs, num_questions=2)
result = workflow.invoke(state)

print(result["questions"])