from langchain_community.document_loaders import DirectoryLoader
from graph import create_graph, GraphState
path = "data/"
loader = DirectoryLoader(path, glob="*.html")
docs = loader.load()

workflow = create_graph()
state = GraphState(docs=docs, num_questions=1)
result = workflow.invoke(state)

print(result["questions"])