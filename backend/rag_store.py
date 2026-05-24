#--------------------------------------------------------------------------------------------------
# MODULE IMPORTS
#--------------------------------------------------------------------------------------------------
# The Path module is used to build a folder path for ChromaDB.
# Optional allows document_id to be either a string or None.
# The Chroma class from LangChain is used to create the VectorDB.
# The OllamaEmbeddings and RecursiveCharacterTextSplitter classes from LangChain allow us to 
# easily use the embedding and text splitting functions/methods that are available.
from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# CONSTANTS
#--------------------------------------------------------------------------------------------------
# This sets the path for where the vector database will live. Makes uploaded data persistent.
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

# This names the collection which is like a table for vectors.
COLLECTION_NAME = "Falcone_uploaded_documents"

# Here we set the URL for Ollama's along with the embedding model we will use.
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

# We set the default chunk size (in characters) and the amount of characters that will overlap 
# between chunks to avoid context from being cut off awkwardly.
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# LANGCHAIN RAG COMPONENTS
#--------------------------------------------------------------------------------------------------
# This creates the ChromaDB folder if it does not already exist.
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# Here we're defining a new embedding class instance (object) that will be called for Chroma.
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# We create an object from the text splitting class within langchain_text_splitters.
# The chunk size and overlap are defined in our constants. The separaters tell the text_splitter 
# where to deliniate the end/beginning of chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

# Here we create an instance of the vector store from the Chroma class and use the parameters which 
# include some of the constants we defined. 
# The collection metadata assigns the method for searching chunks as Cosine similarity.
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_metadata={"hnsw:space": "cosine"},
)

#--------------------------------------------------------------------------------------------------
# Document Preparation
#--------------------------------------------------------------------------------------------------
# This section prepares uploaded text so LangChain and ChromaDB can use it. The parameters are a 
# document_id, a filename, and the content within it. Each of these parameters are strings. 
# The function then produces a List of LangChain Document objects.
def build_langchain_documents(
    document_id: str,
    filename: str,
    content: str,
) -> list[Document]:
    
    # The strip method removes extra white spaces from the beginning and end of the content.
    clean_content = content.strip()

    # Here we check if clean_content has any actual text. If not, exit and return an empty list.
    if not clean_content:
        return []

    # Here we have an instance of the Document class created as base_document. The clean_content
    # is passed in along with the metadata(which is in dictionary format).
    base_document = Document(
        page_content=clean_content,
        metadata={
            "document_id": document_id,
            "filename": filename,
            "source": filename,
        },
    )

    # The base_document from above is split into chunks of text using the split_documents method
    # of the text_splitter object we created above. The result is then placed into split_documents.
    split_documents = text_splitter.split_documents([base_document])

    # This loop goes through each chunk in split_documents and provides a chunk index as well as 
    # apply the metadata of the base_document to keep it all together.
    for chunk_index, document in enumerate(split_documents):
        document.metadata["chunk_index"] = chunk_index
        document.metadata["document_id"] = document_id
        document.metadata["filename"] = filename
        document.metadata["source"] = filename

    # Once that's complete we can return the split_documents.
    return split_documents 

#--------------------------------------------------------------------------------------------------
# Indexing
#--------------------------------------------------------------------------------------------------
# This function takes an uploaded document and stores its chunks in ChromaDB. The function returns 
# an integer which is the number of chunks that were indexed.
def index_document(
        document_id: str,
        filename: str,
        content: str,
) -> int:
    # The build_langchain_documents function we created is called, with the document_id, filename,
    # and content of the document being passed in as a parameter. The result is passed into 
    # the documents variable.
    documents = build_langchain_documents(
        document_id=document_id,
        filename=filename,
        content=content,
    )

    # If the documents list is empty, return 0.
    if not documents:
        return 0
    
    # Here we create the list of IDs for the document chunks.
    ids = [
        f"{document_id}:{chunk_index}"
        for chunk_index in range(len(documents))
    ]

    # Now the document chunks get added to the vectorDB using the vector_store object we created.
    # The ids are included with the document chunks.
    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    # Here we return the number of chunks that were indexed and added to the vectorDB.
    return len(documents)


#--------------------------------------------------------------------------------------------------
# Retrieval
#--------------------------------------------------------------------------------------------------
# This section converts the user prompt into an embedding and searches ChromaDB for similar 
# documents and document chunks. The query is passed in along with the similar chunks to the LLM.

# This function retrieves Document objects from the vector store which are made up of page_content
# and the associated metadata. The user query, the document ID if applicable, and the number of
# chunks to retrieve are passed in as parameters. A list of Document objects are returned.
def retrieve_relevant_documents(
        query: str,
        document_id: Optional[str] = None,
        n_results: int = 5,
) -> list[Document]:
    # Here we check if the query has any actual text after stripping the extra white space.
    if not query.strip():
        return []

    # Here we create a search filter and set it to None. All documents can be scanned if no
    # document_id is provided.
    search_filter = None

    # If there is a document_id, that becomes the search filter.
    if document_id:
        search_filter = {
            "document_id": document_id,
        }
    
    # Here we invoke the similarity_search method and pass the results into the variable documents.
    # The documents variable is then returned.
    documents = vector_store.similarity_search(
        query=query,
        k=n_results,
        filter=search_filter,
    )
    return documents

# The function below formats the document objects into text chunks that can be added to the prompt.
def format_documents_for_prompt(documents: list[Document]) -> str:
    # We check if the list is empty. If it is, an empty string is returned.
    if not documents:
        return""
    
    # This variable is an empty list that will hold the document chunks.
    context_blocks = []

    # This loop will go through each document chunk.
    for document in documents:
        # Here we grab either the filename if it it exists, or unknown if it has no name.
        filename = document.metadata.get("filename", "unknown uploaded document")

        # This adds the filename and the page_content from each document. We're also using strip to
        # remove any excess white spaces from the beginning or end.
        context_blocks.append(
            f"Source document: {filename}\n"
            f"{document.page_content.strip()}"
        )

    # The list of context_blocks get joined together and returned as one long string with a clear
    # separation between chunks for the prompt that is sent to the model.
    return "\n\n---\n\n".join(context_blocks)

# This is the primary function that gets called from main.py to split the document into chunks, 
# store it in the vectorDB and then grab the relevant document chunks.
def retrieve_relevant_chunks(
    query: str,
    document_id: Optional[str] = None,
    n_results: int = 5,
) -> str:
    # Here we actually call the function that grabs the relevant chunks.
    documents = retrieve_relevant_documents(
        query=query,
        document_id=document_id,
        n_results=n_results,
    )

    # Here we pass the relevant chunks through the format_documents_for_prompt function to make it
    # prompt ready and then return the results.
    return format_documents_for_prompt(documents)

