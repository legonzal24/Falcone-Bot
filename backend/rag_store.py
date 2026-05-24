#--------------------------------------------------------------------------------------------------
# MODULE IMPORTS
#--------------------------------------------------------------------------------------------------
# The Path module is used to build a folder path for ChromaDB.
# Optional allows document_id to be either a string or None.
# We import ChromaDB as the vector database. Requests is used for sending HTTP requests.
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

#--------------------------------------------------------------------------------------------------
# MODULE IMPORTS
#--------------------------------------------------------------------------------------------------
# The Path module is used to build a folder path for ChromaDB.
# Optional allows document_id to be either a string or None.
# We import ChromaDB as the vector database. Requests is used for sending HTTP requests.
from pathlib import Path
from typing import Optional
import chromadb
import requests
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# CONSTANTS
#--------------------------------------------------------------------------------------------------
# This sets the path for where the vector database will live. Makes uploaded data persistent.
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

# This names the collection which is like a table for vectors.
COLLECTION_NAME = "Falcone_uploaded_documents"

# Here we set the URL for Ollama's embedding endpoint along with the embedding model we will use.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

# We set the default chunk size (in characters) and the amount of characters that will overlap 
# between chunks to avoid context from being cut off awkwardly.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
#--------------------------------------------------------------------------------------------------

# This creates the ChromaDB folder if it does not already exist.
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# We create the ChromaDB client that will save data to disk.
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Here we pull the existing collection or create it for the first time. 
# This assigns the method for searching chunks as Cosine similarity.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# Here we have a function responsible for breaking down text into chunks.
def chunk_text(
        # The function needs 3 parameters. The text parameter should be a string. We're setting the
        # chunk_size parameter to an int which should be equal to 700. The overlap parameter should
        # be equal to 100.
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
# When the function completes it returns a list of strings.
) -> list[str]:
    # The text passed in gets stripped of extra white space at the beginning and end of documents. 
    # Result is passed into variable clean_text.
    clean_text = text.strip()
    
    # If the clean_text variable is empty, return an empty list.
    if not clean_text:
        return[]
    
    # This creates the variable chunks and makes it an empty list.
    chunks = []

    # This start variable sets the starting position as 0 which is first. This will be used for the 
    # current chunk.
    start = 0
    
    # This starts a loop to keep making chunks until the end of the document is reached.
    while start < len(clean_text):
        # This defines where the end of the chunk should be. Start value plus 700 reaches the end.
        end = start + chunk_size
        # This extracts the text in the clean_text variable and places it into 1 chunk. Then it 
        # gets appended to the chunks list that was previously created above.
        chunk = clean_text[start:end]
        chunks.append(chunk)
        # Checks whether the end of the document was reached. If so end the loop.
        if end >= len(clean_text):
            break
        # This sets the start position to 100 characters before the end to create the overlap.
        start = end - overlap
    # The chunks list is then returned at the end of this function.
    return chunks

# This function gets embeddings for the text that is passed in. It returns a list of 
# floating-point numbers.
def get_embedding(text: str) -> list[float]:
    # A POST request is sent to the embedding model with the passed in text and the response is
    # stored in the response variable.
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=60,
    )

    response.raise_for_status()

    # The response is converted from JSON into a python dictionary.
    data = response.json()

    # This checks if there is a label named embeddings in the data and if the list is not empty.
    if "embeddings" in data and data["embeddings"]:
        # This returns the first embedding in the list. 
        return data["embeddings"][0]
    
    # Same process but for a single embedding format used by older embedding endpoints.
    if "embedding" in data:
        return data["embedding"]
    
    raise ValueError("Ollama did not return an embedding.")

# This function takes an uploaded document and stores its chunks in ChromaDB. The function returns 
# an integer which is the number of chunks that were indexed.
def index_document(
        document_id: str,
        filename: str,
        content: str,
) -> int:
    # The chunk_text function is called, with the content of the document being passed in as a 
    # parameter. The result is passed into the chunks variable.
    chunks = chunk_text(content)

    # If the chunks list is empty, return 0.
    if not chunks:
        return 0
    
    # Here we create empty list variables. ids will be used to store the unique ID of each chunk 
    # stored in ChromaDB. metadatas holds the metadata like the document_id, filename, and chunk 
    # index for each chunk. embeddings holds the vector for each chunk.
    ids = []
    metadatas = []
    embeddings = []

    # Starts a loop where each chunks' chunk_index and chunk text is enumerated. So each chunk has
    # an index number associated with it.
    for chunk_index, chunk in enumerate(chunks):
        # The uploaded document has its document_id and all of the chunk index that belong to it
        # added to the ids list.
        ids.append(f"{document_id}:{chunk_index}")

        # Here the metadatas list is populated with the document_id, filename, and chunk_index.
        metadatas.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk_index,
            }
        )
        # The embeddings list is populated by the get_embedding function which we send the chunk 
        # (currently being processed) as a parameter and get back the data which is the embedding. 
        embeddings.append(get_embedding(chunk))
    
    # The upsert function stores the chunks along with the id, metadata, and embedding when all of 
    # the chunks have been processed.
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # This counts the number of chunks that were processed to note how many chunks were made and 
    # stored from an uploaded document.
    return len(chunks)

# This function converts the user prompt into an embedding and searches ChromaDB for similar 
# document chunks. It then returns the matching chunks as a string. The query is passed in as a 
# string, the document_id if one was uploaded, and n_result which we're setting to 3 by default. 
# This means the 3 most relevant chunks are retrieved.
def retrieve_relevant_chunks(
        query: str,
        document_id: Optional[str] = None,
        n_results: int = 5,
) -> str:
    
    # The prompt is passed into the get_embedding function to get the vector for it.
    query_embedding = get_embedding(query)

    # The variable query_arguments is created as a dictionary. The query embedding is stored along 
    # with n_results.
    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }

    # Check if a document was provided.
    if document_id:
        # This filters the query so that only chunks from the uploaded document are retrieved.
        query_arguments["where"] = {"document_id": document_id}
    
    # The search in ChromaDB is conducted using the query arguments (with our keyword filter where 
    # document_id = document_id which is what **query_arguments signifies). 
    # The document chunks are retrieved and specifically the first one in the list is passed into 
    # the documents variable. The same happens with the metadata.
    results = collection.query(**query_arguments)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # This checks if no matching chunks were retrieved. This means theres no useful RAG context.
    if not documents:
        return""
    
    # This list will hold formatted retrieved chunks.
    context_blocks = []

    # Starts a loop where each documents chunks' index and document_text is enumerated.
    for index, document_text in enumerate(documents):
        # The metadata index that belongs to the chosen chunk is placed in the metadata variable.
        # Only happens if it exists for the result. If not it should remain empty.
        # Gets the filename from the metadata if it exists.
        # Gets the chunk index from the metadata if it exists. 
        # This info is then added to the context_blocks list along with the document_text.
        metadata = metadatas[index] if index < len(metadatas) else {}
        source_filename = metadata.get("filename", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        context_blocks.append(
            f"Source: {source_filename}, chunk {chunk_index}\n"
            f"{document_text}"
        )
    
    # The context blocks get put together and returned to the prompt.
    return "\n\n---\n\n".join(context_blocks)
