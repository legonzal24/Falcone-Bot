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