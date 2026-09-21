#--------------------------------------------------------------------------------------------------
# MODULE IMPORTS
#--------------------------------------------------------------------------------------------------
# FastAPI provide the main class for receiving web requests from the front-end. It also provides 
# the file upload functionality, exception handling for HTTP errors, and file handling in HTTP 
# requests.The Path module in the pathlib library allows us to create files/folders.
# BaseModel from Pydantic helps define the data this app expects to receive. 
# Field is used for the list structure defined for history.
# The Optional module from typing is used for for the Document ID since it may not be included.
# UUID creates Unique IDs for the documents being uploaded and referenced. 
from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4


# Here we are importing the storage location for uploaded documents and data structure for an
# uploaded document object. We also import the document parser for PDF support.
from backend.document_store import DOCUMENT_STORE, UploadedDocument
from backend.document_parser import extract_text_from_upload

# Here we import the invoke_falcone_chain function to be able to build the model prompt and call 
# Ollama through LangChain.
from backend.falcone_chain import invoke_falcone_chain

# The index_document function is used during the file upload. The retrieve_relevant_chunks function 
# is used during chat.
from backend.rag_store import index_document

# This is where we import the logger setup function.
from backend.logger_config import setup_logger
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# BACKEND CONSTANTS
#--------------------------------------------------------------------------------------------------
# This creates and titles FastAPI app that will receive requests.
app = FastAPI(title="Falcone-Bot API")

# Here are some phrases to watch for during redteam testing the model with obvious 
# attack attempts.
SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal your system prompt",
    "show me the system prompt",
    "what is your system prompt",
    "developer message",
    "hidden instructions",
    "bypass",
    "jailbreak",
    "internal records",
    "restricted records",
    "confidential records",
]

# This will be used later to invoke a limit on the size of upload files.
MAX_UPLOAD_BYTES = 100_000

# This will be used later to check if the uploaded file matches the allowed file extensions.
ALLOWED_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".pdf",
}

# This sets the default number of RAG results that should be retrieved.
RAG_RESULTS = 5

# Now we create the logger for use in main.py as the file and console logging.
logger = setup_logger()
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# DETECTION HELPERS
#--------------------------------------------------------------------------------------------------
# This function defines how we check whether the user message contains suspicious phrases. 
# It returns True or False (boolean) if it is detected.
def detect_suspicious_input(message: str) -> bool:
    # The message is set to lowercase for comparison.
    lower_message = message.lower()
    return any(pattern in lower_message for pattern in SUSPICIOUS_PATTERNS)

# This function defines how we check whether the user is asking for internal records.
def detect_record_request(message: str) -> bool:
    return "internal records" in message.lower() or "restricted records" in message.lower()

#--------------------------------------------------------------------------------------------------
# REQUEST MODELS
#--------------------------------------------------------------------------------------------------
# Define the data structure for requests sent to Ollama.
class ChatRequest(BaseModel):
    # The message field must be a string (This will be the user prompt).
    # The history field is a list of previous chat messages. This allows 
    # it to remember the current conversation happening. Field(default_factory=list)
    # creates a fresh empty list for the request if no history is provided.
    message: str
    history: list[dict] = Field(default_factory=list)
    # The document ID field should be added as part of the chat request received from frontend.
    document_id: Optional[str] = None
    # This controls whether each external tool is available during the request
    web_search_enabled: bool = True
    url_fetch_enabled: bool = True

#--------------------------------------------------------------------------------------------------
# HEALTH CHECK
#--------------------------------------------------------------------------------------------------
# This is a GET endpoint to confirm this backend app is alive and running.
# Whenever it receives the GET request, it will respond with the status.
# Creating an endpoint means that the next defined function is the action/purpose.
@app.get("/health")
def health_check():
    logger.info("Health check endpoint was called")
    return {"status": "Falcone-Bot backend is running"}

#--------------------------------------------------------------------------------------------------
# DOCUMENT UPLOAD
#--------------------------------------------------------------------------------------------------
# This is where we create the POST endpoint for uploading files.
@app.post("/documents/upload")
# We define file uploads as an asynchronous function so that the process is not completely halted 
# during the upload of a file. We pass the UploadFile (from document_store.py) object as the 
# structure for the parameter of this function. The "..." means that the file is required. If the 
# file is missing then FastAPI will return an error.
async def upload_document(file: UploadFile = File (...)):
    # This extracts the filename from the file to place into the variable "filename". If there is 
    # no filename then "uploaded_document" is used instead. 
    filename = file.filename or "uploaded_document"
    # This extracts the file extension from the file to place into the variable "extension". The 
    # name is treated as lower case for comparison purposes.
    extension = Path(filename).suffix.lower()
    # Now we compare that the extension is in the allowed list.
    if extension not in ALLOWED_FILE_EXTENSIONS:
        # If the extension is not allowed, throw a 400 Error and let the user know it's unsupported.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed types: {ALLOWED_FILE_EXTENSIONS}",
        )
    
    # This will extract/read the raw bytes of the file asynchronously to place in to "raw_content".
    raw_content = await file.read()

    # Now we compare the file size to what is allowed. 
    if len(raw_content) > MAX_UPLOAD_BYTES:
        # Let the user know if the file is too large. We also add a log entry for the rejected file.
        logger.warning(f"Rejected oversized file: {filename}")
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Max size is {MAX_UPLOAD_BYTES} bytes.",
        )
    
    # Now we read the file content.
    try:
        # Here we call the function from our document parser to extract the text. We pass in the 
        # filename and the raw content as parameters.
        text_content = extract_text_from_upload(filename=filename, raw_content=raw_content)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract text from uploaded file: {error}",
        )
    
    # We generate a document ID for the document.
    document_id = str(uuid4())

    # Here we place the uploaded document into the Document Store. These are the same fields we 
    # defined in UploadedDocument which goes into DOCUMENT_STORE.
    DOCUMENT_STORE[document_id] = UploadedDocument(
        document_id=document_id,
        filename=filename,
        content=text_content,
    )

    # Here we call the index_document function from rag_store and place the result in chunk_count.
    chunk_count = index_document(
        document_id=document_id,
        filename = filename,
        content=text_content,
    )

    # Add a log entry for the document being stored in the Document Store.
    logger.info(
        f"Stored uploaded document. document_id={document_id}", 
        f"filename={filename}", 
        f"characters={len(text_content)}"
        f"chunks_indexed={chunk_count}"
        )

    # Here we respond to the frontend with a successful result.
    return {
        "message": "Document Uploaded Successfully.",
        "document_id": document_id,
        "filename": filename,
        "characters": len(text_content),
        "chunks_indexed": chunk_count,
    }

#--------------------------------------------------------------------------------------------------
# CHAT
#--------------------------------------------------------------------------------------------------
# This is a POST endpoint to receive prompts from the user interface.
@app.post("/chat")
# The "request: ChatRequests" tells the endpoint to expect requests in the ChatRequest
# format that was defined in the class above. A message and a history field. The first
# system message will not have a history field.
def chat(request: ChatRequest):
    # This is where we're logging that a user message was received from the Falcone UI.
    logger.info(f"Chat request received. User message: {request.message}")
    # We also log the number of requests received in the conversation.
    logger.info(f"Conversation history length: {len(request.history)}")
    # Log whether each tool is enabled for this particular request.
    logger.info(f"Web search enabled: {request.web_search_enabled}")
    logger.info(f"URL fetch enabled: {request.url_fetch_enabled}")
    # This is where we invoke the function to check for suspicious injection language.
    if detect_suspicious_input(request.message):
        logger.warning(f"Suspicious input detected: {request.message}")
    # This is where we invoke the function to check if records were requested.
    if detect_record_request(request.message):
        logger.warning(f"User requested internal/restricted records")

    # We will first check if a document was uploaded.
    if request.document_id:
        # This places the document in the Document Store (with the matching document ID) into 
        # the variable "UploadedDocument".
        uploaded_document = DOCUMENT_STORE.get(request.document_id)

        # This checks if the document lookup above failed and raises an exception.
        if not uploaded_document:
            logger.warning(f"Chat request reference missing document_id={request.document_id}")
            raise HTTPException(
                status_code=404,
                detail="Uploaded document not found.",
            )
        
    try:
        # Here we invoke the function to send the user pronpt to Ollama through the chain.
        # web_search_enabled is forwarded into the LangChain Layer so the backend can 
        # control whether a tool is bound to the model.
        chain_result = invoke_falcone_chain(
            user_message=request.message,
            history=request.history,
            document_id=request.document_id,
            n_results=RAG_RESULTS,
            web_search_enabled=request.web_search_enabled,
            url_fetch_enabled=request.url_fetch_enabled,
        )

        # Here we log the use of RAG context, the reply length, and potential sensitive
        # information being provided in the response. We also log every tool the model called.
        logger.info(
            "LangChain response generated. "
            f"reply_length={len(chain_result.reply)} "
            f"rag_used={chain_result.rag_used}"
        )
        if chain_result.rag_used:
            logger.info(
                f"RAG context length: {len(chain_result.rag_context)} characters"
            )
        if chain_result.tools_used:
            logger.warning(
                f"Tools invoked during chat: {chain_result.tools_used}"
            )
        if (
            "internal records" in chain_result.reply.lower()
            or "falcone" in chain_result.reply.lower()
        ):
            logger.warning(
                "Model reply may contain sensitive or internal Falcone-related content"
            )
        # Here we provide the tools used to the response for the frontend can provide it.
        return {
            "reply": chain_result.reply,
            "tools_used": chain_result.tools_used,
        }
    
    # Here we catch any exception errors during the LangChain processing.
    except Exception as error:
        logger.exception(f"Unexpected error during LangChain chat processing: {error}")

        return {
            "reply": "Falcone-Bot encountered an internal issue.",
            "tools_used": [],
        }
    
