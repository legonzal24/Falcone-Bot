# FastAPI provide the main class for receiving web requests from the front-end.
# BaseModel from Pydantic helps define the data this app expects to receive. Field
# is used for the list structure defined for history.
# Requests is the library we use to send HTTP requests (with prompts) to Ollama.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import requests

# Here we are pulling in the System Prompt and Internal Data we previously created.
from backend.falcone_prompt import FALCONE_SYSTEM_PROMPT
from backend.data import INTERNAL_DATA

# This creates and titles FastAPI app that will receive requests.
app = FastAPI(title="Falcone-Bot API")

# Configration of Ollama instance running and Model to use.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"

# Define the data structure for requests sent to Ollama.
class ChatRequest(BaseModel):
    # The message field must be a string (This will be the user prompt).
    # The history field is a list of previous chat messages. This allows 
    # it to remember the current conversation happening. Field(default_factory=list)
    # creates a fresh empty list for the request if no history is provided.
    message: str
    history: list[dict] = Field(default_factory=list)

# This is a GET endpoint to confirm this backend app is alive and running.
# Wheneever it receives the GET request, it will respond with the status.
# Creating an endpoint means that the next defined function is the action/purpose.
@app.get("/health")
def health_check():
    return {"status": "Falcone-Bot backend is running"}

# This is a POST endpoint to receive prompts from the user interface.
@app.post("/chat")
# The "request: ChatRequests" tells the endpoint to expect requests in the ChatRequest
# format that was defined in the class above. A message and a history field. The first
# system message will not have a history field.
def chat(request: ChatRequest):
    messages = [
        {
            # The first message to the model is the system prompt. We always include
            # the role followed by the content for the prompt. This is the only message
            # so far in the message list we've created.
            "role": "system",
            # VULNERABILITY 1: We're including the restricted data along with the 
            # system prompt. If the system prompt is exposed, so is the data.
            "content": FALCONE_SYSTEM_PROMPT + "\n\n" + INTERNAL_DATA
        }
    ]

    # Here we are extending the history field with the last message request.
    messages.extend(request.history)
    # This line adds the user's prompt to the list of messages.
    messages.append({
        "role": "user",
        "content": request.messages
    })

    # This is creating the HTTP data sent in the POST request to Ollama
    payload = {
        "model": MODEL_NAME,
        # This is sending the full list of chat messages recieved.
        "messages": messages,
        # The stream field below tells Ollama not to stream the response token by token.
        # once the app has received the full answer it will display it all at once.
        "stream": False,
        # Here we set the block of settings for the model.
        "options": {
            # Here we set the creativity/randomness of the model.
            "temperature": 0.7,
            # This sets the limit of the tokens that will be created so that the
            # response is not massive. This is not the context window.
            "num_predict": 500
        }
    }

    # Now we will be sending the HTTP request to Ollama. The request is sent to the
    # Ollama URL, the payload field is included as JSON, with a timeout of 120. Then
    # the response to the POST request is saved in the response variable.
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    # Here we check for an error from Ollama in case something went wrong.
    response.raise_for_status

    # Now we take the HTTP response from Ollama which is in the response variable and
    # we convert it into a json response (python dictionary which is a key-value 
    # collection. For example: "role: system", "content: system prompt") and place it 
    # in the data variable.
    data = response.json()
    
    # Here we return the content in the message of Ollama's (llama3.2 model) response,
    # as the value produced by the chat function. This will be sent to the user interface.
    return {
        "reply": data["message"]["content"]
    }
    
