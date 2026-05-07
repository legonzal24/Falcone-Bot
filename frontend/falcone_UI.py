# The Streamlit library is used for the user interface. We import it as st to make it
# easier to invoke its components. Requests is the library we use to send HTTP requests (with prompts) to 
# the backend application.
import streamlit as st
import requests

# This creates a variable for the URL of the backend application's chat and file upload endpoints.
BACKEND_CHAT_URL = "http://localhost:8000/chat"
BACKEND_UPLOAD_URL = "http://localhost:8000/documents/upload"

# Here we set the configuration for the browser tab to show the name of the app and the icon. The title
# and caption will be on the web page itself. We also set the sidebar header.
st.set_page_config(page_title="Falcone-Bot", page_icon="🦇")
st.title("Falcone-Bot")
st.caption("Falcone Family AI assistant")
st.sidebar.header("Falcone-Bot")


# Here we check if there is stored chat history. If sessions_state does not include "messages" then we 
# create it as an empty list where messages will be added. We do the same for "document_id" and 
# "document_name" within the session state.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

# This is a loop to iterate through each message in the session state (chat history) and write it to the
# screen as a chat bubble. Streamlit already knows to differentiate messages with the user role and 
# messages with the assistant role.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# This creates the chat input box at the bottom of the page. and assigns it to the user_input variable
user_input = st.chat_input("Enter your prompt for assistance on the family business...")

# Here we add the file upload capability to the UI.
uploaded_file = st.sidebar.file_uploader(
    "Uploade a case file",
    type=["txt", "md", "csv", "json"],
)

# Here we upload the document to the backend and get back the document_id for use in the /chat endpoint.
if uploaded_file is not None:
    if st.sidebar.button("Upload Document"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "text/plain"
            )
        }
        response = requests.post(
            BACKEND_UPLOAD_URL,
            files=files,
        )

        if response.status_code == 200:
            upload_result = response.json()
            st.session_state.document_id = upload_result["document_id"]
            st.session_state.document_name = upload_result["filename"]

            st.sidebar.success(
                f"Uploaded: {st.session_state.document_name}"
            )
        else:
            st.sidebar.error(response.json().get("detail", "Upload failed."))


# If there is user input submitted, add it to the bottom of the session state (history) of messages including 
# the role and content.
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Here we write the users message stored in user_input to the screen in a chat bubble.
    with st.chat_message("user"):
        st.write(user_input)

    # This provides the chat history into the variable history_for_backend which is a list.
    history_for_backend = [
        # The list includes the "role" and "content" of each message in the conversation.
        # The history of messages will include all messages -1 which is the current message
        # so that it is not duplicated. The last if statement ensures only messages with the 
        # user and assistant role are included in the history. System messages are excluded.
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
        if msg["role"] in ["user", "assistant"]
    ]

    # This sends the POST request to the backend application with the current user_input
    # message and the history of messages so far in JSON format. Then when it receives a 
    # response it gets stored in the response variable. We also add the document_id value 
    # we received from uploading the document.
    try:
        response = requests.post(
            BACKEND_CHAT_URL,
            json={
                "message": user_input,
                "history": history_for_backend,
                "document_id": st.session_state.document_id,
            },
            timeout=120
        )

        # Capture the error if one is received from the backend application. 
        # The reply field of the response is stored in the reply variable.
        response.raise_for_status()
        reply = response.json()["reply"]
    
    # This catches any exceptions so that the error is displayed if an appropriate response
    # is not received from the backend app. The "f" tells python to include the error variable
    # in the string stored in reply.
    except Exception as error:
        reply = f"Falcone-Bot backend error: {error}"

    # The reply is then added to the history of messages with the role of assistant.
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    # Here we display the new message to the screen as a chat bubble as the assistant role.
    with st.chat_message("assistant"):
        st.write(reply)

