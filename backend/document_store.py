# The dataclass module allows us to create a class that is already designed to hold data.
# The Dict module allows us to use the structure of a dictionary to hold document and their data.
from dataclasses import dataclass
from typing import Dict

# This is where we invoke the dataclass for uploaded documents. It sets up a document ID, a filename,
# and the content field. All are defined as a string.
@dataclass
class UploadedDocument:
    document_id: str
    filename: str
    content: str

# Here we create the global variable that will hold all documents that get uploaded. The variable is
# being instructed to be formed as a Dictionary (key/value entry) type of variable. Inside the 
# variable we store data as a string, and the content is the UploadedDocument objects.
DOCUMENT_STORE: Dict[str, UploadedDocument] = {}
