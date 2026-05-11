# These imports will be used to read the text of PDF files.
from io import BytesIO
from pypdf import PdfReader

# This function will extract text from regurlar files uploads.
def extract_text_from_upload(filename: str, raw_content: bytes) -> str:
    extension = filename.lower().split(".")[-1]

    # Check if the extension is a PDF.
    if extension =="pdf":
        # Invoke function to extract text from PDF.
        return extract_text_from_pdf(raw_content)
    
    # Otherwise just decode the content as UTF-8.
    return raw_content.decode("utf-8")

# This function extracts text from PDF files. It will return text as str (-> str).
def extract_text_from_pdf(raw_content: bytes) -> str:
    # BytesIO turns the raw context into a file-like object and places it in the reader variable.
    reader = PdfReader(BytesIO(raw_content))

    # This variable is an empty list that will be filled with text from each page of the PDF.
    pages = []

    # This loop runs through every page in the reader variable.
    for page in reader.pages:
        # This extracts text from the current page and places it in the page_text variable.
        page_text = page.extract_text()

        # This checks if the page_text variable has more text or if it was empty
        if page_text:
            # Here we add the text from the extraction to the pages list variable created.
            pages.append(page_text)
    
    # This combines the pages of text with 2 line breaks in between them.
    return "\n\n".join(pages)
