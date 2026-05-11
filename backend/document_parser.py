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

def extract_text_from_pdf(raw_content: bytes) -> str:
    reader = PdfReader(BytesIO(raw_content))

    pages = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)
    
    return "\n\n".join(pages)