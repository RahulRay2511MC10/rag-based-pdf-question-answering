from pypdf import PdfReader


def load_pdf(file_path):

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": file_path,
                "page": page_number
            }
        })

    return documents