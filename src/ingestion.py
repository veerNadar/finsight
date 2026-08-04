import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += text
    doc.close()
    return full_text 
    
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    text = extract_text_from_pdf("data/raw/infosys_annual_report_fy25.pdf")
    print("Total characters extracted:", len(text))

    chunks = chunk_text(text)
    print("Total chunks created:", len(chunks))
    print("\n--- First chunk ---\n")
    print(chunks[0])
    print("\n--- Second chunk ---\n")
    print(chunks[1])