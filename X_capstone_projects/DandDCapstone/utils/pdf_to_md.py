import os
import time
import pymupdf4llm

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "documents")
PDF_PATH = os.path.join(DOCS_DIR, "DandDRuleset.pdf")
MD_PATH = os.path.join(DOCS_DIR, "DandDRuleset.md")

def convert():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Cannot find PDF at {PDF_PATH}")
        return

    print(f"🚀 Starting conversion of {os.path.basename(PDF_PATH)} to Markdown...")
    start_time = time.time()

    try:
        # Perform the conversion
        # pymupdf4llm is great because it handles tables and image captions well
        md_text = pymupdf4llm.to_markdown(PDF_PATH)
        
        # Save to the documents folder
        with open(MD_PATH, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        elapsed = time.time() - start_time
        file_size_kb = os.path.getsize(MD_PATH) / 1024
        
        print(f"✅ Success! Created: {MD_PATH}")
        print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        print(f"📊 Markdown Size: {file_size_kb:.2f} KB")

    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")

if __name__ == "__main__":
    convert()
