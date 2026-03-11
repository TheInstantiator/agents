import chromadb

DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/chroma_db"
OUTPUT_FILE = "/home/ouar/projects/agents/aiscripts/longsword_metadata_chunks.txt"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v1")

# Fetch all metadata and documents
results = collection.get(include=['documents', 'metadatas'])
metadatas = results['metadatas']
documents = results['documents']

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for doc, m in zip(documents, metadatas):
        # Look for 'longsword' in the document text instead of metadata
        found = False
        if "longsword" in doc.lower():
            found = True
                
        if found:
            count += 1
            f.write(f"--- Chunk {count} ---\n")
            f.write(f"Metadata: {m}\n\n")
            f.write(f"Document:\n{doc}\n")
            f.write("=" * 80 + "\n\n")

print(f"Total chunks with 'Longsword' in their metadata: {count}")
print(f"Dumped chunks to: {OUTPUT_FILE}")
