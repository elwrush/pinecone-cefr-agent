from dotenv import load_dotenv
load_dotenv()

import pinecone
import os
from sentence_transformers import SentenceTransformer
import json
import glob

# --- Pinecone Configuration (same as before, with your hostname) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_HOST = os.environ.get("PINECONE_HOST")  # YOUR HOSTNAME HERE
INDEX_NAME = "cefr-text-index"

#pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT, host=PINECONE_HOST) #This is replaced
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) # The init call is changed, and moved to this
index = pc.Index(INDEX_NAME) # Host no longer used
# The above three lines do the same as the two below, to link to the Pinecone DB and load it.
#You use either set of code

# --- Sentence Transformer Setup ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# --- Data Directory ---
# Get all .txt files in the current directory and subdirectories
DATA_DIR = "C:\\PINECONE"
METADATA_PATTERN = "C1_metadata.json"  # this is the change
#The files are C1, so only use the C1 metadata, because we will load in B1 later on.

def process_file(filepath, metadata_dict, model):
    """Loads a text file, chunks it, generates embeddings, and creates metadata."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

    chunks = text.split("\n\n")
    chunks = [c.strip() for c in chunks if c.strip()]

    data_to_upsert = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        chunk_id = f"{os.path.basename(filepath)}_{i}"

        metadata = {
            "cefr_level": metadata_dict["cefr_level"],
            "filename": os.path.basename(filepath),
            "topic": metadata_dict["topic"],
            "keywords": metadata_dict["keywords"],
            "text": chunk,
        }
        data_to_upsert.append((chunk_id, embedding, metadata))

    return data_to_upsert


def load_metadata(metadata_file):
    """Loads metadata from a JSON file."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {metadata_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {metadata_file}")
        return None


# --- Main Processing Loop ---
metadata_file = os.path.join(DATA_DIR, METADATA_PATTERN)

# Load metadata (Exit if there's an error)
all_metadata = load_metadata(metadata_file)
if all_metadata is None:
    exit() # stop execution if metadata loading fails


files_processed = 0
vectors_upserted = 0

for metadata_item in all_metadata:
    filepath = os.path.join(DATA_DIR, metadata_item["filename"])  # Full path to the text file
    print(f"Processing file: {filepath}")
    processed_data = process_file(filepath, metadata_item, model)

    if processed_data:
        index.upsert(vectors=processed_data)
        vectors_upserted += len(processed_data)
        files_processed += 1
        print(f"  Upserted {len(processed_data)} vectors from {filepath} to Pinecone.")

print(f"\nFinished processing.\nFiles processed: {files_processed}.\nVectors upserted: {vectors_upserted}")
index_stats = index.describe_index_stats()
print(f"Updated Index Stats: {index_stats}")
