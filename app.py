from flask import Flask, request, jsonify
import pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration (Load from .env) ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_HOST = os.environ.get("PINECONE_HOST")  # Hostname is loaded from .env now
INDEX_NAME = "cefr-text-index"  # Or get this from .env if you prefer
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize Pinecone and Sentence Transformer ---
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Pinecone and Sentence Transformer initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    # Optionally, exit or handle the error appropriately if initialization fails
    # exit()

# --- API Endpoint ---
@app.route('/get_context', methods=['GET'])
def get_context():
    """
    Retrieves context (filenames) from Pinecone based on query parameters.
    """
    try:
        # Get parameters from the query string
        workflow = request.args.get('workflow')  # "bespoke" or "differentiated"
        cefr_level = request.args.get('cefr_level')  # e.g., "A2", "B1"
        topic = request.args.get('topic')
        keywords_str = request.args.get('keywords', '')  # Get as string, default empty
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]  # Split into list
        length = request.args.get('length')  # "short", "medium", "long"
        #main_text = request.args.get('main_text')  # We'll handle this later, if needed

        # --- Input Validation ---
        if not workflow or workflow not in ["bespoke", "differentiated"]:
            return jsonify({"status": "error", "message": "Invalid 'workflow' parameter."}), 400
        if not cefr_level or cefr_level not in ["A1", "A2", "B1", "B2", "C1"]:
            return jsonify({"status": "error", "message": "Invalid 'cefr_level' parameter."}), 400
        if not topic:
            return jsonify({"status": "error", "message": "Missing 'topic' parameter."}), 400
        #  You could add validation for 'length' as well.

        print(f"Received request: workflow={workflow}, cefr_level={cefr_level}, topic={topic}, keywords={keywords}, length={length}") # Debug print

        # --- Query Pinecone ---
        metadata_filter = {
            "cefr_level": {"$eq": cefr_level},
            "topic": {"$eq": topic}
        }
        if keywords:
             metadata_filter["keywords"] = {"$in": keywords}

        print(f"Pinecone filter: {metadata_filter}") # Debug print
        query_vector = model.encode(topic).tolist()

        query_results = index.query(
            vector=query_vector,
            top_k=20,  # Query for more initially to increase chance of getting unique files
            filter=metadata_filter,
            include_metadata=True
        )

        print(f"Pinecone query results (raw): {query_results}") # Debug print

        # --- Extract UNIQUE filenames ---
        unique_filenames = []
        seen_filenames = set()
        for match in query_results.matches:
            filename = match.metadata.get('filename')
            if filename and filename not in seen_filenames:
                unique_filenames.append(filename)
                seen_filenames.add(filename)
                if len(unique_filenames) >= 5: # Limit to returning 5 unique filenames
                    break

        print(f"Extracted unique filenames: {unique_filenames}") # Debug print

        # --- Return the Results ---
        return jsonify({"status": "success", "filenames": unique_filenames}), 200 # Return unique list

    except Exception as e:
        print(f"Error processing request: {e}") # Debug print for errors
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
