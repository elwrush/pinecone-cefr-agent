from flask import Flask, request, jsonify
import pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re # Import regular expression module for parsing

# --- Configuration (Load from .env) ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "cefr-text-index"
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
    # exit() # Consider exiting if essential components fail

# --- Helper Function to Parse User Message ---
def parse_user_message(message):
    """
    Parses the user's message to extract parameters.
    Assumes format like:
    Workflow: bespoke
    Level: A2
    Topic: Some Topic
    Keywords: key1, key2
    Length: 200
    Main Text: Optional text...
    """
    params = {
        'workflow': None,
        'cefr_level': None,
        'topic': None,
        'keywords': [],
        'length': None,
        'main_text': None
    }
    # Use regex to find key-value pairs, handling multiline Main Text
    pattern = re.compile(r"^\s*([\w\s]+):\s*(.*?)\s*$", re.MULTILINE | re.IGNORECASE)
    matches = pattern.findall(message)

    # Handle potential multiline Main Text explicitly
    main_text_start_index = message.lower().find("main text:")
    if main_text_start_index != -1:
        main_text_content = message[main_text_start_index + len("main text:") :].strip()
        params['main_text'] = main_text_content if main_text_content else None
        # Remove Main Text from matches to avoid double parsing
        matches = [(k,v) for k,v in matches if k.lower().strip() != 'main text']


    for key, value in matches:
        key_lower = key.lower().strip().replace(" ", "_") # Normalize key
        value = value.strip()

        if key_lower == "workflow" and value.lower() in ["bespoke", "differentiated"]:
            params['workflow'] = value.lower()
        elif key_lower == "level" and value.upper() in ["A1", "A2", "B1", "B2", "C1"]:
            params['cefr_level'] = value.upper()
        elif key_lower == "topic" and value:
            params['topic'] = value
        elif key_lower == "keywords" and value:
            params['keywords'] = [k.strip() for k in value.split(',') if k.strip()]
        elif key_lower == "length" and value.isdigit():
            params['length'] = value # Keep as string or convert to int if needed later

    # Basic validation
    if not params['workflow'] or not params['cefr_level'] or not params['topic']:
         print(f"Parsing failed or missing required fields: {params}") # Debug print
         return None # Indicate failure if essential fields are missing

    return params


# --- API Endpoint ---
@app.route('/get_context', methods=['GET'])
def get_context():
    """
    Retrieves context (filenames) from Pinecone based on parameters
    parsed from the user_message.
    """
    try:
        user_message = request.args.get('user_message')
        if not user_message:
            return jsonify({"status": "error", "message": "Missing 'user_message' parameter."}), 400

        print(f"Received user_message: {user_message}") # Debug print

        # --- Parse the user message ---
        params = parse_user_message(user_message)
        if not params:
             return jsonify({"status": "error", "message": "Could not parse required parameters from user_message."}), 400

        # Extract parsed parameters
        workflow = params['workflow']
        cefr_level = params['cefr_level']
        topic = params['topic']
        keywords = params['keywords']
        length = params['length']
        main_text = params['main_text'] # Extracted but not used in query yet

        print(f"Parsed parameters: workflow={workflow}, cefr_level={cefr_level}, topic={topic}, keywords={keywords}, length={length}") # Debug print

        # --- Query Pinecone ---
        metadata_filter = {
            "cefr_level": {"$eq": cefr_level},
            "topic": {"$eq": topic}
        }
        if keywords:
             metadata_filter["keywords"] = {"$in": keywords}

        print(f"Pinecone filter: {metadata_filter}") # Debug print

        # Use the topic for embedding search
        query_vector = model.encode(topic).tolist()

        query_results = index.query(
            vector=query_vector,
            top_k=20,
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
                if len(unique_filenames) >= 5:
                    break

        print(f"Extracted unique filenames: {unique_filenames}") # Debug print

        # --- Return the Results ---
        # Format the response as JSON string for TypingMind (as per their docs recommendation)
        response_data = {"status": "success", "filenames": unique_filenames}
        return jsonify(response_data), 200 # jsonify handles JSON formatting

    except Exception as e:
        print(f"Error processing request: {e}") # Debug print for errors
        # Ensure error message is JSON serializable
        error_message = str(e)
        return jsonify({"status": "error", "message": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

