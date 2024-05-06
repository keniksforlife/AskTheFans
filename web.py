import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
from tempfile import NamedTemporaryFile
from google.oauth2 import service_account
from google.cloud import bigquery
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
import logging

# Load environment variables
load_dotenv()

# App configuration
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify domains for Botpress server if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


OPENAI_KEY = os.getenv("OPENAI_KEY")
PINECONE_API = os.getenv("PINECONE_API")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables for asynchronous initialization
model = None
pc = None
index = None

@app.on_event("startup")
async def startup_event():
    global model, pc, index
    # Asynchronously load SentenceTransformer model
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logging.info("Model loaded successfully.")
    except Exception as e:
        model = None
        logging.error(f"Failed to load model: {e}")

    logging.info('hello')
    
    # Pinecone API setup
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", PINECONE_API)
    pc = Pinecone(api_key=pinecone_api_key)

    # Create or connect to a Pinecone index
    index_name = "askthefans"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=512, metric='cosine', spec=ServerlessSpec(cloud="aws",region="us-east-1"))
    index = pc.Index(index_name)

def generate_vector2(text):
    embeddings = model.encode(text)  # This produces a vector
    adjusted_embeddings = adjust_vector_dimension(embeddings, 512)  # Ensure 512 dimensions
    return adjusted_embeddings.tolist()

def generate_vector(text):
    if text is None or text.strip() == "":
        logging.error("Empty or None text provided to generate_vector")
        return []

    try:
        embeddings = model.encode(text)  # This produces a vector
        if embeddings is None:
            logging.error("Model returned None embeddings")
            return []
        adjusted_embeddings = adjust_vector_dimension(embeddings, 512)  # Ensure 512 dimensions
        return adjusted_embeddings.tolist()
    except Exception as e:
        logging.error(f"Error during vector generation: {e}")
        return []

def adjust_vector_dimension(vector, target_dim=512):
    current_dim = vector.shape[0]
    if current_dim < target_dim:
        padding = np.zeros(target_dim - current_dim)
        vector = np.concatenate((vector, padding))
    elif current_dim > target_dim:
        vector = vector[:target_dim]
    return vector

def authenticate_google_cloud():
    credentials_base64 = os.environ["GOOGLE_CREDENTIALS_BASE64"]
    credentials_bytes = base64.b64decode(credentials_base64)
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(credentials_bytes)
        temp.flush()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp.name
    return service_account.Credentials.from_service_account_file(temp.name)

def fetch_data_from_bigquery():
    credentials = authenticate_google_cloud()
    client = bigquery.Client(credentials=credentials)
    query = "SELECT post_id, question, answer, answer_index FROM `civic-badge-410308.askthefans.question_answers`"
    results = client.query(query).result()
    return [(row['post_id'], row['question'], row['answer'], row['answer_index']) for row in results]

def process_and_upsert_data():
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        data = fetch_data_from_bigquery()
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        return


    max_bytes = 38000  # Maximum size of metadata in bytes

    for post_id, post_text, comment_text, answer_index in data:
        max_length_post = max_bytes // 2
        max_length_comment = max_bytes // 2

        try:
            encoded_post_text = post_text.encode('utf-8')
            encoded_comment_text = comment_text.encode('utf-8')
        except UnicodeEncodeError as e:
            logging.error(f"Encoding error for post ID {post_id}: {e}")
            continue

        if len(encoded_post_text) > max_length_post:
            post_text = post_text.encode('utf-8')[:max_length_post].decode('utf-8', 'ignore')
        
        if len(encoded_comment_text) > max_length_comment:
            comment_text = comment_text.encode('utf-8')[:max_length_comment].decode('utf-8', 'ignore')

        combined_text = post_text + " " + comment_text
        # logging.info(combined_text)

        try:
            combined_vector = generate_vector(combined_text)
        except Exception as e:
            logging.error(f"Failed to generate vector for post ID {post_id}: {e}")
            continue

        combined_vector_id = f"qa_{post_id}_{answer_index}"
        print("processing " + combined_vector_id)

        try:
            response = index.upsert(vectors=[
                (combined_vector_id, combined_vector, {
                    "post_id": post_id, 
                    "question": post_text, 
                    "answer": comment_text
                })
            ])
            print(f"Upserted post ID {post_id} with response: {response}")
        except Exception as e:
            logging.error(f"Error upserting post ID {post_id}: {e}")
            logging.info(f"Failed text: Question - {post_text} | Answer - {comment_text}")

def setup_openai(api_key):
    openai.api_key = api_key

def generate_response(question, answer):
    setup_openai(OPENAI_KEY)

    system_message = """You are a knowledgeable assistant. Summarize the provided experiences and give informed advice 
                      that considers these specifics. Offer empathy and understand the complexity of the user's situation. 
                      Provide direct, clear, and helpful advice without prefacing your response with apologies or qualifiers. 
                      Focus on delivering factual information and practical recommendations."""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        temperature=0.5,  # This temperature promotes precision and clarity
        max_tokens=250    # Constrained to ensure brevity
    )
    last_message = response['choices'][0]['message']['content']
    return last_message

def simplify_response(original_text):
    setup_openai(OPENAI_KEY)
    
    try:
        # Request to the AI model to correct grammar and spelling on
        # ly
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "This is a simple task. Correct spelling and grammar errors."},
            {"role": "user", "content": original_text}
        ],
        # max_tokens=len(original_text.split()) + 10,  # Slightly more tokens to allow for corrections
        temperature=0.1  # Low temperature to minimize creativity and maintain originality
        )
        corrected_text = response['choices'][0]['message']['content']
        return corrected_text
    except Exception as e:
        # Log errors and return original text if there's a problem
        logging.error(f"Failed to simplify response: {e}")
        return original_text
    
@app.get("/health")
async def health_check():
    return {"status": "alive"}

@app.get("/ready")
async def readiness_check():
    if model is None or pc is None or index is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@app.get("/upsert")
async def upsert():
    process_and_upsert_data()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/query")
async def run_query(request: Request):
    logging.info("Received a new query request.")
    data = await request.json()
    question = data.get('question', '')
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    logging.info("Generating vector for the question.")
    query_vector = generate_vector(question)

    logging.info("Executing Pinecone query.")
    top_k_results = 5  # Adjust based on desired number of results
    query_results = index.query(vector=query_vector, top_k=top_k_results, include_metadata=True, filter={"answer": {"$ne": ""}})
    responses = []
    seen_answers = set()  # Track seen answers to avoid duplicates

    for match in query_results['matches']:
        metadata = match.get('metadata', {})
        match_score = match.get('score', 0)

        logging.info(f"Match Score: {match_score}")
        logging.info(metadata)

        if match_score < 0.6:  # Threshold for relevance
            logging.info(f"Match score {match_score} is too low for post ID {metadata.get('post_id')}.")
            continue

        answer = metadata.get('answer', "")
        if len(answer.strip()) < 30:  # Minimum character length filter
            logging.info(f"Filtered out too short answer: {answer}")
            continue

        if answer in seen_answers:  # Check for duplicates
            logging.info(f"Duplicate answer filtered out: {answer}")
            continue

        seen_answers.add(answer)  # Mark this answer as seen
        related_question = metadata.get('question', 'No related question provided')

        logging.info("Generating response using the provided answer.")
        response_content = simplify_response(answer)

        responses.append({
            "text": response_content,
            "original": answer,
            "related_question": related_question,
            "post_id": metadata.get('post_id'),
            "score": match_score
        })

    if not responses:
        logging.error("No relevant answers found for the question.")
        raise HTTPException(status_code=404, detail="No relevant answers found")

    # Include the count of answers in the response
    return JSONResponse(content={
        "question": question,
        "answers_count": len(responses),  # Count of the answers
        "answers": responses
    })
