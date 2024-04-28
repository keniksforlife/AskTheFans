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
from pinecone import Pinecone
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
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Pinecone API setup
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", PINECONE_API)
    pc = Pinecone(api_key=pinecone_api_key)

    # Create or connect to a Pinecone index
    index_name = "askthefans"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=512, metric='cosine')
    index = pc.Index(index_name)

def generate_vector(text):
    embeddings = model.encode(text)  # This produces a vector
    adjusted_embeddings = adjust_vector_dimension(embeddings, 512)  # Ensure 512 dimensions
    return adjusted_embeddings.tolist()

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
    query = "SELECT post_id, question, answer FROM `civic-badge-410308.askthefans.cleaned_data`"
    results = client.query(query).result()
    return [(row['post_id'], row['question'], row['answer']) for row in results]

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

@app.get("/health")
async def health_check():
    return {"status": "alive"}

@app.get("/ready")
async def readiness_check():
    if model is None or pc is None or index is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

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
    query_results = index.query(vector=query_vector, top_k=1, include_metadata=True, filter={"answer": {"$ne": ""}})
    if query_results['matches']:
        match = query_results['matches'][0]
        metadata = match.get('metadata', {})
        match_score = match.get('score', 0)

        logging.info("Match Score: {match_score}")

         # Check if the score is above a defined threshold
        if match_score < 0.8:  # Example threshold
            logging.info("Match score is too low.{match_score}")
            raise HTTPException(status_code=404, detail="Match confidence too low")

        if 'answer' in metadata:
            answer = metadata['answer']
           
            logging.info("Generating response using the provided answer.")
            response_content = generate_response(question, answer)
            return JSONResponse(content={"question": question, "answer": response_content, "metadata": metadata})
        else:
            logging.error("Answer not found in metadata.")
            raise HTTPException(status_code=404, detail="Answer not found in metadata")
    else:
        logging.error("No relevant answers found for the question.")
        raise HTTPException(status_code=404, detail="No relevant answers found")
