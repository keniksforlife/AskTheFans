import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import base64
from tempfile import NamedTemporaryFile
from google.oauth2 import service_account
from google.cloud import bigquery
from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
import openai

# Load environment variables
load_dotenv()

# App configuration
app = FastAPI()

OPENAI_KEY = os.getenv("OPENAI_KEY")
PINECONE_API = os.getenv("PINECONE_API")

# Global variables for asynchronous initialization
model = None
pc = None
index = None

@app.on_event("startup")
async def startup_event():
    global model, pc, index
    # Asynchronously load SentenceTransformer model
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Pinecone API setup
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", PINECONE_API)
    pc = Pinecone(api_key=pinecone_api_key)

    # Create or connect to a Pinecone index
    index_name = "askthefans"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=512, metric='cosine')
    index = pc.Index(index_name)

# def generate_vector(text):
#     embeddings = model.encode(text)  # This produces a vector
#     adjusted_embeddings = adjust_vector_dimension(embeddings, 512)  # Ensure 512 dimensions
#     return adjusted_embeddings.tolist()

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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Can you explain more about that?"}
        ]
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
    data = await request.json()
    question = data.get('question', '')
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    query_vector = question #generate_vector(question)
    query_results = index.query(vector=query_vector, top_k=1, include_metadata=True, filter={"answer": {"$ne": ""}})
    if query_results['matches']:
        match = query_results['matches'][0]
        metadata = match.get('metadata', {})
        if 'answer' in metadata:
            answer = metadata['answer']
            response_content = generate_response(question, answer)
            return JSONResponse(content={"question": question, "answer": response_content, "metadata": metadata})
        else:
            raise HTTPException(status_code=404, detail="Answer not found in metadata")
    else:
        raise HTTPException(status_code=404, detail="No relevant answers found")
