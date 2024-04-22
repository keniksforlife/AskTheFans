from google.cloud import bigquery
from google.oauth2 import service_account
import re
import base64
from tempfile import NamedTemporaryFile
import os
from dotenv import load_dotenv
import emoji

load_dotenv()

def authenticate_google_cloud():
    credentials_base64 = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
    credentials_bytes = base64.b64decode(credentials_base64)
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(credentials_bytes)
        temp.flush()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp.name
    return service_account.Credentials.from_service_account_file(temp.name)

def create_bigquery_client():
    credentials = authenticate_google_cloud()
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    return client

def clean_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
    return text

def extract_content(question_text):
    pattern = re.compile(r"Can anyone help our Facebook Fan\? (.*?) #askyourbabyclub", re.DOTALL)
    match = pattern.search(question_text)
    if match:
        return match.group(1).strip()
    else:
        return question_text  # Fallback to original text if pattern does not match

def preprocess_and_combine_data():
    client = create_bigquery_client()

    # Fetch questions
    question_query = """
        SELECT post_id, post_text
        FROM `civic-badge-410308.askthefans.posts`
        GROUP BY post_id, post_text
        """
    question_results = client.query(question_query).result()

    for question_row in question_results:
        post_id = question_row['post_id']
        question_text = clean_text(question_row['post_text'])

        # Fetch related comments
        comment_query = f"""
        SELECT commenter_text FROM `civic-badge-410308.askthefans.comments` WHERE post_url LIKE '%{post_id}%'
        """
        comment_results = client.query(comment_query).result()

        # Combine and clean comments
        combined_comments = ' '.join([clean_text(comment_row['commenter_text']) for comment_row in comment_results])

        # Combine question and comments
        combined_text = question_text + " " + combined_comments

        # Print combined text for demonstration
        # print(combined_text)

        # Insert combined and cleaned text into another table (adjust the table name and field names as needed)
        insert_query = """
        INSERT INTO `civic-badge-410308.askthefans.cleaned_data` (post_id, question, answer)
        VALUES (?, ?, ?)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(None, "STRING", post_id),
                bigquery.ScalarQueryParameter(None, "STRING", question_text),
                bigquery.ScalarQueryParameter(None, "STRING", combined_comments)
            ]
        )
        query_job = client.query(insert_query, job_config=job_config)
        query_job.result()  # Wait for the query to finish

        print(f"Successfully inserted data for post_id: {post_id}")

preprocess_and_combine_data()
