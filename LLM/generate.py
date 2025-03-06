import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import logging

# Tải các biến môi trường từ file .env
load_dotenv()

# Khởi tạo client OpenAI với khóa API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain (query):
    prompt = f"""
            You are provided with a job title {query}, please describe the job briefly.
            For example, job title is "Air commodore"
            The describe is "Commissioned armed forces officers provide leadership and management to organizational units in the armed forces and/or perform similar tasks to those performed in a variety of civilian occupations outside the armed forces."
            """
    
    retries = 5
    for attempt in range(retries):
        try:
            # Gọi API với client chat completion
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        except Exception as e:
            logging.error(f"API request failed: {e}")
            if attempt < retries - 1:
                time.sleep(60)  # Exponential backoff
            else:
                return "None"

