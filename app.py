import os
import base64
import asyncio
import aiohttp
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import openai
from pymongo import MongoClient
from bson import ObjectId
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

app = Flask(__name__)

# MongoDB setup
MONGODB_URI = os.getenv('MONGODB_URI')

def get_mongo_client():
    return MongoClient(MONGODB_URI)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            return response['data'][0]['url']
        except openai.error.OpenAIError as e:
            if attempt < retry_attempts - 1:
                time.sleep(2)
                continue
            else:
                raise e

# Async function to download and resize an image
async def download_resize_image(url: str, size: tuple):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        image = Image.open(BytesIO(data))
                        image = image.resize(size, Image.Resampling.LANCZOS)
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG", quality=10)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        return img_str
        except aiohttp.ClientError as e:
            if attempt < retry_attempts - 1:
                time.sleep(2)
                continue
            else:
                raise e
    return None

# Function to store image data in MongoDB
def store_image_data(original_image_data: str, resized_image_data: str, mongo_client):
    document = {
        "original_image": original_image_data,
        "resized_image": resized_image_data
    }
    result = mongo_client['images']['image_data'].insert_one(document)
    return result.inserted_id

# Function to generate a detailed description of the image using GPT-4 Vision
def generate_image_description(image_url: str):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image in detail."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message['content']
        except openai.error.OpenAIError as e:
            if attempt < retry_attempts - 1:
                time.sleep(2)
                continue
            else:
                raise e

# Function to generate a question from a detailed description using GPT-4
def generate_mcq_from_description(description: str, tone: str, subject: str):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            prompt = [
                {"role": "system", "content": "You are an expert in generating educational content."},
                {"role": "user", "content": f"Based on the following detailed description of an image related to the topic '{subject}', generate a multiple-choice question in a {tone} tone with 4 options and provide the correct answer: {description}"}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=prompt,
                max_tokens=1000,
                temperature=0.5
            )
            return response.choices[0].message['content']
        except openai.error.OpenAIError as e:
            if attempt < retry_attempts - 1:
                time.sleep(2)
                continue
            else:
                raise e

# Consolidated function to generate image and MCQ
async def generate_image_mcq(number: int, subject: str, tone: str):
    images_and_questions = []
    mongo_client = get_mongo_client()
    for i in range(number):
        # Generate image
        image_prompt = f"An illustration representing the topic: {subject}"
        image_url = generate_image(image_prompt)
        
        # Resize and encode image
        original_image_data = await download_resize_image(image_url, (1024, 1024))
        resized_image_data = await download_resize_image(image_url, (750, 319))
        
        if original_image_data and resized_image_data:
            # Store image data in MongoDB
            image_id = store_image_data(original_image_data, resized_image_data, mongo_client)
            
            # Generate a detailed description of the image
            description = generate_image_description(image_url)
            
            # Generate MCQ based on the detailed description
            mcq_text = generate_mcq_from_description(description, tone, subject)
            
            images_and_questions.append({
                'mcq': mcq_text,
                'question_image_id': str(image_id),
                'question_image_url': image_url,  # Original URL
                'resized_image_url': f'/image/{image_id}'  # URL to access the resized image
            })
    mongo_client.close()
    return images_and_questions

@app.route('/generate_content', methods=['GET'])
def generate_content():
    number = request.args.get('number')
    subject = request.args.get('subject')
    tone = request.args.get('tone')

    if not number or not subject or not tone:
        return jsonify({"error": "Missing required parameters: number, subject, and tone"}), 400

    try:
        number = int(number)
    except ValueError:
        return jsonify({"error": "number must be an integer"}), 400

    try:
        content = asyncio.run(generate_image_mcq(number, subject, tone))
        return jsonify(content)
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/image/<image_id>', methods=['GET'])
def get_image(image_id):
    try:
        mongo_client = get_mongo_client()
        document = mongo_client['images']['image_data'].find_one({"_id": ObjectId(image_id)})
        if not document:
            return jsonify({"error": "Image not found"}), 404
        image_data = base64.b64decode(document['resized_image'])
        return send_file(BytesIO(image_data), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Image not found: {e}"}), 404
    finally:
        mongo_client.close()

if __name__ == '__main__':
    app.run(debug=True)
