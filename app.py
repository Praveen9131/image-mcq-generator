import os
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

app = Flask(__name__)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

# Function to generate multiple image options based on a prompt
def generate_image_options(prompt: str, num_options: int = 4):
    options = []
    for i in range(num_options):
        option_prompt = f"{prompt} option {i+1}"
        image_url = generate_image(option_prompt)
        options.append(image_url)
    return options

# Function to generate a question with image options from an image URL using GPT-4 Vision
def generate_mcq_with_image_options(image_url: str):
    description_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are an expert in generating educational content. Describe the image and generate a prompt for generating four image options related to the image's content and you should follow the format like image and questionwith four options and correct answer."},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        ],
    }
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[description_prompt],
        max_tokens=1000,
        n=1,
        stop="",  # Set stop to an empty string or a specific stop sequence
        temperature=0.5
    )
    
    description = response.choices[0].message['content']
    options = generate_image_options(description)
    correct_answer = options[0]  # Assuming the first option is the correct one
    
    return {
        "question_image_url": image_url,
        "options": options,
        "correct_answer": correct_answer
    }

@app.route('/generate_content', methods=['GET'])
def generate_content():
    topic = request.args.get('topic')
    num_questions = int(request.args.get('num_questions'))

    images_and_questions = []
    for _ in range(num_questions):
        # Generate image
        image_prompt = f"An illustration representing the topic: {topic}"
        question_image_url = generate_image(image_prompt)
        
        # Generate MCQ with image options based on the question image
        mcq_with_images = generate_mcq_with_image_options(question_image_url)
        
        images_and_questions.append(mcq_with_images)

    return jsonify(images_and_questions)

if __name__ == "__main__":
    app.run(debug=True)
