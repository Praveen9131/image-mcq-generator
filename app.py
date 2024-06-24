import os
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai.api_key = openai_api_key

app = Flask(__name__)

# Create a ThreadPoolExecutor for asynchronous task execution
executor = ThreadPoolExecutor(max_workers=10)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

# Asynchronous wrapper for generate_image
async def async_generate_image(prompt: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, partial(generate_image, prompt))

# Function to generate multiple image options based on a prompt
async def generate_image_options(prompts: list):
    tasks = [async_generate_image(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Function to generate a question with image options based on a description
async def generate_mcq_with_image_options(description: str):
    description_prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Generate a multiple-choice question with four options based on the following description. Use the following format:\n\n**Question:** [Question based on the description]\n\n**Options:**\n1. [Option 1]\n2. [Option 2]\n3. [Option 3]\n4. [Option 4]\n\n**Correct Answer:** [Correct Option]\n\nDescription: {description}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=description_prompt,
        max_tokens=1000,
        temperature=0.5
    )
    content = response.choices[0].message['content']
    
    question_section = content.split("**Question:**")[1].split("**Options:**")[0].strip()
    options_section = content.split("**Options:**")[1].split("**Correct Answer:**")[0].strip()
    correct_answer = content.split("**Correct Answer:**")[1].strip()

    options = options_section.split('\n')
    option_prompts = [option.split('. ')[1] for option in options]

    option_images = await generate_image_options(option_prompts)
    
    correct_answer_index = option_prompts.index(correct_answer)
    
    return {
        "question": question_section,
        "options": {
            "Option 1": option_images[0],
            "Option 2": option_images[1],
            "Option 3": option_images[2],
            "Option 4": option_images[3]
        },
        "correct_answer": f"Option {correct_answer_index + 1}"
    }

@app.route('/generate_content', methods=['GET'])
async def generate_content():
    topic = request.args.get('topic')
    num_questions = int(request.args.get('num_questions'))

    tasks = []
    for _ in range(num_questions):
        image_prompt = f"An illustration representing the topic: {topic}"
        question_image_url = await async_generate_image(image_prompt)

        description = f"This is an illustration representing the topic '{topic}'."
        mcq_with_images_task = generate_mcq_with_image_options(description)
        tasks.append(mcq_with_images_task)

    images_and_questions = await asyncio.gather(*tasks)
    for mcq_with_images in images_and_questions:
        mcq_with_images["question_image_url"] = question_image_url

    return jsonify(images_and_questions)

if __name__ == "__main__":
    # Increase worker timeout and scale server resources
    from waitress import serve
    logger.info("Starting server on http://0.0.0.0:8080")
    serve(app, host='0.0.0.0', port=8080, threads=6)
