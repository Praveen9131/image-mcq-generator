import os
import logging
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set")
openai.api_key = openai_api_key

app = Flask(__name__)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# Function to generate multiple image options based on a prompt
def generate_image_options(prompts: list):
    options = []
    for prompt in prompts:
        image_url = generate_image(prompt)
        if image_url:
            options.append(image_url)
        else:
            logger.error(f"Failed to generate image for prompt: {prompt}")
    return options

# Function to generate a question with image options based on a description
def generate_mcq_with_image_options(description: str):
    description_prompt = [
        {"role": "system", "content": "You are an expert in generating  content."},
        {"role": "user", "content": f" you are expert in to generate a multiple-choice question with four options based on the following description. Use the following format:\n\n**Question:** [Question based on the description]\n\n**Options:**\n1. [Option 1]\n2. [Option 2]\n3. [Option 3]\n4. [Option 4]\n\n**Correct Answer:** [Correct Option]\n\nDescription: {description}"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=description_prompt,
            max_tokens=1000,
            temperature=0.5
        )
        content = response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error generating MCQ with image options: {e}")
        return {"error": "Failed to generate MCQ"}
    
    try:
        question_section = content.split("**Question:**")[1].split("**Options:**")[0].strip()
        options_section = content.split("**Options:**")[1].split("**Correct Answer:**")[0].strip()
        correct_answer = content.split("**Correct Answer:**")[1].strip()

        options = options_section.split('\n')
        option_prompts = [option.split('. ')[1] for option in options]

        option_images = generate_image_options(option_prompts)
        
        if correct_answer not in option_prompts:
            raise ValueError(f"Correct answer '{correct_answer}' not found in options: {option_prompts}")

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
    except IndexError as e:
        logger.error(f"Error processing response: {e}")
        logger.error(f"Response content: {content}")
        return {
            "error": "Failed to parse the response from OpenAI",
            "response_content": content
        }
    except ValueError as e:
        logger.error(f"Error: {e}")
        return {
            "error": str(e),
            "response_content": content
        }

@app.route('/generate_content', methods=['GET'])
def generate_content():
    try:
        topic = request.args.get('topic')
        num_questions = int(request.args.get('num_questions'))

        images_and_questions = []
        for _ in range(num_questions):
            image_prompt = f"An illustration representing the topic: {topic}"
            question_image_url = generate_image(image_prompt)
            if not question_image_url:
                return jsonify({"error": "Failed to generate question image"}), 500

            description = f"This is an illustration representing the topic '{topic}'."
            mcq_with_images = generate_mcq_with_image_options(description)
            if "error" in mcq_with_images:
                return jsonify(mcq_with_images), 500

            mcq_with_images["question_image_url"] = question_image_url
            images_and_questions.append(mcq_with_images)

        return jsonify(images_and_questions)
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
