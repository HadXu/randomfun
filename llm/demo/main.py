from openai import OpenAI

client = OpenAI(
    base_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/",
    api_key="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"
)

PROMPT = """
Write a short, imperative description of the provided app's purpose. It MUST ALWAYS be under 80 characters and a single-sentence. You can mention some technology names that you extract from the source code.
Example descriptions: "Remove background from images.", "Generate captions for images using ViT and GPT2.", "Predict the nutritional value of food based on an image of the food."
The provided app.py file:
"""

import requests


space_id = "julien-c/coqui"

app_py = requests.get(url=f"https://huggingface.co/spaces/{space_id}/resolve/main/app.py").text

print(app_py)

input = PROMPT + f"```py{app_py}```"

chat_completion = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": input},
    ],
    max_tokens=500,
)
output = chat_completion.choices[0].message.content + "\n"
print(output)