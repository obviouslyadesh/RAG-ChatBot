import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_response(prompt: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using only the context below.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content
