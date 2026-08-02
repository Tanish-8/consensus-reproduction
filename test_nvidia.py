
from openai import OpenAI

print("OpenAI imported")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="REDACTED_NVIDIA_KEY"
)


response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: NVIDIA API Working"
        }
    ],
    temperature=0.0,
    top_p=1.0,
    max_tokens=20,
)

print(response.choices[0].message.content)