import os
import sys
from openai import OpenAI


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[Error] NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("OpenAI imported")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
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


if __name__ == "__main__":
    main()
