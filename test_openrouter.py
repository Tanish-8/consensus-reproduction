import os
import asyncio

from core.model import load_models


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    model_pool = load_models(
        ["openrouter:openrouter/free"],
        temperature=0.0,
        top_p=1.0,
        num_ctx=15000,
    )

    response = await model_pool.invoke(
        [
            {
                "role": "user",
                "content": "Reply with exactly: ASYNC_POOL_TEST_SUCCESS",
            }
        ]
    )

    print("Response:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
