import os
import asyncio

from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


def load_models(models, temperature=0.0, top_p=1.0, num_ctx=15000):
    model_pool = AsyncModelPool()
    for model in models:
        model_pool.add_model(
            Model(model, temperature=temperature, top_p=top_p, num_ctx=num_ctx)
        )
    model_pool.init_models()
    return model_pool


class Model:
    def __init__(self, model, temperature, top_p, num_ctx):
        if model.startswith("ollama:"):
            model = model.replace("ollama:", "")
            if "url:" in model:
                model = model.replace("url:", "")
                base_url = model.split("/")[0]
                model_type = model.split("/")[1]
                self.model = ChatOllama(
                    model=model_type,
                    base_url=base_url,
                    temperature=temperature,
                    top_p=top_p,
                    num_ctx=num_ctx,
                )
            else:
                self.model = ChatOllama(
                    model=model.replace("ollama:", ""),
                    temperature=temperature,
                    top_p=top_p,
                    num_ctx=num_ctx,
                )
        elif model.startswith("together"):
            if "TOGETHER_API_KEY" not in os.environ:
                print("[!] TOGETHER_API_KEY is not set")
                assert 0
            self.model = ChatTogether(
                model=model.replace("together:", ""),
                temperature=temperature,
                top_p=top_p,
                max_tokens=num_ctx,
                max_retries=3,
            )
        elif model.startswith("openai"):
            if "OPENAI_API_KEY" not in os.environ:
                print("[!] OPENAI_API_KEY is not set")
                assert 0
            self.model = ChatOpenAI(
                model=model.replace("openai:", ""),
                temperature=temperature,
                top_p=top_p,
            )
        else:
            self.model = init_chat_model(
                model=model,
                temperature=temperature,
            )

    def invoke(self, messages):
        try:
            response = self.model.invoke(messages)
            return response
        except Exception as e:
            print(f"[Error] Error occurred while invoking LLM: {e}")
            return e


class AsyncModel:
    def __init__(self, model):
        self.model = model

    async def invoke(self, content):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.invoke(content),
        )
        return response


class AsyncModelPool:
    def __init__(self):
        self.models = []
        self._available_models = None
        self._model_semaphore = None

    def add_model(self, model):
        self.models.append(model)

    def init_models(self):
        print(f"Initializing {len(self.models)} models...")
        self._available_models = asyncio.Queue()
        for model in self.models:
            async_model = AsyncModel(model)
            self._available_models.put_nowait(async_model)
        self._model_semaphore = asyncio.Semaphore(len(self.models))

    async def invoke(self, content):
        if self._available_models is None:
            raise RuntimeError("Model pool not initialized. Call init_models() first.")
        async_model = await self._available_models.get()
        try:
            response = await async_model.invoke(content)
            return response
        finally:
            self._available_models.put_nowait(async_model)
