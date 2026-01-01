import os
import re
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class Agent:
    def __init__(
        self,
        name,
        model_pool,
        log_path,
    ):
        self.name = name
        self.model_pool = model_pool
        self.log_path = log_path
        self.root_log_path = log_path
        self.agent_log_path = os.path.join(log_path, name)
        os.makedirs(self.agent_log_path, exist_ok=True)

        self.long_term_memory = []
        self.short_term_memory = []
        self.volatile_memory = []

    def log(self, message, local=True):
        path = os.path.join(self.root_log_path, "log.txt")
        with open(path, "a", encoding="utf-8") as f:
            message_type = "UNKNOWN"
            if isinstance(message, SystemMessage):
                message_type = "SYSTEM"
            if isinstance(message, HumanMessage):
                message_type = "HUMAN"
            if isinstance(message, AIMessage):
                message_type = "AI"
            content = message.content.strip()
            name = self.name
            f.write(f"[{name}] [{message_type}]\n{content}\n\n\n")
        if local:
            local_path = os.path.join(self.agent_log_path, "log.txt")
            with open(local_path, "a", encoding="utf-8") as f:
                f.write(f"[{name}] [{message_type}]\n{content}\n\n\n")

    def update_memory(self):
        self.long_term_memory.extend(self.short_term_memory)
        self.clean_short_term_memory()
        self.clean_volatile_memory()

    def clean_short_term_memory(self):
        self.short_term_memory = []

    def clean_volatile_memory(self):
        self.volatile_memory = []

    def clean_long_term_memory(self):
        self.long_term_memory = []

    def clean_json_text(self, text):
        text = text.strip()
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", "'").replace("”", "'")
        text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        text = "".join(ch for ch in text if ch.isprintable())
        text = text.replace("][", ",")
        return text

    def safe_parse_json(self, text):
        if not text:
            return None
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
            text = self.clean_json_text(text)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                print(f"[!] JSON parse failed: {e}")
                return None
        elif not text.endswith("}"):
            text += "}"
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
                text = self.clean_json_text(text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"[!] JSON parse failed: {e}")
                    return None
        print("[!] JSON parse failed")
        return None

    async def validate_response(self, response, fields, volatile=False):
        if (
            not response
            or not isinstance(response, dict)
            or not all(field in response for field in fields)
        ):
            print("[!] The JSON failed to be parsed. Trying again.")
            content = (
                "Failed to parse the JSON from the previous response. Please try again."
            )
            response = await self.invoke(content, volatile=volatile)
            response = self.safe_parse_json(response)
        if (
            not response
            or not isinstance(response, dict)
            or not all(field in response for field in fields)
        ):
            print("[!] Retry failed.")
            return None
        return response

    def get_last_response(self):
        if len(self.long_term_memory) >= 2:
            last_msg = self.long_term_memory[-1]
            if isinstance(last_msg, AIMessage):
                return self.safe_parse_json(last_msg.content)
        return None

    def set_system_message(self, content, local=True):
        system_message = SystemMessage(content=content)
        self.log(system_message, local)
        self.long_term_memory.append(system_message)

    async def invoke(self, content, volatile=False, local=True):
        messages = self.long_term_memory.copy()
        if volatile:
            messages.extend(self.volatile_memory)
        else:
            messages.extend(self.short_term_memory)
        messages.append(HumanMessage(content=content))
        try:
            response = await self.model_pool.invoke(messages)
            if volatile:
                self.volatile_memory.extend([HumanMessage(content=content), response])
            else:
                self.short_term_memory.extend([HumanMessage(content=content), response])
            local_ = not volatile and local
            self.log(HumanMessage(content=content), local=local_)
            self.log(response, local=local_)
            return response.content.strip()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[Error] Error occurred while invoking LLM: {e}")
