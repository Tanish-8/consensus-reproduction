import os
import asyncio


class AgentPool:
    def __init__(self, log_path):
        self.agent = None
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

    def add_agent(self, agent):
        self.agent = agent

    def log_summary(self, message, print_log=True):
        path = os.path.join(self.log_path, "summary.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        if print_log:
            print(message)

    async def run_parallel_interpretation(self):
        tasks = []
        tasks.append(asyncio.create_task(self.agent.interpret()))
        results = await asyncio.gather(*tasks)
        if None in results:
            self.log_summary(f"[Error] Failed to interpret")
            return None

        self.agent.update_memory()

        response = self.agent.get_last_response()
        answer = response.get("ANSWER", "")
        self.log_summary(f"[Interpretation] provided answer: {answer}")
        return answer
