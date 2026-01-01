import os
import numpy as np
import asyncio


class AgentPool:
    def __init__(self, log_path):
        self.agents = {}
        self.groups = []
        self.secretary_agent = None
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

    def add_agent(self, agent):
        self.agents[agent.name] = agent

    def add_secretary_agent(self, agent):
        self.secretary_agent = agent

    def group_agents(self, num_groups=2):
        self.groups = []
        agents = list(self.agents.keys())
        np.random.shuffle(agents)
        for i in range(num_groups):
            self.groups.append(agents[i::num_groups])

    def log_summary(self, message, print_log=True):
        path = os.path.join(self.log_path, "summary.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        if print_log:
            print(message)

    def get_last_responses(self):
        responses = {}
        for name, agent in self.agents.items():
            response = agent.get_last_response()
            if response:
                responses[name] = response
        return responses

    def vote(self, answers):
        answer_cnt = {}
        for _, answer in answers.items():
            if answer not in answer_cnt:
                answer_cnt[answer] = 0
            answer_cnt[answer] += 1
        max_val = max(answer_cnt.values())
        max_keys = [k for k, v in answer_cnt.items() if v == max_val]
        is_tie = len(max_keys) > 1
        return max_keys[0], is_tie

    async def run_parallel_interpretation(self):
        tasks = []
        for _, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.interpret()))
        results = await asyncio.gather(*tasks)
        if None in results:
            self.log_summary(f"[Error] Failed to interpret")
            return None

        for _, agent in self.agents.items():
            agent.update_memory()

        responses = self.get_last_responses()
        answers = {k: v.get("ANSWER", "") for k, v in responses.items()}
        for k, v in answers.items():
            self.log_summary(f"[Interpretation] <{k}> provided answer: {v}")
        return self.vote(answers)[0]

    async def run_parallel_debate(self):
        responses = self.get_last_responses()
        tasks = []
        for _, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.debate(responses, self.groups)))
        results = await asyncio.gather(*tasks)
        if None in results:
            self.log_summary(f"[Error] Failed to debate")
            return None

        for _, agent in self.agents.items():
            agent.update_memory()

        responses = self.get_last_responses()
        answers = {k: v.get("ANSWER", "") for k, v in responses.items()}
        for k, v in answers.items():
            self.log_summary(f"[Debate] <{k}> provided answer: {v}")
        return self.vote(answers)

    async def run_secretary_break_tie(self):
        responses = self.get_last_responses()
        result = await self.secretary_agent.break_tie(responses)
        if result is None or result.get("ANSWER", None) is None:
            self.log_summary(f"[Error] Failed to break tie")
            return None
        return result.get("ANSWER", None)
