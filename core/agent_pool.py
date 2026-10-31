import os
import asyncio


class AgentPool:
    def __init__(self, log_path):
        self.modality_agents = {}
        self.semantic_fusion_agent = None
        self.statistical_fusion_agent = None
        self.hybrid_fusion_agent = None
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

    def add_modality_agent(self, agent):
        self.modality_agents[agent.name] = agent

    def add_semantic_fusion_agent(self, agent):
        self.semantic_fusion_agent = agent

    def add_statistical_fusion_agent(self, agent):
        self.statistical_fusion_agent = agent

    def add_hybrid_fusion_agent(self, agent):
        self.hybrid_fusion_agent = agent

    def log_summary(self, message, print_log=True):
        path = os.path.join(self.log_path, "summary.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        if print_log:
            print(message)

    def get_last_responses(self):
        responses = {}
        for name, agent in self.modality_agents.items():
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
        max_key = [k for k, v in answer_cnt.items() if v == max_val][0]
        return max_key

    async def run_parallel_interpretation(self):
        tasks = []
        for _, agent in self.modality_agents.items():
            tasks.append(asyncio.create_task(agent.interpret()))
        results = await asyncio.gather(*tasks)
        if None in results:
            self.log_summary(f"[Error] Failed to interpret")
            return None

        for _, agent in self.modality_agents.items():
            agent.update_memory()

        responses = self.get_last_responses()
        answers = {k: v.get("ANSWER", "") for k, v in responses.items()}
        self.modality_outputs = answers  # expose for external checkpoint
        unique_answers = set(answers.values())

        print("\n[MODALITY OUTPUTS]")
        for k, v in answers.items():
            print(f"  {k}: {v}")
        print(f"  Unique answers: {unique_answers}  (disagreement={'YES' if len(unique_answers) > 1 else 'NO — all identical'})")

        for k, v in answers.items():
            self.log_summary(f"[Interpretation] <{k}> provided answer: {v}")

        majority = self.vote(answers)
        print(f"[MAJORITY ANSWER] {majority}")
        return majority

    async def run_semantic_fusion(self):
        responses = self.get_last_responses()
        result = await self.semantic_fusion_agent.fuse(responses)
        if result is None:
            self.log_summary("[Error] Semantic fusion failed")
            return None

        answer = result.get("ANSWER", "")
        self.semantic_fusion_agent.update_memory()
        return answer

    async def run_statistical_fusion(self):
        responses = self.get_last_responses()
        answers = {k: v.get("ANSWER", "") for k, v in responses.items()}
        final_answer = self.vote(answers)
        result = await self.statistical_fusion_agent.fuse(responses, final_answer)
        if result is None:
            self.log_summary("[Error] Statistical fusion failed")
            return None

        answer = result.get("ANSWER", "")
        print(f"\n[STAT FUSION RAW] full parsed dict: {result}")
        print(f"[STAT FUSION PARSED] ANSWER field: {repr(answer)}")
        print(f"[STAT FUSION] majority_answer hint was: {repr(final_answer)}  |  same={answer == final_answer}")
        self.statistical_fusion_agent.update_memory()
        return answer

    async def run_hybrid_fusion(self):
        responses = self.get_last_responses()
        semantic_response = self.semantic_fusion_agent.get_last_response()
        statistical_response = self.statistical_fusion_agent.get_last_response()

        result = await self.hybrid_fusion_agent.fuse(
            responses, semantic_response, statistical_response
        )
        if result is None:
            self.log_summary("[Error] Hybrid fusion failed")
            return None

        answer = result.get("ANSWER", "")
        self.hybrid_fusion_agent.update_memory()
        return answer
