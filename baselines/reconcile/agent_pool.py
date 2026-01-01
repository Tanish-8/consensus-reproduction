import os
import asyncio


class AgentPool:
    def __init__(self, log_path):
        self.agents = {}
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

    def add_agent(self, agent):
        self.agents[agent.name] = agent

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

    def vote(self, answers, confidences):
        answer_cnt = {}
        for k, answer in answers.items():
            confidence = confidences[k]
            if not isinstance(answer, str):
                continue
            if answer not in answer_cnt:
                answer_cnt[answer] = 0
            answer_cnt[answer] += confidence
        max_val = max(answer_cnt.values())
        max_key = [k for k, v in answer_cnt.items() if v == max_val][0]
        return max_key

    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, returning default if conversion fails."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def recalibrate_confidences(self, confidences):
        for k, v in confidences.items():
            if v >= 0.9 and v < 1.0:
                confidences[k] = 0.8
            elif v >= 0.8 and v < 0.9:
                confidences[k] = 0.5
            elif v > 0.6 and v < 0.8:
                confidences[k] = 0.3
            else:
                confidences[k] = 0.1
        return confidences

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
        confidences = {
            k: self._safe_float(v.get("CONFIDENCE", 0.0)) for k, v in responses.items()
        }
        for k, v in answers.items():
            confidence = confidences[k]
            self.log_summary(
                f"[Interpretation] <{k}> provided answer: {v} (confidence: {confidence})"
            )
        recalibrated_confidences = self.recalibrate_confidences(confidences)
        return self.vote(answers, recalibrated_confidences)

    async def run_parallel_debate(self):
        responses = self.get_last_responses()
        tasks = []
        for _, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.debate(responses)))
        results = await asyncio.gather(*tasks)
        if None in results:
            self.log_summary(f"[Error] Failed to debate")
            return None

        for _, agent in self.agents.items():
            agent.update_memory()

        responses = self.get_last_responses()
        answers = {k: v.get("ANSWER", "") for k, v in responses.items()}
        confidences = {
            k: self._safe_float(v.get("CONFIDENCE", 0.0)) for k, v in responses.items()
        }
        for k, v in answers.items():
            confidence = confidences[k]
            self.log_summary(
                f"[Debate] <{k}> provided answer: {v} (confidence: {confidence})"
            )
        recalibrated_confidences = self.recalibrate_confidences(confidences)
        return self.vote(answers, recalibrated_confidences)
