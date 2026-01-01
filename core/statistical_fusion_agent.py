import json
from .agent import Agent


class StatisticalFusionAgent(Agent):
    def __init__(
        self,
        name="StatisticalFusionAgent",
        model_pool=None,
        task_info=None,
        classes_info=None,
        log_path=None,
    ):
        super().__init__(
            name=name,
            model_pool=model_pool,
            log_path=log_path,
        )

        self.task_info = task_info
        self.classes_info = classes_info
        self.init_system_message()

    def init_system_message(self):
        content = (
            "You are a fusion agent that solves a multimodal sensing task "
            "based on interpretations from multiple sensing agents.\n"
            "You have the following information about the task:\n"
            f"{self.task_info}\n\n"
            "You will receive reasonings and answers from multiple agents "
            "based on their interpretations of different modalities.\n"
            "Your goal is to provide a final reasoned answer for the task."
        )
        self.set_system_message(content)

    async def fuse(self, responses, majority_answer):
        responses_info = json.dumps(responses, indent=2, sort_keys=True)
        content = (
            "You have received responses from multiple sensing agents:\n"
            f"{responses_info}\n\n"
            f"You are on the side that the correct answer is {majority_answer} which is the majority answer.\n"
            "Based on the given reasonings and answers, provide a final reasoned answer for the task.\n"
            f"For agents who provided different answers from {majority_answer}, "
            "explain why their reasoning is likely affected by noise or unreliable signal interpretation.\n"
            f"Stay consistent with the position that the correct answer is likely {majority_answer}.\n"
            "Respond in the following strict JSON format:\n"
            "{\n"
            '  "REASON": "<Reasoning for the answer>",\n'
            f'  "ANSWER": "{majority_answer}",\n'
            "}\n\n"
            "Do not include any additional text outside the JSON."
        )
        response = await self.invoke(content)
        parsed_response = self.safe_parse_json(response)
        parsed_response = await self.validate_response(
            parsed_response, ["REASON", "ANSWER"]
        )
        return parsed_response
