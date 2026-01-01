import json
from .agent import Agent


class SemanticFusionAgent(Agent):
    def __init__(
        self,
        name="SemanticFusionAgent",
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

    async def fuse(self, responses):
        responses_info = json.dumps(responses, indent=2, sort_keys=True)
        content = (
            "You have received responses from multiple sensing agents:\n"
            f"{responses_info}\n\n"
            "Using your own knowledge and expertise, "
            "analyze the reasonings and answers and provide a final reasoned answer.\n"
            "Respond in the following strict JSON format:\n"
            "{\n"
            '  "REASON": "<Reasoning for the answer>",\n'
            f'  "ANSWER": "<Answer among {self.classes_info}>",\n'
            "}\n\n"
            "Do not include any additional text outside the JSON."
        )
        response = await self.invoke(content)
        parsed_response = self.safe_parse_json(response)
        parsed_response = await self.validate_response(
            parsed_response, ["REASON", "ANSWER"]
        )
        return parsed_response
