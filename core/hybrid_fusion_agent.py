import json
from .agent import Agent


class HybridFusionAgent(Agent):
    def __init__(
        self,
        name="HybridFusionAgent",
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
            "You are a coordinator agent that solves a multimodal sensing task "
            "based on interpretations from multiple sensing agents.\n"
            "You have the following information about the task:\n"
            f"{self.task_info}\n\n"
            "You will receive reasonings and answers from multiple agents, "
            "each interpreting different modalities, "
            "as well as two fusion agents that have already aggregated these responses in different ways.\n"
            "Your goal is to coordinate the responses from the fusion agents "
            "and provide a final reasoned answer for the task."
        )
        self.set_system_message(content)

    async def fuse(self, responses, semantic_response, statistical_response):
        responses_info = json.dumps(responses, indent=2, sort_keys=True)
        semantic_info = json.dumps(semantic_response, indent=2, sort_keys=True)
        statistical_info = json.dumps(statistical_response, indent=2, sort_keys=True)

        content = (
            "You have received the following information.\n"
            "1. Original responses from individual modality agents:\n"
            f"{responses_info}\n\n"
            "2. Responses from fusion agents:\n"
            f"{semantic_info}\n\n"
            f"{statistical_info}\n\n"
            "Your task is to determine the responses from the fusion agents "
            "and provide a final reasoned answer for the task.\n"
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
