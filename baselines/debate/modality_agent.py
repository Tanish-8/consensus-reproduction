import json
from core.agent import Agent


class ModalityAgent(Agent):
    def __init__(
        self,
        name,
        model_pool,
        task_info,
        classes_info,
        modality_info,
        sample,
        examples,
        log_path,
    ):
        super().__init__(
            name=name,
            model_pool=model_pool,
            log_path=log_path,
        )

        self.task_info = task_info
        self.classes_info = classes_info
        self.modality_info = modality_info
        self.sample = sample
        self.examples = examples
        self.init_system_message()

    def init_system_message(self):
        modality_info = json.dumps(self.modality_info, indent=2)
        content = (
            f"You are {self.name} agent that solves a sensing task.\n"
            "You have the following information about the task:\n"
            f"{self.task_info}\n\n"
            f"You will receive sensor features from {self.name} modality, "
            "and you have the following information about the modality:\n"
            f"{modality_info}\n\n"
            "Your goal is to analyze the features and "
            "provide a reasoned answer using your knowledge."
        )
        self.set_system_message(content)

    def gen_feature_info(self):
        feature_info = f"{self.name} features:\n"
        if len(self.examples) > 0:
            feature_info += f"{self.gen_example_info()}\n\n"
        feature_info += "**Current sample features**:\n"
        for k, v in self.sample["features"].items():
            if self.name in k:
                feature_info += f"  - {k}: {self.format_feature(v)}\n"
        feature_info = feature_info.strip()
        return feature_info

    def gen_example_info(self):
        example_info = (
            "**Examples**\n"
            "Sensor values might not always align with your inherent "
            "knowledge due to differences in data collection or processing. "
            "So, we included a few labeled examples to help your interpretation:\n"
        )
        for example in self.examples:
            example_info += f"*Example of {example['label']}*:\n"
            for k, v in example["features"].items():
                if self.name in k:
                    example_info += f"  - {k}: {self.format_feature(v)}\n"
            example_info += "\n"
        example_info = example_info.strip()
        return example_info

    def format_feature(self, value):
        if isinstance(value, float):
            if abs(value) >= 1e4 or abs(value) < 1e-2:
                return f"{value:.2e}"
            return f"{value:.2f}"
        return value

    async def interpret(self):
        feature_info = self.gen_feature_info()
        content = (
            f"You have received sensor features from {self.name} modality:\n"
            f"{feature_info}\n\n"
            f"Please provide your answer for the task among {self.classes_info} "
            "and the reasoning for your answer.\n"
            "Note that the sensor features might be wrong due to the data collection or processing.\n"
            "You can evaluate the quality of the features by checking the examples you have.\n"
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

    async def debate(self, responses):
        responses = {k: v for k, v in responses.items() if k != self.name}
        content = (
            f"Other agents have provided their answers for the same task:\n"
            f"{json.dumps(responses, indent=2)}\n\n"
            f"Please debate with them and provide your answer for the task among {self.classes_info} "
            "and the reasoning for your answer.\n"
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
