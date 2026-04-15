import os
import sys
import yaml
import numpy as np
import asyncio

from fire import Fire

current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(baseline_dir)

for path in (project_root, baseline_dir):
    if path and path not in sys.path:
        sys.path.insert(0, path)

from core.data_loader import DataLoader
from core.model import load_models
from baselines.single_agent.agent_pool import AgentPool
from baselines.single_agent.multimodal_agent import MultimodalAgent


async def run_sample(
    idx,
    test_sample,
    examples,
    task_info,
    classes_info,
    modality_info_dict,
    model_pool,
    config,
):
    ground_truth = test_sample["label"]
    sample_log_path = os.path.join(config["log_path"], f"{idx}_{ground_truth}")
    agent_pool = AgentPool(sample_log_path)

    agent = MultimodalAgent(
        model_pool=model_pool,
        task_info=task_info,
        classes_info=classes_info,
        modality_info=modality_info_dict,
        sample=test_sample,
        examples=examples,
        log_path=sample_log_path,
    )
    agent_pool.add_agent(agent)

    print(f"Running interpretation for sample {idx}")
    result = await agent_pool.run_parallel_interpretation()
    if result is None:
        return None
    agent_pool.log_summary(
        f"[Interpretation] Decision made: {result} (ground truth: {test_sample['label']})"
    )
    print(f"sample {idx} interpretation finished")
    return


async def run_parallel_samples(config, model_pool):
    dataloader = DataLoader(config["data_path"])
    task_info = dataloader.get_task_info()
    classes_info = dataloader.get_classes_info()
    modality_info_dict = dataloader.get_modality_info_dict()
    tasks = []
    for idx in range(len(dataloader)):
        test_sample, examples = dataloader[idx]
        task = asyncio.create_task(
            run_sample(
                idx,
                test_sample,
                examples,
                task_info,
                classes_info,
                modality_info_dict,
                model_pool,
                config,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)
    return


def run(config_path):
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    np.random.seed(config["seed"])
    if config["use_large_context"]:
        num_ctx = 30000
    else:
        num_ctx = 12000
    model_pool = load_models(config["models"], top_p=config.get("top_p", 1.0), num_ctx=num_ctx)
    print("Model pool initialized.")
    asyncio.run(run_parallel_samples(config, model_pool))


if __name__ == "__main__":
    Fire(run)
