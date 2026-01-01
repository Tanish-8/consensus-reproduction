import os
import sys
import yaml
import numpy as np
import asyncio

from fire import Fire

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.data_loader import DataLoader
from core.model import load_models
from core.agent_pool import AgentPool
from core.modality_agent import ModalityAgent
from core.semantic_fusion_agent import SemanticFusionAgent
from core.statistical_fusion_agent import StatisticalFusionAgent
from core.hybrid_fusion_agent import HybridFusionAgent


async def run_sample(
    idx,
    test_sample,
    examples,
    modalities,
    task_info,
    classes_info,
    modality_info_dict,
    model_pool,
    config,
):
    ground_truth = test_sample["label"]
    sample_log_path = os.path.join(config["log_path"], f"{idx}_{ground_truth}")
    agent_pool = AgentPool(sample_log_path)

    for modality in modalities:
        modality_info = modality_info_dict[modality]
        agent = ModalityAgent(
            name=modality,
            model_pool=model_pool,
            task_info=task_info,
            classes_info=classes_info,
            modality_info=modality_info,
            sample=test_sample,
            examples=examples,
            log_path=sample_log_path,
        )
        agent_pool.add_modality_agent(agent)

    semantic_fusion_agent = SemanticFusionAgent(
        name="SemanticFusionAgent",
        model_pool=model_pool,
        task_info=task_info,
        classes_info=classes_info,
        log_path=sample_log_path,
    )
    agent_pool.add_semantic_fusion_agent(semantic_fusion_agent)

    statistical_fusion_agent = StatisticalFusionAgent(
        name="StatisticalFusionAgent",
        model_pool=model_pool,
        task_info=task_info,
        classes_info=classes_info,
        log_path=sample_log_path,
    )
    agent_pool.add_statistical_fusion_agent(statistical_fusion_agent)

    hybrid_fusion_agent = HybridFusionAgent(
        name="HybridFusionAgent",
        model_pool=model_pool,
        task_info=task_info,
        classes_info=classes_info,
        log_path=sample_log_path,
    )
    agent_pool.add_hybrid_fusion_agent(hybrid_fusion_agent)

    print(f"Running interpretation for sample {idx}")
    result = await agent_pool.run_parallel_interpretation()
    if result is None:
        return None
    agent_pool.log_summary(
        f"[Interpretation] Decision made: {result} (ground truth: {test_sample['label']})"
    )
    print(f"sample {idx} interpretation finished")

    print(f"Running semantic fusion for sample {idx}")
    result = await agent_pool.run_semantic_fusion()
    if result is None:
        return
    agent_pool.log_summary(
        f"[Semantic Fusion] Decision made: {result} (ground truth: {test_sample['label']})"
    )
    print(f"sample {idx} semantic fusion finished")

    print(f"Running statistical fusion for sample {idx}")
    result = await agent_pool.run_statistical_fusion()
    if result is None:
        return
    agent_pool.log_summary(
        f"[Statistical Fusion] Decision made: {result} (ground truth: {test_sample['label']})"
    )
    print(f"sample {idx} statistical fusion finished")

    print(f"Running hybrid fusion for sample {idx}")
    result = await agent_pool.run_hybrid_fusion()
    if result is None:
        return
    agent_pool.log_summary(
        f"[Hybrid Fusion] Decision made: {result} (ground truth: {test_sample['label']})"
    )
    print(f"sample {idx} hybrid fusion finished")
    print(f"sample {idx} finished")
    return


async def run_parallel_samples(config, model_pool):
    dataloader = DataLoader(config["data_path"])
    modalities = dataloader.get_modalities()
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
                modalities,
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
    model_pool = load_models(config["models"])
    print("Model pool initialized.")
    asyncio.run(run_parallel_samples(config, model_pool))


if __name__ == "__main__":
    Fire(run)
