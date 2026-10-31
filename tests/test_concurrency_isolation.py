import os
import json
import asyncio
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from run_consensus import run_sample


class MockConcurrentModelPool:
    """
    Mock model pool that simulates asynchronous LLM responses with intentional
    interleaving delays to verify concurrency isolation across multiple samples.
    """

    def __init__(self):
        self.invocation_count = 0
        self.lock = asyncio.Lock()

    async def invoke(self, messages):
        async with self.lock:
            self.invocation_count += 1

        # Force cooperative task switching between concurrent samples
        await asyncio.sleep(0.01)

        # Inspect all message contents in the conversation history
        all_text = " ".join(str(m.content) for m in messages if hasattr(m, "content"))
        content = all_text
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        last_human = human_msgs[-1].content if human_msgs else ""

        # Modality Agent responses
        if "0.12" in last_human:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 0 ECG indicates calm baseline",
                    "ANSWER": "baseline",
                })
            )
        if "0.34" in last_human:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 0 EDA indicates baseline tonic level",
                    "ANSWER": "baseline",
                })
            )
        if "8.91" in last_human:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 1 ECG shows tachycardia indicating stress",
                    "ANSWER": "stress",
                })
            )
        if "9.45" in last_human:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 1 EDA shows high SCR response indicating stress",
                    "ANSWER": "stress",
                })
            )

        # Statistical Fusion Agent responses (distinguished by majority vote clause)
        if "The majority vote among agents is: baseline" in all_text:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 0 statistical fusion agrees on baseline",
                    "ANSWER": "baseline",
                })
            )
        if "The majority vote among agents is: stress" in all_text:
            return AIMessage(
                content=json.dumps({
                    "REASON": "Sample 1 statistical fusion agrees on stress",
                    "ANSWER": "stress",
                })
            )

        # Semantic Fusion Agent responses
        if "You have received responses from multiple sensing agents:" in all_text:
            if "tachycardia" in all_text:
                # Deliberate persistent failure in Sample 1 to verify failure isolation
                return AIMessage(content="Simulated Semantic Failure: invalid JSON")
            if "calm baseline" in all_text:
                return AIMessage(
                    content=json.dumps({
                        "REASON": "Sample 0 semantic fusion confirms baseline state",
                        "ANSWER": "baseline",
                    })
                )

        # Hybrid Fusion Agent responses
        if "Responses from fusion agents:" in content:
            if "Sample 0" in content or "baseline" in content:
                return AIMessage(
                    content=json.dumps({
                        "REASON": "Sample 0 hybrid fusion decides baseline",
                        "ANSWER": "baseline",
                    })
                )

        return AIMessage(content=json.dumps({"REASON": "Default fallback", "ANSWER": "baseline"}))


def test_concurrent_sample_isolation(tmp_path):
    """
    Deterministic unit test proving concurrent sample isolation:
    1. Sample 0 (ground truth: baseline) and Sample 1 (ground truth: stress) execute concurrently.
    2. Modality inputs, agent outputs, and fusion responses do not cross-contaminate.
    3. Sample 1 semantic fusion failure does not affect Sample 0's successful completion.
    4. Logs and summaries remain strictly partitioned by sample ID and ground truth folder.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    modalities = ["chest_ECG", "chest_EDA"]
    task_info = "Classify emotional state"
    classes_info = ["amusement", "baseline", "stress"]
    modality_info_dict = {
        "chest_ECG": {
            "Data collection": "Chest ECG at 700 Hz",
            "Feature extraction": "Heart rate variability",
        },
        "chest_EDA": {
            "Data collection": "Chest EDA at 700 Hz",
            "Feature extraction": "Skin conductance response",
        },
    }

    sample_0 = {
        "label": "baseline",
        "features": {
            "chest_ECG_mean": 0.12,
            "chest_EDA_mean": 0.34,
        },
    }

    sample_1 = {
        "label": "stress",
        "features": {
            "chest_ECG_mean": 8.91,
            "chest_EDA_mean": 9.45,
        },
    }

    config = {"log_path": str(log_dir)}
    model_pool = MockConcurrentModelPool()

    async def run_concurrent():
        task0 = asyncio.create_task(
            run_sample(
                0,
                sample_0,
                examples=[],
                modalities=modalities,
                task_info=task_info,
                classes_info=classes_info,
                modality_info_dict=modality_info_dict,
                model_pool=model_pool,
                config=config,
            )
        )
        task1 = asyncio.create_task(
            run_sample(
                1,
                sample_1,
                examples=[],
                modalities=modalities,
                task_info=task_info,
                classes_info=classes_info,
                modality_info_dict=modality_info_dict,
                model_pool=model_pool,
                config=config,
            )
        )
        await asyncio.gather(task0, task1)

    asyncio.run(run_concurrent())

    # --- Verification 1: Distinct Log Folders ---
    sample_0_dir = log_dir / "0_baseline"
    sample_1_dir = log_dir / "1_stress"

    assert sample_0_dir.exists(), "Sample 0 folder was not created"
    assert sample_1_dir.exists(), "Sample 1 folder was not created"

    # --- Verification 2: Sample 0 Outcomes & Isolation ---
    s0_summary = (sample_0_dir / "summary.txt").read_text(encoding="utf-8")
    assert "[Interpretation] Decision made: baseline (ground truth: baseline)" in s0_summary
    assert "[Semantic Fusion] Decision made: baseline (ground truth: baseline)" in s0_summary
    assert "[Statistical Fusion] Decision made: baseline (ground truth: baseline)" in s0_summary
    assert "[Hybrid Fusion] Decision made: baseline (ground truth: baseline)" in s0_summary
    assert "stress" not in s0_summary, "Sample 1 state leaked into Sample 0 summary"

    s0_log = (sample_0_dir / "log.txt").read_text(encoding="utf-8")
    assert "Sample 0" in s0_log
    assert "Sample 1" not in s0_log, "Sample 1 agent log leaked into Sample 0 root log"
    assert "tachycardia" not in s0_log

    # --- Verification 3: Sample 1 Outcomes & Isolation ---
    s1_summary = (sample_1_dir / "summary.txt").read_text(encoding="utf-8")
    assert "[Interpretation] Decision made: stress (ground truth: stress)" in s1_summary
    assert "[Semantic Fusion] Execution failed (ground truth: stress)" in s1_summary
    assert "[Statistical Fusion] Decision made: stress (ground truth: stress)" in s1_summary
    assert "[Hybrid Fusion] Skipped (dependency failed) (ground truth: stress)" in s1_summary
    assert "baseline" not in s1_summary, "Sample 0 state leaked into Sample 1 summary"

    s1_log = (sample_1_dir / "log.txt").read_text(encoding="utf-8")
    assert "Sample 1" in s1_log
    assert "Sample 0" not in s1_log, "Sample 0 agent log leaked into Sample 1 root log"
    assert "calm baseline" not in s1_log

    # --- Verification 4: Failure Isolation ---
    # Sample 1's Semantic Fusion failed, but Sample 0's Semantic and Hybrid Fusion succeeded completely!
    assert "[Semantic Fusion] Decision made: baseline" in s0_summary
    assert "[Hybrid Fusion] Decision made: baseline" in s0_summary
