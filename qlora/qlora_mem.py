from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def configure_logging(log_dir: Path, run_name: str):
    log_file = log_dir / f"{run_name}-{datetime.now()}.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),  # For stdout logging
            logging.FileHandler(filename=log_file.as_posix(), mode="w"),  # For file logging
        ],
        format="%(message)s",
    )
    logging.info(f"Logging configured to use {log_file}")


def create_run_directory(run_name: str) -> Path:
    base_path = run_dir = Path.cwd() / run_name
    counter = 1

    while run_dir.exists():
        run_dir = Path(f"{base_path}_{counter}")
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class TrainingConfigs:
    quant: Optional[BitsAndBytesConfig]
    lora: Optional[LoraConfig]
    train: TrainingArguments


def get_cfgs(quantized: bool, lora: bool, non_reentrant: bool, run_dir: Path) -> TrainingConfigs:
    logging.info("Configuring training settings")
    cfgs = TrainingConfigs(
        quant=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_quant_type="nf4"
        )
        if quantized
        else None,
        lora=LoraConfig(
            r=8,
            target_modules=["q_proj", "v_proj", "k_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        if lora
        else None,
        train=TrainingArguments(
            output_dir=run_dir / "results",
            save_strategy="no",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_dir="./logs",
            logging_steps=10,
            bf16=True,
            gradient_checkpointing_kwargs={"use_reentrant": non_reentrant},
        ),
    )
    logging.info(f"Training settings: {cfgs}")
    return cfgs


def get_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfgs: TrainingConfigs,
    dataset,
) -> SFTTrainer:
    logging.info("Initializing the trainer")
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=cfgs.train,
        peft_config=cfgs.lora,
        train_dataset=dataset,
        dataset_text_field="text",
    )


def human_readable_size(size, decimal_places=2):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def start_memory_recording(full_profile: bool, max_entries: int = 100000):
    if full_profile:
        logging.info("Starting CUDA memory profiling.")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=max_entries)


def stop_memory_recording(full_profile: bool):
    if full_profile:
        logging.info("Stopping CUDA memory profiling.")
        torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(full_profile: bool, run_dir: Path, file_prefix: str):
    if full_profile:
        logging.info("Exporting CUDA memory profiling data.")
        torch.cuda.memory._dump_snapshot(run_dir / "memory_snapshot.pickle")


def main(
    run_name: Optional[str] = None,
    quantized: bool = False,
    lora: bool = False,
    non_reentrant: bool = False,
    full_profile: bool = True,
    model_id: str = "facebook/opt-350m",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Aborting...")

    run_name = (
        run_name
        or f"run--{'q' if quantized else 'nq'}-{'lora' if lora else ''}-"
        f"{'non_reentrant' if non_reentrant else 'reentrant'}-"
        f"{model_id.split('/')[-1]}"
    )
    run_dir = create_run_directory(run_name)
    configure_logging(log_dir=run_dir, run_name=run_name)
    cfgs = get_cfgs(quantized, lora, non_reentrant, run_dir)

    logging.info(f"Starting training run: {run_name}")

    dataset = load_dataset("imdb", split="train[:1%]")
    logging.info(f"Loaded dataset: {dataset}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=cfgs.quant)
    trainer = get_trainer(model, tokenizer, cfgs, dataset)

    start_memory_recording(full_profile)
    trainer.train()
    export_memory_snapshot(full_profile, run_dir, run_name)
    stop_memory_recording(full_profile)

    max_mem = torch.cuda.max_memory_allocated()  # Get max memory allocated in bytes
    human_readable_max_mem = human_readable_size(max_mem)
    logging.info(f"Peak memory usage: {human_readable_max_mem}")
