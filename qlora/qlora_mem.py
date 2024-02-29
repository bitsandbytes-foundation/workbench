from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
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


class StreamLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


@dataclass
class RunSettings:
    run_name: Optional[str]
    run_dir: Optional[Path]
    quantized: bool
    lora: bool
    non_reentrant: bool
    full_profile: bool
    gradient_accumulation_steps: int
    model_id: str

    def __post_init__(self):
        self.run_name = self.run_name or get_run_name(self)
        self.run_dir = self.run_dir or create_run_directory(self.run_name)


@dataclass
class TrainingConfigs:
    quant: Optional[BitsAndBytesConfig]
    lora: Optional[LoraConfig]
    train: TrainingArguments


def configure_logging(settings: RunSettings):
    log_file = settings.run_dir / f"{settings.run_name}-{datetime.now()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=log_file.as_posix(), mode="w"),
        ],
    )
    sys.stdout = StreamLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = StreamLogger(logging.getLogger("stderr"), logging.ERROR)
    logging.info(f"Logging configured to use {log_file}")


def create_run_directory(run_name: str) -> Path:
    base_path = run_dir = Path.cwd() / run_name
    counter = 1

    while run_dir.exists():
        run_dir = Path(f"{base_path}_{counter}")
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_cfgs(settings: RunSettings) -> TrainingConfigs:
    logging.info("Configuring training settings")
    cfgs = TrainingConfigs(
        quant=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_quant_type="nf4"
        )
        if settings.quantized
        else None,
        lora=LoraConfig(
            r=8,
            target_modules=["q_proj", "v_proj", "k_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        if settings.lora
        else None,
        train=TrainingArguments(
            output_dir=settings.run_dir / "results",
            save_strategy="no",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_dir="./logs",
            logging_steps=10,
            bf16=True,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            gradient_checkpointing_kwargs={"use_reentrant": settings.non_reentrant},
        ),
    )
    logging.info(f"Training settings (short): {settings!r}")
    logging.info(f"Training settings (long): {cfgs!r}")
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
    # we need to clear the cache after quantization and lora adaption etc, otherwise you get spurious results
    torch.cuda.empty_cache()

    if full_profile:
        logging.info("Stopping CUDA memory profiling.")
        torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(full_profile: bool, settings: RunSettings):
    if full_profile:
        logging.info("Exporting CUDA memory profiling data.")
        torch.cuda.memory._dump_snapshot(settings.run_dir / "memory_snapshot.pickle")


def get_run_name(settings: RunSettings) -> str:
    return (
        f"run--{'q' if settings.quantized else 'nq'}-{'lora' if settings.lora else 'vanillla'}-"
        f"{'non_reentrant' if settings.non_reentrant else 'reentrant'}-"
        f"grad_acc_steps_{settings.gradient_accumulation_steps}-"
        f"{settings.model_id.split('/')[-1]}"
    )


def main(
    run_name: Optional[str] = None,
    quantized: bool = False,
    lora: bool = False,
    non_reentrant: bool = False,
    full_profile: bool = True,
    gradient_accumulation_steps: int = 8,
    model_id: str = "facebook/opt-350m",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Aborting...")

    run_settings = RunSettings(
        run_name,
        None,
        quantized,
        lora,
        non_reentrant,
        full_profile,
        gradient_accumulation_steps,
        model_id,
    )

    configure_logging(run_settings)
    cfgs = get_cfgs(run_settings)

    logging.info(f"Starting training run: {run_name}")

    dataset = load_dataset("imdb", split="train[:1%]")
    logging.info(f"Loaded dataset: {dataset}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=cfgs.quant)
    trainer = get_trainer(model, tokenizer, cfgs, dataset)

    start_memory_recording(full_profile)
    trainer.train()
    export_memory_snapshot(full_profile, run_settings)
    stop_memory_recording(full_profile)

    max_mem_in_bytes = torch.cuda.max_memory_allocated()
    logging.info(f"Peak memory usage: {human_readable_size(max_mem_in_bytes)}")
