import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer


set_seed(42)

if not torch.cuda.is_available():
    raise RuntimeError("Run this script in a Slurm GPU job.")

print("GPU:", torch.cuda.get_device_name(0), flush=True)

tokenizer = AutoTokenizer.from_pretrained("base_model")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = Qwen3_5ForCausalLM.from_pretrained(
    "base_model",
    dtype=torch.bfloat16,
).to("cuda")

model.config.use_cache = False


def make_prompt(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"

    if example["input"].strip():
        text += f"### Input:\n{example['input']}\n\n"

    return text + "### Response:\n"


def format_example(example):
    return {
        "text": make_prompt(example)
        + example["output"]
        + tokenizer.eos_token
    }


dataset = load_dataset(
    "json",
    data_files="data/alpaca_data_cleaned.json",
    split="train",
)

split = dataset.train_test_split(test_size=200, seed=42)
examples = list(split["test"].select(range(3)))

train_data = split["train"].map(
    format_example,
    remove_columns=dataset.column_names,
)

eval_data = split["test"].map(
    format_example,
    remove_columns=dataset.column_names,
)

print("Training examples:", len(train_data), flush=True)
print("Held-out examples:", len(eval_data), flush=True)


def show_predictions(title):
    model.eval()
    print(f"\n{title}", flush=True)

    for example in examples:
        inputs = tokenizer(
            make_prompt(example),
            return_tensors="pt",
            truncation=True,
            max_length=384,
        ).to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print("\nInstruction:", example["instruction"], flush=True)
        print("Input:", example["input"], flush=True)
        print("Reference:", example["output"], flush=True)
        print("Generated:", answer, flush=True)


show_predictions("BEFORE FINE-TUNING")

model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

model.print_trainable_parameters()

args = SFTConfig(
    output_dir="results",
    dataset_text_field="text",
    max_length=512,
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=10,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
)

before = trainer.evaluate()
print("Before evaluation loss:", before["eval_loss"], flush=True)
trainer.save_metrics("before", before)

result = trainer.train()
trainer.save_metrics("train", result.metrics)

after = trainer.evaluate()
print("After evaluation loss:", after["eval_loss"], flush=True)
trainer.save_metrics("after", after)

trainer.model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

model.gradient_checkpointing_disable()
show_predictions("AFTER FINE-TUNING")

print("\nFinished. Adapter saved in lora_adapter.", flush=True)
