import transformers
from datasets import load_dataset
raw_dataset=load_dataset("raw_dataset")
import accelerate
print("Acclerate version is: \n")
print(accelerate.__version__)
from transformers import AutoTokenizer
model_checkpoint="t5-small"
from Addition import AdditionDataset 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
import os

output_dir = "/results/checkpoint-500"

try:
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
except OSError as e:
    print(f"Error creating directory: {e}")
sample=tokenizer(raw_dataset["test"][5]["article"])
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""
max_input_length = 1024
max_target_length = 128

def preprocess_function(raw_dataset):
    inputs = [prefix + arc for arc in raw_dataset["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=raw_dataset["highlights"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
preprocess_function(raw_dataset['train'][:2])
tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args =Seq2SeqTrainingArguments (
    output_dir='results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=True,
    fp16_full_eval=False,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=AdditionDataset(2,"train"),
    eval_dataset=AdditionDataset(2,"test"),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

new_model=trainer.train()
print(new_model)
eval=trainer.evaluate()