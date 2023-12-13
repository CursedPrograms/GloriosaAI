import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
config = GPT2Config.from_json_file('config.json')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='path_to_train_dataset.txt',
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
model = GPT2LMHeadModel(config)
training_args = TrainingArguments(
    output_dir='./mini_gpt_model',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)
trainer.train()