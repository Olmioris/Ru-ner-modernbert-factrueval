from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def load_mlm_model(model_id):
    return AutoModelForMaskedLM.from_pretrained(model_id)

def create_mlm_trainer(model, tokenizer, train_ds, output_dir="./mlm_pretraining"):
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator
    )

    return trainer
