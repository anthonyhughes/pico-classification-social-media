from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorWithPadding

from annotations import ID_TO_LABEL, LABEL_TO_ID
from constants import MODEL_NAME
from metrics import compute_metrics
from span_model import RobertaForSpanCategorization
from token_utils import tokenize_and_adjust_labels_w_tokenizer


def model_init():
    # For reproducibility
    return RobertaForSpanCategorization.from_pretrained("roberta-base", id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)


def run_train():
    print("Starting span classification!")

    print("Loading training arguments...")
    training_args = TrainingArguments(
        output_dir="./models/fine_tune_bert_output_span_cat",
        evaluation_strategy="epoch",
        learning_rate=2.5e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        log_level='critical',
        seed=12345
    )

    print("Loading datasets...")
    train_ds = Dataset.from_json("social_media_medical_claim_corpus/st1/st1_train_inc_text.jsonl")
    validation_ds = Dataset.from_json("social_media_medical_claim_corpus/st1/st1_val_inc_text.jsonl")

    print("Tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train_ds = train_ds.map(
        tokenize_and_adjust_labels_w_tokenizer(tokenizer),
        remove_columns=train_ds.column_names)
    tokenized_validation_ds = validation_ds.map(
        tokenize_and_adjust_labels_w_tokenizer(tokenizer),
        remove_columns=validation_ds.column_names)

    # https://huggingface.co/transformers/v4.8.1/main_classes/data_collator.html
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    print("Initializing trainer...")
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_validation_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()


if __name__ == '__main__':
    print("Starting!")
    run_train()
