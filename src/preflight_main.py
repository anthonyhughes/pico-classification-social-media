"""
File for testing various parts of the code prior to running the main script.
"""
from datasets import Dataset
from transformers import AutoTokenizer

import annotations
from constants import MODEL_NAME
from token_utils import tokenize_and_adjust_labels_w_tokenizer, view_bio_corpus_sample


def annotation_viewer(train_ds: Dataset) -> None:
    """
    Prints out a few examples of annotations from the dataset.
    :param train_ds:
    :return:
    """
    print("Example annotations:")
    for i in range(0):
        example = train_ds[i]
        print(f"\n{example['text']}")
        for tag_item in example["tags"]:
            print(tag_item["tag"].ljust(10), "-", example["text"][tag_item["start"]: tag_item["end"]])


def run():
    """
    Runs the preflight tests.
    :return:
    """
    print("Starting span classification!")
    print(annotations.ID_TO_TAG)
    print(annotations.ID_TO_LABEL)
    train_ds = Dataset.from_json("social_media_medical_claim_corpus/st1/st1_train_inc_text.jsonl")
    # print(train_ds[0])
    print(annotation_viewer(train_ds))
    distilbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = distilbert_tokenizer("Hello world!", return_offsets_mapping=True)
    print(tokens.offset_mapping)

    tokenized_train_ds = train_ds.map(
        tokenize_and_adjust_labels_w_tokenizer(distilbert_tokenizer),
        remove_columns=train_ds.column_names)
    print(tokenized_train_ds[0])
    view_bio_corpus_sample(tokenizer=distilbert_tokenizer, sample=tokenized_train_ds[2])


if __name__ == '__main__':
    print("Starting!")
    run()
