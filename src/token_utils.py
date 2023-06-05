from typing import Dict, Callable

from annotations import LABEL_TO_ID, ID_TO_LABEL
from constants import MAX_LENGTH


def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int) -> str:
    """
    Check if the token is inside a span.
    :param token_start:
    :param token_end:
    :param span_start:
    :param span_end:
    :return:
      - "B" if beginning
      - "I" if inner
      - "O" if outer
      - "N" if not valid token (like <SEP>, <CLS>, <UNK>)
    """
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"


def tokenize_and_adjust_labels_w_tokenizer(tokenizer) -> Callable:
    """
    Tokenize the text and adjust the labels accordingly.
    :param tokenizer:
    :return:
    """
    def tokenize_and_adjust_labels(sample: Dict):
        """
        Args:
            - sample (dict): {"id": "...", "text": "...", "tags": [{"start": ..., "end": ..., "tag": ...}, ...]
        Returns:
            - The tokenized version of `sample` and the labels of each token.
        """
        # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
        # Use max_length and truncation to adjust the text length
        tokenized = tokenizer(sample["text"],
                              return_offsets_mapping=True,
                              padding="max_length",
                              max_length=MAX_LENGTH,
                              truncation=True)

        # Multilabel classification task for each token
        labels = [[0 for _ in LABEL_TO_ID.keys()] for _ in range(MAX_LENGTH)]

        # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
        # or inside the spans
        for (token_start, token_end), token_labels in zip(tokenized["offset_mapping"], labels):
            for span in sample["tags"]:
                role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
                if role == "B":
                    token_labels[LABEL_TO_ID[f"B-{span['tag']}"]] = 1
                elif role == "I":
                    token_labels[LABEL_TO_ID[f"I-{span['tag']}"]] = 1

        return {**tokenized, "labels": labels}

    return tokenize_and_adjust_labels


def view_bio_corpus_sample(tokenizer, sample) -> None:
    """
    Print the tokenized text and the corresponding labels.
    :param tokenizer:
    :param sample:
    :return:
    """
    print("--------Token---------|--------Labels----------")
    for token_id, token_labels in zip(sample["input_ids"], sample["labels"]):
        # Decode the token_id into text
        token_text = tokenizer.decode(token_id)

        # Retrieve all the indices corresponding to the "1" at each token, decode them to label name
        labels = [ID_TO_LABEL[label_index] for label_index, value in enumerate(token_labels) if value == 1]

        # Decode those indices into label name
        print(f" {token_text:20} | {labels}")

        # Finish when we meet the end of sentence.
        if token_text == "</s>":
            break
