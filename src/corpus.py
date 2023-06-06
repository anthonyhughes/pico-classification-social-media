import json
from typing import Dict, List

import pandas as pd

from annotations import label_lookup


def write_jsonl_to_file(all_lines: List, set: str) -> None:
    # write all_lines to a jsonlines file
    with open(f"social_media_medical_claim_corpus/st1/st1_{set}_inc_text.jsonl", "w") as f:
        for line in all_lines:
            f.write(json.dumps(line))
            f.write("\n")


def convert_crowd_annotations_to_new_format(original: List) -> List:
    """
    Convert the original format of the crowd annotations to the new format.
    :param original:
    :return:
    """
    tags = original[0]['crowd-entity-annotation']['entities']
    for tag in tags:
        tag['tag'] = label_lookup(tag['label'])
        del tag['label']
        tag['start'] = tag['startOffset']
        del tag['startOffset']
        tag['end'] = tag['endOffset']
        del tag['endOffset']
    print(tags)
    return [tag for tag in tags]


def run(set: str, with_val_split: bool) -> None:
    """
    Run the conversion.
    :param set:
    :return:
    """
    print(f"Starting {set} corpus conversion for datasets!")
    df = pd.read_csv(f"social_media_medical_claim_corpus/st1/st1_{set}_inc_text_.csv")
    all_lines = []
    # iterate over all examples
    for row in df.itertuples():
        new_tag_set = {
            "id": row.post_id,
            "text": row.text,
        }
        if set == 'train':
            annotated_labels = row.stage1_labels
            parsed_labels = json.loads(annotated_labels)
            new_tag_set["tags"] = convert_crowd_annotations_to_new_format(parsed_labels)
        elif set == 'test':
            new_tag_set["tags"] = []

        if len(new_tag_set["tags"]) > 0:
            all_lines.append(new_tag_set)

    if set == 'train' and with_val_split:
        train_length = round(len(all_lines) - len(all_lines) * 0.2)
        write_jsonl_to_file(all_lines[:train_length], set='train')
        write_jsonl_to_file(all_lines[train_length:], set='val')
    else:
        write_jsonl_to_file(all_lines, set='test')
    print("Done!")


if __name__ == '__main__':
    print("Starting!")
    run(set='train', with_val_split=True)
    run(set='test', with_val_split=False)
