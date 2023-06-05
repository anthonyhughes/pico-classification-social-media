import json
from typing import Dict, List

import pandas as pd

from annotations import label_lookup


# {"tags":[{"end":32,"start":19,"tag":"PER"},{"end":32,"start":6,"tag":"NCHUNK"},{"end":152,"start":143,"tag":"NCHUNK"},{"end":225,"start":211,"tag":"NCHUNK"}
# {"end":79,"start":45,"tag":"NCHUNK"}],"id":"train_253","text":"Selon l'ethnologue Maurice Duval, \u00ab dire que ce mouvement de la gauche radicale est \u00ab une secte \u00bb, ce n'est pas argumenter l\u00e9gitimement contre ses id\u00e9es, mais c'est sugg\u00e9rer qu'il est malfaisant, malsain et que sa disparition serait souhaitable \u00bb."}
# {"tags":[{"end":167,"start":155,"tag":"PER"},{"end":190,"start":169,"tag":"PER"},{"end":206,"start":192,"tag":"PER"},{"end":227,"start":218,"tag":"PER"}


# read social_media_medical_claim_corpus/st1/st1_train_inc_text_.csv and convert each line to json format


def convert_crowd_annotations_to_new_format(original: List) -> List:
    """
    Convert the original format of the crowd annotations to the new format.
    :param original:
    :return:
    """
    # [{""crowd-entity-annotation"":{""entities"":[{""endOffset"":858,""label"":""per_exp"",""startOffset"":661},{""endOffset"":2213,""label"":""per_exp"",""startOffset"":1861},{""endOffset"":2407,""label"":""per_exp"",""startOffset"":2255},{""endOffset"":3254,""label"":""claim_per_exp"",""startOffset"":2697},{""endOffset"":3620,""label"":""claim_per_exp"",""startOffset"":3294},{""endOffset"":3751,""label"":""claim_per_exp"",""startOffset"":3621},{""endOffset"":4480,""label"":""per_exp"",""startOffset"":3752},{""endOffset"":4759,""label"":""question"",""startOffset"":4482}]}}]
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


def run(set: str) -> None:
    """
    Run the conversion.
    :param set:
    :return:
    """
    print(f"Starting {set} corpus conversion for datasets!")
    df = pd.read_csv(f"social_media_medical_claim_corpus/st1/st1_{set}_inc_text_.csv")
    all_lines = []
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
        all_lines.append(new_tag_set)

    # write all_lines to a jsonlines file
    with open(f"social_media_medical_claim_corpus/st1/st1_{set}_inc_text.jsonl", "w") as f:
        for line in all_lines:
            f.write(json.dumps(line))
            f.write("\n")

    print("Done!")


if __name__ == '__main__':
    print("Starting!")
    run(set='train')
    run(set='test')
