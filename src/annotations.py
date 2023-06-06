TAG_TO_ID = {'QUESTION': 1, 'CLAIM': 2}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}
LABEL_TO_ID = {
    'O': 0,
    **{f'B-{k}': 2 * v - 1 for k, v in TAG_TO_ID.items()},
    **{f'I-{k}': 2 * v for k, v in TAG_TO_ID.items()}
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
N_LABELS = len(ID_TO_LABEL)


def label_lookup(param):
    """
    Convert a label to a corresponding ID
    Use to consolidate all claims classes into one
    :param param:
    :return:
    """
    if param == 'question':
        return 'QUESTION'
    else:
        return 'CLAIM'
