from collections import Counter
import math



def get_IDF_weights(dataset, data_size = None, note_types = ['Top', 'Middle', 'Base']):
    all_notes = []

    if data_size is None:
        data_size = len(dataset)

    for note_type in note_types:
        notes = dataset[note_type].str.split(',').explode().str.strip().tolist()
        all_notes.extend(notes)

    note_counts = Counter(all_notes) # type: dict

    IDF_weight = {
        note : math.log(data_size / cnt) + 1
        for note, cnt in note_counts.items()
    }
    return IDF_weight