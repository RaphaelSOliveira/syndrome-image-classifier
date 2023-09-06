# data manipulation
import numpy as np

# Extract embeddings and labels
def extract_Xy(data:dict) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for syndrome_id, syndrome_data in data.items():
        for _, subject_data in syndrome_data.items():
            for _, encoding in subject_data.items():
                X.append(encoding)
                y.append(syndrome_id)

    return np.array(X), np.array(y)

# Count distinct ids for every level of the data dictionary (syndrome_id, subject_id, image_id)
def count_levels_distinct_ids(data:dict) -> tuple[int, int, int]:
    n_classes = len(data.keys())

    n_images = 0
    n_subjects = 0
    for syndrome_id in data.keys():
        for subject_id in data[syndrome_id].keys():
            n_subjects += 1
            for _ in data[syndrome_id][subject_id].keys():
                n_images += 1
    
    return n_classes, n_subjects, n_images