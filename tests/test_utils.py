import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils
import numpy as np

data = np.load('data/mini_gm_public_v0.1.p', allow_pickle=True)

def test_extract_Xy():
    X, y = utils.extract_Xy(data)
    
    assert len(X) == len(y)
    assert X.shape[1] == 320

def test_count_levels_distinct_ids():
    n_classes, n_subjects, n_images = utils.count_levels_distinct_ids(data)
    
    assert n_classes > 0
    assert n_subjects > 0
    assert n_images > 0
    assert n_classes < n_subjects < n_images