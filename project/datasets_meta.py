"""Dataset registry shared by augmentation.py, train_msf_scratch.py and
run_experiment.py — the single place that knows what each MedMNIST dataset
looks like. pneumoniamnist entries reproduce the previously hard-coded values
exactly, so binary behaviour is unchanged.
"""

DATASETS = {
    "pneumoniamnist": {
        "medmnist_class": "PneumoniaMNIST",
        "n_classes": 2,
        "channels": 1,
        "splits": (4708, 524, 624),
        "class_names": ["normal", "pneumonia"],
        # published MSF 28px reference (paper Table 2)
        "c1_auc": 0.952, "c1_acc": 0.880,
        "gen_flips": False,          # matches the repo's pneumonia train loader
    },
    "retinamnist": {
        "medmnist_class": "RetinaMNIST",
        "n_classes": 5,
        "channels": 3,
        "splits": (1080, 120, 400),
        "class_names": ["grade0", "grade1", "grade2", "grade3", "grade4"],
        "c1_auc": 0.731, "c1_acc": 0.514,
        "gen_flips": True,           # repo's retina train loader uses H+V flips
    },
    "dermamnist": {
        "medmnist_class": "DermaMNIST",
        "n_classes": 7,
        "channels": 3,
        "splits": (7007, 1003, 2005),
        "class_names": ["actinic", "bcc", "keratosis", "dermatofibroma",
                        "melanoma", "nevus", "vascular"],
        "c1_auc": 0.896, "c1_acc": 0.788,
        "gen_flips": False,
    },
}


def dataset_meta(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]
