{
    "paths": {
        "base_dir": "~/storage",
        "train_dir": "sliced_train_data",
        "save_model_name": "efficientnetv2_m_sliced_train_data.pth"
    },
    "hyperparameters": {
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 0.0001,
        "image_size": 480,
        "seed": 42,
        "val_split": 0.2
    },
    "classes": [
        "hindi", "english", "assamese", "bengali",
        "gujarati", "kannada", "malayalam", "marathi",
        "odia", "punjabi", "tamil", "telugu"
    ]
}
