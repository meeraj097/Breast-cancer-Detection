import os
import gdown

def download_weights():
    os.makedirs("weights", exist_ok=True)
    gdown.download("https://drive.google.com/uc?id=1--7p9rRJy7WU4OmomkzM8i0veetZctTT", "weights/modeldense1.h5", quiet=False)
