import os
import random

def load_training_data(data_directory: str = "datasets/aclImdb/train", split: float = 0.8, limit: int = 0):
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        # print(os.listdir(labeled_directory))
        for review in os.listdir(labeled_directory):
            # print(review)
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))

load_training_data()