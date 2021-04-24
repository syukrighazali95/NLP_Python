# https://realpython.com/sentiment-analysis-python/
import os
import random
import spacy
from spacy.util import minibatch, compounding

def load_training_data(data_directory: str = "datasets/aclImdb/train", split: float = 0.8, limit: int = 0) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        # print(os.listdir(labeled_directory))
        for review in os.listdir(labeled_directory):
            # print(review)
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
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
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

def train_model(training_data: list, test_data: list, iterations: int = 20) -> None:
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    
    print(textcat)
    textcat.add_label("pos")
    textcat.add_label("neg")

    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    # print(training_excluded_pipes)



train, test = load_training_data(limit = 2500)
train_model(train, test)