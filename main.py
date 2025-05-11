from adaptive_classifier import AdaptiveClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv("./data/training.csv")
test = pd.read_csv("./data/test.csv")

clf = AdaptiveClassifier("distilbert-base-uncased")

clf.add_examples(list(train["text"]), list(train["label"]))

prediction = clf.predict(test["text"])

accuracy = accuracy_score(test["text"], prediction)
print(f"Accuracy: {accuracy}")

# Отчет по классификации
print("Classification Report:\n", classification_report(test["text"], test["text"]))
