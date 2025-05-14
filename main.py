import torch.cuda
from adaptive_classifier import AdaptiveClassifier
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("./data/training.csv")
test_df = pd.read_csv("./data/test.csv")

X_train = train_df["text"].tolist()
y_train = train_df["label"].tolist()

X_test = test_df["text"].tolist()
y_test = test_df["label"].tolist()

print(f"Cuda is {'available' if torch.cuda.is_available() else 'unavailable'}")

clf = AdaptiveClassifier(
    "bert-base-uncased",
    "cuda"
)

print("Read data, starting training")

for i in tqdm(range(len(X_train))):
    clf.add_examples(X_train[i:i+1], y_train[i:i+1])

clf.save("./clf")

print("Training done")

prediction = clf.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy}")

# Отчет по классификации
print("Classification Report:\n", classification_report(prediction, y_test))
