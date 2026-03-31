# Fake Review Detector (Complete Project in One File)

import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# STEP 1: CREATE DATASET
# ----------------------------

fake_reviews = [
    "Amazing product!!! Best ever buy now!!!",
    "Excellent quality, highly recommended!!!!!",
    "Wow this is perfect, must buy!!!",
    "Cheap and best product ever",
    "I love this product so much!!!!",
    "Best thing I have purchased online",
    "Superb quality, 5 stars!!!",
    "Totally worth it!!! Buy now",
    "Fantastic product, unbelievable price",
    "Highly recommended to everyone!!!"
]

real_reviews = [
    "The product is decent and works as expected",
    "Quality is okay but could be better",
    "I received the item on time and it works fine",
    "Not bad, but not excellent either",
    "The material feels average",
    "It does the job but has some issues",
    "Packaging was good, performance is average",
    "Satisfied with the purchase overall",
    "The product is useful but slightly expensive",
    "Good but not great"
]

# Expand dataset
data = []
for _ in range(200):
    data.append([random.choice(fake_reviews), 1])  # Fake = 1
    data.append([random.choice(real_reviews), 0])  # Real = 0

df = pd.DataFrame(data, columns=["review", "label"])

# ----------------------------
# STEP 2: CLEAN TEXT
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

# ----------------------------
# STEP 3: FEATURE EXTRACTION
# ----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_review"])
y = df["label"]

# ----------------------------
# STEP 4: TRAIN MODEL
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# STEP 5: EVALUATION
# ----------------------------

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# STEP 6: PREDICTION FUNCTION
# ----------------------------

def predict_review(review):
    review_clean = clean_text(review)
    review_vector = vectorizer.transform([review_clean])
    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        return "⚠️ Fake Review"
    else:
        return "✅ Genuine Review"

# ----------------------------
# STEP 7: USER INPUT
# ----------------------------

print("\n--- Fake Review Detector ---")

while True:
    user_input = input("\nEnter a review (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result = predict_review(user_input)
    print("Result:", result)