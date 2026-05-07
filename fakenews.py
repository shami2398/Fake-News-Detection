import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load fake and real news datasets
pd.read_csv("dataset/Fake.csv")
pd.read_csv("dataset/True.csv")

# Add labels
fake_df["label"] = 0   # Fake news
true_df["label"] = 1   # Real news

# Combine both datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle data
df = df.sample(frac=1, random_state=42)

# Input and output
X = df["text"]
y = df["label"]

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Take user input
news = input("\nEnter a news article or headline: ")

# Transform input text
news_vec = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vec)

if prediction[0] == 0:
    print("Prediction: Fake News")
else:
    print("Prediction: Real News")