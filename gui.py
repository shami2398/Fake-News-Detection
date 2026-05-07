import pandas as pd
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ===== LOAD DATASETS =====
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# ===== ADD LABELS =====
fake_df["label"] = 0
true_df["label"] = 1

# ===== COMBINE DATA =====
df = pd.concat([fake_df, true_df])

# Remove empty rows
df = df.dropna()

# Combine title + text
df["content"] = (df["title"] + " " + df["text"]).str.lower()

# ===== FEATURES AND LABELS =====
X = df["content"]
y = df["label"]

# ===== TEXT VECTORIZATION =====
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_vec = vectorizer.fit_transform(X)

# ===== TRAIN MODEL =====
model = MultinomialNB()
model.fit(X_vec, y)

# ===== PREDICTION FUNCTION =====
def predict_news():

    # Get input text
    news = entry.get("1.0", END).strip()

    # Empty input check
    if news == "":
        output_label.config(
            text="Please enter news text ⚠️",
            fg="orange"
        )
        return

    # Small input check
    if len(news) < 20:
        output_label.config(
            text="Enter proper news ⚠️",
            fg="orange"
        )
        return

    # Convert to lowercase
    news = news.lower()

    # Convert text into vectors
    news_vec = vectorizer.transform([news])

    # Predict
    result = model.predict(news_vec)

    # Show output
    if result[0] == 0:

        output_label.config(
            text="Fake News ❌",
            fg="red"
        )

    else:

        output_label.config(
            text="Real News ✅",
            fg="green"
        )

# ===== GUI WINDOW =====
root = Tk()

root.title("Fake vs Real News Classification")

root.geometry("700x650")

root.config(bg="#ffc0cb")

# ===== TITLE =====
title = Label(
    root,
    text="Fake vs Real News Classification",
    font=("Arial", 24, "bold"),
    bg="#ffc0cb",
    fg="black"
)

title.pack(pady=10)

# ===== IMAGE =====
img = Image.open("assets/nicon.png")

# Resize image
img = img.resize((180, 180))

# Convert image for tkinter
img = ImageTk.PhotoImage(img)

# Display image
img_label = Label(
    root,
    image=img,
    bg="#ffc0cb"
)

img_label.pack(pady=10)

# ===== INPUT LABEL =====
input_label = Label(
    root,
    text="Enter News Text:",
    font=("Arial", 14, "bold"),
    bg="#ffc0cb"
)

input_label.pack()

# ===== TEXT AREA =====
entry = Text(
    root,
    height=10,
    width=75,
    font=("Arial", 11)
)

entry.pack(pady=10)

# ===== BUTTON =====
check_button = Button(
    root,
    text="Check News",
    command=predict_news,
    bg="#ff1493",
    fg="white",
    font=("Arial", 13, "bold"),
    padx=15,
    pady=5
)

check_button.pack(pady=10)

# ===== RESULT LABEL =====
output_label = Label(
    root,
    text="Result will appear here",
    font=("Arial", 18, "bold"),
    bg="#ffc0cb"
)

output_label.pack(pady=20)

# ===== RUN GUI =====
root.mainloop()