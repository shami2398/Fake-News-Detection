import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage

# Load dataset
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])
df = df.sample(frac=1)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


def check_news():
    news = entry.get()

    if news == "":
        messagebox.showwarning("Warning", "Please enter a news headline")
        return

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    if prediction[0] == 0:
        result_label.config(text="Fake News ❌", fg="red")
    else:
        result_label.config(text="Real News ✅", fg="green")


# GUI Window
window = tk.Tk()
window.title("Fake News Detector")
window.geometry("550x420")
window.configure(bg="#e6f2ff")

# Image banner
img = PhotoImage(file="assets/nicon.png")
img_label = tk.Label(window, image=img, bg="#e6f2ff")
img_label.pack(pady=10)

# Title
title = tk.Label(window, text="Fake News Detection System",
                 font=("Helvetica", 18, "bold"),
                 bg="#e6f2ff")
title.pack(pady=5)

# Input label
label = tk.Label(window, text="Enter News Headline:",
                 font=("Arial", 12),
                 bg="#e6f2ff")
label.pack()

# Entry box
entry = tk.Entry(window, width=60, font=("Arial", 11))
entry.pack(pady=10)

# Button
check_button = tk.Button(window,
                         text="🔍 Check News",
                         font=("Arial", 12),
                         bg="#4CAF50",
                         fg="white",
                         padx=10,
                         pady=5,
                         command=check_news)
check_button.pack(pady=15)

# Result label
result_label = tk.Label(window,
                        text="Result will appear here",
                        font=("Arial", 14, "bold"),
                        bg="#e6f2ff")
result_label.pack(pady=20)

window.mainloop()