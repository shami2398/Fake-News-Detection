import pandas as pd
from tkinter import *
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ===== TRAIN MODEL =====
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])
df = df.dropna()

df["content"] = (df["title"] + " " + df["text"]).str.lower()

X = df["content"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ===== PREDICT FUNCTION =====
def predict_news():
    news = entry.get("1.0", END).strip()

    if len(news) < 20:
        output_label.config(text="Enter proper news ⚠️", fg="orange")
        return

    news = news.lower()
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)

    if result[0] == 0:
        output_label.config(text="Fake News ❌", fg="red")
    else:
        output_label.config(text="Real News ✅", fg="green")

# ===== GUI =====
root = Tk()
root.title("Fake vs Real News Classification")
root.geometry("650x520")
root.config(bg="#ffc0cb")

Label(root, text="Fake vs Real News Classification",
      font=("Arial", 20, "bold"),
      bg="#ffc0cb").pack(pady=10)

img = Image.open("nicon.png")
img = img.resize((80, 80))
img = ImageTk.PhotoImage(img)
Label(root, image=img, bg="#ffc0cb").pack()

Label(root, text="Enter News Text:",
      bg="#ffc0cb", font=("Arial", 12)).pack()

entry = Text(root, height=8, width=70)
entry.pack(pady=10)

Button(root, text="Check News",
       command=predict_news,
       bg="#ff69b4", fg="white",
       font=("Arial", 12, "bold")).pack()

output_label = Label(root, text="", font=("Arial", 16, "bold"),
                     bg="#ffc0cb")
output_label.pack(pady=20)

root.mainloop()