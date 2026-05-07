import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# ================= LOAD DATA =================
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])
df = df.dropna()
df = df.sample(frac=1).reset_index(drop=True)

df["content"] = (df["title"] + " " + df["text"]).str.lower()

X = df["content"]
y = df["label"]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= MODELS =================
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

lr.fit(X_train_vec, y_train)
rf.fit(X_train_vec, y_train)

# ================= PREDICTIONS =================
lr_pred = lr.predict(X_test_vec)
rf_pred = rf.predict(X_test_vec)

# ================= METRICS =================
lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

lr_prob = lr.predict_proba(X_test_vec)[:, 1]
rf_prob = rf.predict_proba(X_test_vec)[:, 1]

lr_auc = roc_auc_score(y_test, lr_prob)
rf_auc = roc_auc_score(y_test, rf_prob)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

# ================= PRINT =================
print("\n===== MODEL RESULTS =====")
print(f"Logistic Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

print("\nLogistic Confusion Matrix:\n", lr_cm if 'lr_cm' in globals() else confusion_matrix(y_test, lr_pred))
print("\nRandom Forest Confusion Matrix:\n", rf_cm if 'rf_cm' in globals() else confusion_matrix(y_test, rf_pred))

# ================= PLOTS =================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2)

# Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(["Logistic", "Random Forest"], [lr_acc * 100, rf_acc * 100])
ax1.set_title("Model Accuracy")

# ROC AUC
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(["Logistic", "Random Forest"], [lr_auc, rf_auc])
ax2.set_title("ROC-AUC Score")

# Dataset Distribution
ax3 = fig.add_subplot(gs[1, 0])
labels = ["Fake", "Real"]
counts = [len(fake_df), len(true_df)]
ax3.pie(counts, labels=labels, autopct="%1.1f%%")
ax3.set_title("Dataset Distribution")

# ROC Curve
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(lr_fpr, lr_tpr, label="Logistic")
ax4.plot(rf_fpr, rf_tpr, label="Random Forest")
ax4.plot([0, 1], [0, 1], '--')
ax4.legend()
ax4.set_title("ROC Curve")

# ===== FIXED LAYOUT =====
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.1)

# ===== FULL SCREEN =====
mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed')
except:
    pass

plt.show()