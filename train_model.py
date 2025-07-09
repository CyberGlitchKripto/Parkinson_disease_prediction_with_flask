# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# === Load Dataset ===
df = pd.read_csv('parkinsons_updrs.csv')

# === Basic Info ===
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe().T)
print("\nMissing Values:", df.isnull().sum().sum())

# === Handle Categorical Features ===
if df['sex'].dtype == 'object':
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    joblib.dump(le, 'label_encoder.pkl')  # Save for Flask

# === Group by subject# & Drop ===
# Group by subject# and compute mean
df = df.groupby('subject#').mean().reset_index()
# Create binary class column *after* grouping using mean motor_UPDRS
df['class'] = (df['motor_UPDRS'] > 20).astype(int)
df.drop(columns=['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS'], inplace=True)

# === Check Class Distribution ===
plt.figure(figsize=(6, 6))
plt.pie(df['class'].value_counts(),
        labels=df['class'].value_counts().index,
        autopct='%1.1f%%')
plt.title("Class Distribution")
plt.show()

# === Correlation Filter (0.7 threshold) ===
cor_matrix = df.drop(columns=['class']).corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
df.drop(columns=to_drop, inplace=True)
print("\nShape after removing correlated features:", df.shape)
joblib.dump(to_drop, 'to_drop.pkl')  # Save for Flask

# === Feature Selection with Chi2 ===
X = df.drop(columns=['class'])
y = df['class']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save for Flask

selector = SelectKBest(chi2, k=min(30, X.shape[1]))  # Ensure k doesn't exceed feature count
X_selected = selector.fit_transform(X_scaled, y)

selected_columns = X.columns[selector.get_support()]
df = pd.DataFrame(X_selected, columns=selected_columns)
df['class'] = y.reset_index(drop=True)
print("\nShape after feature selection:", df.shape)
joblib.dump(selector, 'selector.pkl')  # Save for Flask

# === Train-Test Split ===
X = df.drop(columns=['class'])
y = df['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Handle Imbalance ===
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print("\nShape of resampled training data:", X_resampled.shape)
print("Class distribution after resampling:")
print(y_resampled.value_counts())

# === Train and Evaluate Models ===
models = [
    LogisticRegression(class_weight='balanced'),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    SVC(kernel='rbf', probability=True)
]
train_roc_scores = []
val_roc_scores = []
train_f1_scores = []
val_f1_scores = []

for model in models:
    print(f"\nModel: {model.__class__.__name__}")
    model.fit(X_resampled, y_resampled)

    train_preds = model.predict(X_resampled)
    val_preds = model.predict(X_val)

    train_roc = roc_auc_score(y_resampled, train_preds)
    val_roc = roc_auc_score(y_val, val_preds)
    train_f1 = classification_report(y_resampled, train_preds, output_dict=True)['weighted avg']['f1-score']
    val_f1 = classification_report(y_val, val_preds, output_dict=True)['weighted avg']['f1-score']

    print("Train ROC AUC:", train_roc)
    print("Train F1 Score:", train_f1)
    print("Validation ROC AUC:", val_roc)
    print("Validation F1 Score:", val_f1)
    print("Classification Report (Validation):\n", classification_report(y_val, val_preds))

    train_roc_scores.append(train_roc)
    val_roc_scores.append(val_roc)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)

# === Save Best Model ===
best_model_idx = np.argmax(val_f1_scores)
best_model = models[best_model_idx]
joblib.dump(best_model, 'best_model.pkl')
print(f"Saved {models[best_model_idx].__class__.__name__} as best_model.pkl")
print("\nBest Model Summary:")
print(f"📌 Best Model: {best_model.__class__.__name__}")
print(f"✅ Validation F1 Score: {val_f1_scores[best_model_idx]:.4f}")
print(f"✅ Validation ROC AUC: {val_roc_scores[best_model_idx]:.4f}")

# === Confusion Matrix for Logistic Regression ===
ConfusionMatrixDisplay.from_estimator(models[0], X_val, y_val)
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# === Model Performance Comparison ===
plt.figure(figsize=(10, 6))
models_names = [model.__class__.__name__ for model in models]
x = np.arange(len(models_names))
width = 0.2

plt.bar(x - width - width/2, train_roc_scores, width, label='Train ROC AUC', color='#1f77b4')
plt.bar(x - width/2, val_roc_scores, width, label='Validation ROC AUC', color='#ff7f0e')
plt.bar(x + width/2, train_f1_scores, width, label='Train F1 Score', color='#2ca02c')
plt.bar(x + width + width/2, val_f1_scores, width, label='Validation F1 Score', color='#d62728')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(x, models_names)
plt.legend()
plt.tight_layout()
plt.show()