import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
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

# === Improved Data Aggregation ===
# Define constant and time-varying features
constant_features = ['age', 'sex']
time_varying_features = [col for col in df.columns if col not in ['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS'] + constant_features]

def extract_features(group):
    last_row = group.iloc[-1]
    constants = last_row[constant_features]
    last_values = last_row[time_varying_features]
    std_values = group[time_varying_features].std().add_suffix('_std')
    motor_UPDRS_last = last_row['motor_UPDRS']
    return pd.concat([constants, last_values, std_values, pd.Series({'motor_UPDRS_last': motor_UPDRS_last})])

# Sort by subject# and test_time, then extract features
df = df.sort_values(by=['subject#', 'test_time'])
df = df.groupby('subject#').apply(extract_features).reset_index()

# Create class based on last motor_UPDRS and drop unnecessary columns
df['class'] = (df['motor_UPDRS_last'] > 20).astype(int)
df.drop(columns=['subject#', 'motor_UPDRS_last'], inplace=True)

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

# === Train-Test Split ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df.drop(columns=['class'])
y_train = train_df['class']
X_val = val_df.drop(columns=['class'])
y_val = val_df['class']
print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# === Feature Selection with SelectFromModel ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold='mean')
sfm.fit(X_train, y_train)
X_train_selected = sfm.transform(X_train)
X_val_selected = sfm.transform(X_val)
selected_columns = X_train.columns[sfm.get_support()]
print("\nNumber of selected features:", len(selected_columns))
joblib.dump(sfm, 'selector.pkl')  # Save for Flask

# === Scale Selected Features ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
joblib.dump(scaler, 'scaler.pkl')  # Save for Flask

# === Handle Imbalance ===
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
print("\nShape of resampled training data:", X_resampled.shape)
print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts())

# === Define Hyperparameter Grids ===
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# === Train and Tune Models ===
models = [
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    RandomForestClassifier(random_state=42)
]
model_names = ['XGBoost', 'RandomForest']
param_grids = [xgb_param_grid, rf_param_grid]
best_models = []

for model, name, param_grid in zip(models, model_names, param_grids):
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    print(f"Best {name} Params:", grid_search.best_params_)
    best_models.append(grid_search.best_estimator_)

# === Evaluate Models ===
val_f1_scores = []
val_roc_scores = []

for model, name in zip(best_models, model_names):
    print(f"\nModel: {name}")
    val_preds = model.predict(X_val_scaled)
    val_roc = roc_auc_score(y_val, val_preds)
    val_f1 = classification_report(y_val, val_preds, output_dict=True)['weighted avg']['f1-score']
    print("Validation ROC AUC:", val_roc)
    print("Validation F1 Score:", val_f1)
    print("Classification Report (Validation):\n", classification_report(y_val, val_preds))
    val_f1_scores.append(val_f1)
    val_roc_scores.append(val_roc)

# === Save Best Model ===
best_model = best_models[np.argmax(val_f1_scores)]
best_model_name = model_names[np.argmax(val_f1_scores)]
joblib.dump(best_model, 'best_model.pkl')
print(f"Saved {best_model_name} as best_model.pkl")
print("\nBest Model Summary:")
print(f"📌 Best Model: {best_model_name}")
print(f"Validation F1 Score: {val_f1_scores[np.argmax(val_f1_scores)]:.4f}")
print(f"Validation ROC AUC: {val_roc_scores[np.argmax(val_f1_scores)]:.4f}")

# === Confusion Matrix for Best Model ===
ConfusionMatrixDisplay.from_estimator(best_model, X_val_scaled, y_val)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.show()

# === Baseline Check ===
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_scaled, y_train)
val_preds_dummy = dummy.predict(X_val_scaled)
print("\nBaseline F1 Score:", classification_report(y_val, val_preds_dummy, output_dict=True)['weighted avg']['f1-score'])