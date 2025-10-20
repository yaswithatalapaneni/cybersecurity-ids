
#  Intrusion Detection System (IDS) using ML
# Dataset: NSL-KDD
# Author: Yaswitha talapaneni 


# ---- STEP 1: Import Libraries ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---- STEP 2: Load the Dataset ----
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty_level'
]

print("📂 Loading dataset...")
train = pd.read_csv("data/KDDTrain+.txt", names=columns)
test = pd.read_csv("data/KDDTest+.txt", names=columns)
print("✅ Dataset loaded successfully!\n")

# ---- STEP 3: Encode Categorical Variables ----
print("🔡 Encoding categorical columns...")
categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    encoder = LabelEncoder()
    encoder.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = encoder.transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))

print("✅ Encoding complete!\n")


# ---- STEP 4: Split Features & Target ----
X_train = train.drop(['label', 'difficulty_level'], axis=1)
y_train = train['label']
X_test = test.drop(['label', 'difficulty_level'], axis=1)
y_test = test['label']

# ---- STEP 5: Feature Scaling ----
print("⚙️ Scaling numerical features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Feature scaling done!\n")

# ---- STEP 6: Train & Compare Multiple Models ----
print("🚀 Training and comparing multiple models...\n")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {round(acc * 100, 2)}% accuracy")

# ---- STEP 7: Pick Best Model ----
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} with {round(results[best_model_name] * 100, 2)}% accuracy\n")

# ---- STEP 8: Detailed Evaluation of Best Model ----
y_pred = best_model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- STEP 9: Feature Importance (Random Forest only) ----
if best_model_name == "Random Forest":
    print("🔍 Showing top 10 important features...")
    rf = best_model
    feat_importances = pd.Series(
        rf.feature_importances_, 
        index=train.drop(['label','difficulty_level'], axis=1).columns
    )
    plt.figure(figsize=(8,5))
    feat_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title("Top 10 Important Features for IDS")
    plt.xlabel("Importance Score")
    plt.show()

# ---- STEP 10: Save Model ----
joblib.dump(best_model, "ids_model.pkl")
print(f"\n💾 Model saved successfully as ids_model.pkl (Best Model: {best_model_name})")

# ---- STEP 11: Test Reload ----
loaded_model = joblib.load("ids_model.pkl")
reload_acc = loaded_model.score(X_test, y_test)
print(f"✅ Reloaded model works! Accuracy: {round(reload_acc * 100, 2)}%")
