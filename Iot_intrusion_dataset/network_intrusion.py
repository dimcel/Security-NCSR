import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, categorical_columns, binary_columns, numerical_columns, scaler=None, fit_scaler=True):
    df = pd.get_dummies(df, columns=categorical_columns)
    df[binary_columns] = df[binary_columns].astype(int)
    if fit_scaler:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df, scaler

def split_data(df, target_column='class', test_size=0.2, random_state=1):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, 50, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=1, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix

def plot_confusion_matrix(conf_matrix, labels, output_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()

def plot_feature_importances(model, X, output_path, top_n=20):
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Main execution
df = load_data('data/Train_data.csv')
categorical_columns = ['protocol_type', 'service', 'flag'] 
binary_columns = ['land', 'urgent', 'logged_in', 'root_shell', 'num_shells', 'is_guest_login', 'serror_rate',
                  'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                  'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                  'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
numerical_columns = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot', 'num_failed_logins', 'num_compromised',
                     'su_attempted', 'num_root', 'num_file_creations', 'num_access_files', 'num_outbound_cmds', 
                     'is_host_login', 'count', 'srv_count']

df, scaler = preprocess_data(df, categorical_columns, binary_columns, numerical_columns)
X_train, X_test, y_train, y_test = split_data(df)

model, best_params = train_model(X_train, y_train)
accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)

print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print(report)

with open("metrics/evaluation.txt", "w") as file:
    file.write(f"Best Parameters: {best_params}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(report)

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, set(y_test), 'metrics/confusion_matrix.png')

# Plot feature importances
plot_feature_importances(model, X_train, 'metrics/feature_importances.png')

# Load and preprocess test data
df_test = load_data('data/Test_data.csv')
df_test, _ = preprocess_data(df_test, categorical_columns, binary_columns, numerical_columns, scaler=scaler, fit_scaler=False)
y_test_pred = model.predict(df_test)


unique, counts = np.unique(y_test_pred, return_counts=True)
prediction_distribution = dict(zip(unique, counts))

print("Predicted label distribution on test data:", prediction_distribution)


#todo: add code to save the model
#todo: fix the collums in train test for inference