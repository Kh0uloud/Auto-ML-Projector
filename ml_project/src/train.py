from sklearn.tree import DecisionTreeClassifier
import optuna
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
def prepare_data(data, test_size=0.15, random_state=42):
    X = data.drop(['id', 'depression'], axis=1)
    y = data['depression']
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

def objective(trial, model_name, X_train, y_train, X_test, y_test, random_state=42):
    if model_name == 'GradientBoosting':
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5), 'random_state': random_state}
        model = GradientBoostingClassifier(**params)
    elif model_name == 'RandomForest':
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100), 'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5), 'random_state': random_state}
        model = RandomForestClassifier(**params)
    elif model_name == 'AdaBoost':
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True), 'random_state': random_state}
        model = AdaBoostClassifier(**params)
    elif model_name == 'DecisionTree':
        params = {'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']), 'max_depth': trial.suggest_categorical('max_depth', [5, 10, 20, 30, None]), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5), 'random_state': random_state}
        model = DecisionTreeClassifier(**params)
    elif model_name == 'SVM':
        params = {'C': trial.suggest_loguniform('C', 0.01, 1000), 'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']), 'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1]), 'random_state': random_state}
        model = SVC(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def optimize_models(data, n_trials=30):
    (X_train, X_test, y_train, y_test) = prepare_data(data)
    models = ['SVM', 'GradientBoosting', 'RandomForest', 'AdaBoost', 'DecisionTree']
    best_results = []
    for model_name in tqdm(models):
        print(f'\nOptimizing {model_name}...')
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), n_trials=n_trials)
        print(f'Best Params for {model_name}: {study.best_params}\nBest Accuracy: {study.best_value:.4f}')
        best_results.append({'Model': model_name, 'Best Params': study.best_params, 'Best Accuracy': study.best_value})
    return pd.DataFrame(best_results)

