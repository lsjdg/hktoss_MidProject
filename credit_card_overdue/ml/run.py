from data import *
from trainer import *
from parameters.params import *
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

def save_results(results, file_path):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)


def plot_model_comparison(results):
    df = pd.DataFrame(results)
    df.plot(kind='bar', x='Model', y='Performance Metric')
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Performance Metric')
    plt.show()

def run():
    # Load and preprocess data
    df = load_data()
    X = df.drop('credit', axis=1)
    y = df['credit']
    

    results = []

    # Model tuning and training
    for model_name, param_grid in param_grids.items():
        print(f"Tuning {model_name}...")
        
        if model_name == 'RandomForest':
            model = RandomForestClassifier()
        elif model_name == 'XGBoost':
            model = XGBClassifier()
        elif model_name == 'LightGBM':
            model = LGBMClassifier()
        else:
            raise ValueError("Unknown model")

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        
        # Save best parameters and results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        results.append({
            'Model': model_name,
            'Best Parameters': best_params,
            'Best Score': best_score
        })
        
        # Save results to CSV
        save_results(results, 'model_results.csv')

    # Compare models and select the best one
    plot_model_comparison(results)

if __name__ == "__main__":
    run()

