import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import time
import shap

def RF_ensemble(date_range, ensemble_composition, time_series_data_training, time_series_data_testing):


    # Features and target for training
    X_train = time_series_data_training.drop(columns=['actual', 'date'])
    y_train = time_series_data_training['actual']

    # Features and target for testing
    X_test =time_series_data_testing.drop(columns=['actual', 'date'])
    y_test = time_series_data_testing['actual']

    starting_time = time.time()

    # Grid search for RandomForestRegressor
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True],
        'random_state': [50]
    }

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Print the best parameters
    print(f"Best Parameters: {grid_search.best_params_}")

    ending_time = time.time()

    # Evaluate the best model
    y_train_pred_best = best_model.predict(X_train)
    y_test_pred_best = best_model.predict(X_test)

    #save the RF results
    train_full_df = pd.DataFrame({
        'Date': time_series_data_training['date'],
        'Actual': time_series_data_training['actual'],
        'Predicted': y_train_pred_best
    })

    test_full_df = pd.DataFrame({
        'Date': time_series_data_testing['date'],
        'Actual': time_series_data_testing['actual'],
        'Predicted': y_test_pred_best
    })

    # Save with additional context
    train_full_df.to_csv(f"../forecasting_results/random_forest_results/y_train_predictions_with_actual_{ensemble_composition}_{date_range}.csv", index=False)
    test_full_df.to_csv(f"../forecasting_results/random_forest_results/y_test_predictions_with_actual_{ensemble_composition}_{date_range}.csv", index=False)

    # Calculate metrics
    train_mse_best = mean_squared_error(y_train, y_train_pred_best)
    test_mse_best = mean_squared_error(y_test, y_test_pred_best)
    train_mape_best = mean_absolute_percentage_error(y_train, y_train_pred_best)
    test_mape_best = mean_absolute_percentage_error(y_test, y_test_pred_best)
    train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
    test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
    train_r2_best = r2_score(y_train, y_train_pred_best)
    test_r2_best = r2_score(y_test, y_test_pred_best)

    # Define metrics
    metrics = {
        'Metric': ['MSE', 'MAPE', 'MAE', 'R2'],
        'Train': [train_mse_best, train_mape_best, train_mae_best, train_r2_best],
        'Test': [test_mse_best, test_mape_best, test_mae_best, test_r2_best]
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)

    output_filename = f"../forecasting_results/random_forest_results/RF_evaluation_metrics_{ensemble_composition}_{date_range}.csv"
    metrics_df.to_csv(output_filename, index=False)

    # Plot 1: Training Set - Actual vs Fitted
    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data_training['date'], y_train, label='Actual', alpha=0.8)
    plt.plot(time_series_data_training['date'], y_train_pred_best, label='Fitted (Training)', linestyle='--', alpha=0.8)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Ensemble Model- Training Set: Actual vs Fitted\nMSE: {train_mse_best:.2f}')
    output_filename = f"../forecasting_results/random_forest_results/RF_Ensemble Model-Training_Actual_vs_Fitted_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    # Plot 2: Testing Set - Actual vs Forecasted
    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data_testing['date'], y_test, label='Actual')
    plt.plot(time_series_data_testing['date'], y_test_pred_best, label='Forecast (Testing)', linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Ensemble Model-- Testing Set: Actual vs Forecast\nMSE: {test_mse_best:.2f}')
    output_filename = f"../forecasting_results/random_forest_results/RF_Ensemble Model-Testing_Actual_vs_Forecast_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    # Plot 3: R² for Training Set
    plt.figure(figsize=(12, 6))
    plt.scatter(y_train, y_train_pred_best, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Fitted Values')
    plt.title(f'Ensemble Model-Training Set: R² = {train_r2_best:.2f}')
    output_filename = f"../forecasting_results/random_forest_results/RF_Ensemble_Model-Training_R2_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    # Plot 4: R² for Testing Set
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_test_pred_best, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Ensemble Model-Testing Set: R² = {test_r2_best:.2f}')
    output_filename = f"../forecasting_results/random_forest_results/RF_Ensemble_Model-Testing_R2_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    # Select the first tree from the best model
    tree = best_model.estimators_[0]

    # Plot the tree
    feature_names = X_train.columns

    plt.figure(figsize=(20, 10))
    plot_tree(tree, filled=True, feature_names=feature_names, fontsize=10)
    output_filename = f"../forecasting_results/random_forest_results/RF_Ensemble_one_tree_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    best_params = grid_search.best_params_

    # Convert dictionary to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # Save to CSV
    output_filename = f"../forecasting_results/random_forest_results/RF_best_model_params_{ensemble_composition}_{date_range}.csv"
    df_best_params.to_csv(output_filename, index=False)



    # Get feature importance
    importance = best_model.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame
    importance_df = pd.DataFrame({
        'Forecaster': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Save to CSV
    csv_filename = f"../forecasting_results/random_forest_results/feature_importance_{ensemble_composition}_{date_range}.csv"
    importance_df.to_csv(csv_filename, index=False)

    plt.figure(figsize=(8, 6))
    plt.bar(importance_df['Forecaster'], importance_df['Importance'])
    plt.title('Base Forecaster Contribution to RF Ensemble')
    plt.ylabel('Feature Importance')
    plt.xlabel('Base Forecaster')
    plt.xticks(rotation=90)  # Just rotate, no need to pass labels

    plot_filename = f"../forecasting_results/random_forest_results/feature_importance_plot_{ensemble_composition}_{date_range}.png"
    plt.tight_layout()  # Optional: fix label cutoff
    plt.savefig(plot_filename)
    plt.show()

    training_time = ending_time-starting_time

    # Create SHAP summary plot and save as PDF
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_train)


    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)  # Prevent automatic display
    output_filename = f"../forecasting_results/random_forest_results/shap_plot{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.close()

    return(training_time)

