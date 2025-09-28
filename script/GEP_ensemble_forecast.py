import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import geppy as gep
from deap import tools, creator, base
from itertools import product
import operator
import time
import shap
from sklearn.preprocessing import StandardScaler
from deap import creator
import random
import os, numpy as np


os.environ["PYTHONHASHSEED"] = "50"
SEED = 50
random.seed(SEED)
np.random.seed(SEED)



if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


def GEP_ensemble(date_range, ensemble_composition, time_series_data_training, time_series_data_testing):
    # Features and target for training
    x_scaler = StandardScaler()
    X_train_raw = time_series_data_training.drop(columns=['actual', 'date']).astype(float).values
    X_test_raw = time_series_data_testing.drop(columns=['actual', 'date']).astype(float).values

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_train = time_series_data_training['actual'].astype(float).values
    y_test = time_series_data_testing['actual'].astype(float).values

    RANDOM_SEED = 60
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


    def protected_div(x1, x2):
        return x1 / (x2 + 1e-10)

    def protected_log(x):
        x = np.array(x, dtype=float)  # Ensure input is float
        x = np.where(x <= 0, 1e-10, x)  # Avoid log(0) or log(negative)
        return np.log(x)

    def protected_exp(x):
        return np.exp(np.clip(x, -100, 100))

    def square(x):
        x = np.clip(x, -1e5, 1e5)
        return x * x

    def power_3(x):
        x = np.clip(x, -1e3, 1e3)
        return x ** 3

    def slog(x):  # safe log
        x = np.asarray(x, dtype=float)
        return np.log(np.abs(x) + 1e-10)

    def tanh_(x):
        return np.tanh(np.clip(x, -20, 20))

    def sin_(x):
        return np.sin(np.clip(x, -1e3, 1e3))

    def cos_(x):
        return np.cos(np.clip(x, -1e3, 1e3))

    def erc_wide(): # to generate Ephemeral Random Constant
        return float(random.uniform(-5.0, 5.0))

    def erc_narrow(): # to generate Ephemeral Random Constant
        return float(random.uniform(-1.0, 1.0))


    # Create primitive set with those column names
    input_columns = time_series_data_training.drop(columns=['actual', 'date']).columns.tolist()
    pset = gep.PrimitiveSet('MAIN', input_names=input_columns)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(protected_div, 2)
    pset.add_function(protected_exp, 1)
    pset.add_constant_terminal(np.pi)
    pset.add_constant_terminal(1)
    pset.add_constant_terminal(2)
    pset.add_constant_terminal(3)
    pset.add_function(slog, 1)
    pset.add_function(tanh_, 1)
    pset.add_function(sin_, 1)
    pset.add_function(cos_, 1)
    pset.add_ephemeral_terminal('ERC_WIDE', erc_wide)
    pset.add_ephemeral_terminal('ERC_NARROW', erc_narrow)

    # Define fitness and individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create('Individual', gep.Chromosome, fitness=creator.FitnessMin)

    # Define a custom linker function
    def custom_linker(*args):
        return sum(args)

    # Initialize the toolbox
    toolbox = gep.Toolbox()
    toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=10)
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=9, linker=custom_linker)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gep.compile_, pset=pset)

    # Evaluation function

    def evaluate(individual):
        func = toolbox.compile(individual)
        try:
            raw_predictions = [func(*row) for row in X_train]
            predictions = np.nan_to_num(raw_predictions, nan=0.0, posinf=1e10, neginf=-1e10)
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'),  # Penalize bad individuals

        return mean_squared_error(y_train, predictions),

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.1)
    toolbox.register('mut_invert', gep.invert, pb=0.1)
    toolbox.register('mut_is_ts', gep.is_transpose, pb=0.1)
    toolbox.register('mut_ris_ts', gep.ris_transpose, pb=0.1)
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)

    starting_time = time.time()

    # Grid search parameters
    param_grid = {
        'head_size': [10, 15, 20],
        'n_pop': [200, 300],
        'n_genes': [len(input_columns)]
    }

    # Create combinations of parameters
    param_combinations = list(product(param_grid['head_size'], param_grid['n_pop'], param_grid['n_genes']))

    # Store results
    grid_search_results = []

    # Perform grid search
    for head_size, n_pop, n_genes in param_combinations:

        random.seed(SEED)
        np.random.seed(SEED)
        print(f"Testing combination: head_size={head_size}, n_pop={n_pop}, n_genes={n_genes}")

        # Update toolbox with new parameters
        toolbox.unregister('gene_gen')
        toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=head_size)
        toolbox.unregister('individual')
        toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=custom_linker)
        toolbox.unregister('population')
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Initialize population
        pop = toolbox.population(n=n_pop)

        # Run GEP
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)

        # Run the GEP algorithm
        pop, log = gep.gep_simple(pop, toolbox, n_generations=50, n_elites=1, stats=stats, hall_of_fame=hof)

        # Evaluate the best model
        best_model = hof[0]
        func = toolbox.compile(best_model)
        train_predictions = np.clip([func(*row) for row in X_train], -1e10, 1e10)
        train_mse = mean_squared_error(y_train, train_predictions)

        # Store results
        grid_search_results.append({
            'head_size': head_size,
            'n_pop': n_pop,
            'n_genes': n_genes,
            'train_mse': train_mse,
            'best_model': best_model
        })

    ending_time = time.time()

    # Find the best combination
    best_result = min(grid_search_results, key=lambda x: x['train_mse'])
    print("Best Parameters:")
    print(f"Head Size: {best_result['head_size']}, Population Size: {best_result['n_pop']}, Number of Genes: {best_result['n_genes']}")
    print(f"Train MSE: {best_result['train_mse']}")
    print(f"Best Model Equation: {str(best_result['best_model'])}")


    # Use the best model for predictions
    best_model = best_result['best_model']
    func = toolbox.compile(best_model)

    # Training and testing predictions
    train_predictions = np.clip([func(*row) for row in X_train], -1e10, 1e10)
    test_predictions = np.clip([func(*row) for row in X_test], -1e10, 1e10)

    # save the GEP results
    train_full_df = pd.DataFrame({
        'Date': time_series_data_training['date'],
        'Actual': time_series_data_training['actual'],
        'Predicted': train_predictions
    })

    test_full_df = pd.DataFrame({
        'Date': time_series_data_testing['date'],
        'Actual': time_series_data_testing['actual'],
        'Predicted': test_predictions
    })

    # Save with additional context
    train_full_df.to_csv(
        f"../forecasting_results/GEP_ensemble_forecasts/y_train_predictions_with_actual_{ensemble_composition}_{date_range}.csv",
        index=False)
    test_full_df.to_csv(
        f"../forecasting_results/GEP_ensemble_forecasts/y_test_predictions_with_actual_{ensemble_composition}_{date_range}.csv",
        index=False)

    # Calculate metrics
    train_mse_best = mean_squared_error(y_train, train_predictions)
    test_mse_best = mean_squared_error(y_test, test_predictions)
    train_mape_best = mean_absolute_percentage_error(y_train, train_predictions)
    test_mape_best = mean_absolute_percentage_error(y_test, test_predictions)
    train_mae_best = mean_absolute_error(y_train, train_predictions)
    test_mae_best = mean_absolute_error(y_test, test_predictions)
    train_r2_best = r2_score(y_train, train_predictions)
    test_r2_best = r2_score(y_test, test_predictions)

    # Define metrics
    metrics = {
        'Metric': ['MSE', 'MAPE', 'MAE', 'R2'],
        'Train': [train_mse_best, train_mape_best, train_mae_best, train_r2_best],
        'Test': [test_mse_best, test_mape_best, test_mae_best, test_r2_best]
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)

    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_evaluation_metrics_{ensemble_composition}_{date_range}.csv"
    metrics_df.to_csv(output_filename, index=False)

    # Plots
    plt.figure(figsize=(12, 6))
    plt.plot(y_train, label="Actual (Training)", color='blue')
    plt.plot(train_predictions, label="Predicted (Training)", color='orange')
    plt.legend()
    plt.title(f"GEP Ensemble - Training: MSE = {train_mse_best:.2f}, R2 = {train_r2_best :.2f}")
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Ensemble Model-Training_Actual_vs_Fitted_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual (Testing)", color='blue')
    plt.plot(test_predictions, label="Predicted (Testing)", color='green')
    plt.legend()
    plt.title(f"GEP Ensemble - Testing: MSE = {test_mse_best:.2f}, R2 = {test_r2_best:.2f}")
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Ensemble Model-Testing_Actual_vs_Forecast_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, train_predictions, alpha=0.7, label='Training Data')
    plt.plot(y_train, y_train, color='red', label='Ideal Fit')
    plt.title(f'GEP Ensemble - Training Set: Actual vs Predicted\nR² = {train_r2_best:.2f}')
    plt.xlabel('Actual Energy (kWh)')
    plt.ylabel('Predicted Energy (kWh)')
    plt.legend()
    plt.grid()
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Ensemble_Model-Training_R2_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_predictions, alpha=0.7, label='Testing Data')
    plt.plot(y_test, y_test, color='red', label='Ideal Fit')
    plt.title(f'GEP Ensemble - Testing Set: Actual vs Predicted\nR² = {test_r2_best:.2f}')
    plt.xlabel('Actual Energy (kWh)')
    plt.ylabel('Predicted Energy (kWh)')
    plt.legend()
    plt.grid()
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Ensemble_Model-Testing_R2_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.show()


    output_text = (
        "Best Parameters:\n"
        f"Head Size: {best_result['head_size']}, Population Size: {best_result['n_pop']}, Number of Genes: {best_result['n_genes']}\n"
        f"Train MSE: {best_result['train_mse']}\n"
        f"Best Model Equation: {str(best_result['best_model'])}\n"
    )

    # Save to file
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_best_model_report{ensemble_composition}_{date_range}.txt"
    with open(output_filename, "w") as file:
        file.write(output_text)

    def gep_predict(X_input):
        # Ensure input is a 2D NumPy array
        X_input = np.asarray(X_input)
        return np.array([func(*row) for row in X_input])

    # Use a sample from the training data as SHAP background
    background = X_train[np.random.choice(X_train.shape[0], size=50, replace=False)]

    # Use KernelExplainer because GEP is a black-box model
    explainer = shap.KernelExplainer(gep_predict, background)

    # Compute SHAP values (slow for large datasets — use a subset if needed)
    shap_values = explainer.shap_values(X_train[:100])  # Use a subset for performance

    # Create summary plot
    shap.summary_plot(shap_values, X_train[:100], feature_names=input_columns, show=False)

    # Save the plot
    output_filename = f"../forecasting_results/GEP_ensemble_forecasts/shap_plot_{ensemble_composition}_{date_range}.png"
    plt.savefig(output_filename)
    plt.close()

    training_time = ending_time-starting_time
    return(training_time)