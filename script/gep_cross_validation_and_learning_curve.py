import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import operator
import geppy as gep
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from deap import creator, base, tools

def protected_div(x1, x2):
    return x1 / (x2 + 1e-10)

def protected_log(x):
    x = np.array(x, dtype=float)
    x = np.where(x <= 0, 1e-10, x)
    return np.log(x)

def protected_exp(x):
    return np.exp(np.clip(x, -100, 100))

def square(x):
    x = np.clip(x, -1e5, 1e5)
    return x * x

def power_3(x):
    x = np.clip(x, -1e3, 1e3)
    return x ** 3

def custom_linker(*args):
    return sum(args)


# Load and prepare data
input_filename = "../data/ensemble_data_with_context_2022-09-01_2022-11-30.csv"
ensemble_data = pd.read_csv(input_filename)
date_range = re.search(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", input_filename).group()

X_data = ensemble_data.drop(columns=['actual', 'date']).astype(float).values
y_data = ensemble_data['actual'].values

tscv = TimeSeriesSplit(n_splits=10)

train_scores = []
test_scores = []
train_mse_scores = []
test_mse_scores = []
train_sizes = []

# Create GEP primitive set
input_columns = ensemble_data.drop(columns=['actual', 'date']).columns.tolist()
pset = gep.PrimitiveSet('Main', input_names=input_columns)
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_function(square, 1)
pset.add_function(power_3, 1)
pset.add_function(protected_log, 1)
pset.add_function(protected_exp, 1)
pset.add_constant_terminal(1)
pset.add_constant_terminal(2)
pset.add_constant_terminal(3)

# GEP Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

def custom_linker(*args):
    return sum(args)

# Initialize the toolbox
toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=10)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=10, linker=custom_linker)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)

for train_index, test_index in tscv.split(X_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    def evaluate(individual):
        func = toolbox.compile(individual)
        try:
            preds = np.array([func(*row) for row in X_train])
            preds = np.nan_to_num(preds)
            return ((preds - y_train) ** 2).mean(),
        except:
            return float('inf'),

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.1)
    toolbox.register('mut_invert', gep.invert, pb=0.1)
    toolbox.register('mut_is_ts', gep.is_transpose, pb=0.1)
    toolbox.register('mut_ris_ts', gep.ris_transpose, pb=0.1)
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)

    # GEP training
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    gep.gep_simple(pop, toolbox, n_generations=50, n_elites=1, stats=stats, hall_of_fame=hof)

    best_model = hof[0]
    func = toolbox.compile(best_model)

    y_train_pred = np.array([func(*row) for row in X_train])
    y_test_pred = np.array([func(*row) for row in X_test])

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    train_scores.append(train_r2)
    test_scores.append(test_r2)
    train_sizes.append(len(train_index))
    train_mse_scores.append(mse_train)
    test_mse_scores.append(mse_test)

# Plot Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Train R²', marker='o')
plt.plot(train_sizes, test_scores, label='Test R²', marker='o')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title(f'GEP_Learning_Curve_R_Square({date_range})', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Learning_Curve_R_Square{date_range}.png"
plt.savefig(plot_filename)
plt.show()


# Plot Learning Curve (mse)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mse_scores, label='MSE_training', marker='o')
plt.plot(train_sizes, test_mse_scores, label='MSE_testing', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title(f'GEP_Learning_Curve_mse_{date_range}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"../forecasting_results/GEP_ensemble_forecasts/GEP_Learning_Curve_mse_{date_range}.png"
plt.savefig(plot_filename)
plt.show()

# Save results
learning_curve_df = pd.DataFrame({
    'Train Size': train_sizes,
    'Train R²': train_scores,
    'Test R²': test_scores,
    'Train MSE': mse_train,
    'Test MSE': mse_test
})
learning_curve_df.to_csv(f"../forecasting_results/GEP_ensemble_forecasts/GEP_learning_curve_results_{date_range}.csv", index=False)
