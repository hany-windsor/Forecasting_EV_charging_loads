import pandas as pd
from pysr import PySRRegressor
import julia
julia.install()

# Example data
X = pd.DataFrame({
    "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
    "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
})
y = X["x1"]**2 + X["x2"]

# Run symbolic regression
model = PySRRegressor(
    niterations=40,
    unary_operators=["sin", "cos", "exp", "log"],
    binary_operators=["+", "-", "*", "/"],
)

model.fit(X, y)

print(model)
