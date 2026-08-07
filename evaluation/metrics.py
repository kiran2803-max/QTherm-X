import torch

from physics.heat_equation import analytical_solution


# ==========================================
# Mean Absolute Error (MAE)
# ==========================================

def compute_mae(model, X):

    with torch.no_grad():

        prediction = model(X)

        exact = analytical_solution(X)

        mae = torch.mean(torch.abs(prediction - exact))

    return mae.item()


# ==========================================
# Mean Squared Error (MSE)
# ==========================================

def compute_mse(model, X):

    with torch.no_grad():

        prediction = model(X)

        exact = analytical_solution(X)

        mse = torch.mean((prediction - exact) ** 2)

    return mse.item()


# ==========================================
# Relative L2 Error
# ==========================================

def compute_relative_l2_error(model, X):

    with torch.no_grad():

        prediction = model(X)

        exact = analytical_solution(X)

        numerator = torch.norm(prediction - exact)

        denominator = torch.norm(exact)

        relative_l2 = numerator / denominator

    return relative_l2.item()