import torch
import math

ALPHA = 1.0


def compute_heat_residual(model, X):

    """
    Computes Heat Equation Residual

    u_t = alpha*u_xx
    """

    u = model(X)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]

    u_t = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(
        outputs=u_x,
        inputs=X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_xx = grad_u_x[:, 0:1]

    residual = u_t - ALPHA * u_xx

    return residual


# ==========================================
# Initial Condition Loss
# ==========================================

def compute_ic_loss(model):

    x = torch.rand(500, 1)

    t = torch.zeros_like(x)

    X_ic = torch.cat([x, t], dim=1)

    prediction = model(X_ic)

    target = torch.sin(math.pi * x)

    return torch.mean((prediction - target) ** 2)


# ==========================================
# Boundary Condition Loss
# ==========================================

def compute_bc_loss(model):

    t = torch.rand(500, 1)

    x_left = torch.zeros_like(t)

    x_right = torch.ones_like(t)

    X_left = torch.cat([x_left, t], dim=1)

    X_right = torch.cat([x_right, t], dim=1)

    left_prediction = model(X_left)

    right_prediction = model(X_right)

    loss_left = torch.mean(left_prediction ** 2)

    loss_right = torch.mean(right_prediction ** 2)

    return loss_left + loss_right
# ==========================================
# Analytical Solution
# ==========================================

def analytical_solution(X):

    """
    Exact solution of the 1D Heat Equation

    u(x,t) = exp(-pi^2*t) * sin(pi*x)
    """

    x = X[:, 0:1]
    t = X[:, 1:2]

    return torch.exp(-(math.pi ** 2) * t) * torch.sin(math.pi * x)
# ==========================================
# Relative L2 Error
# ==========================================

def compute_relative_l2_error(model, X):

    """
    Relative L2 Error

    ||u_pred - u_true|| / ||u_true||
    """

    with torch.no_grad():

        prediction = model(X)

        exact = analytical_solution(X)

        numerator = torch.norm(prediction - exact)

        denominator = torch.norm(exact)

        relative_l2 = numerator / denominator

    return relative_l2.item()