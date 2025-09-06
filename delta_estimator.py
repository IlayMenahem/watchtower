import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from implementations.bc import train_model
from utils import quantize, get_delta_func
import sympy as sp
from ramanujantools.cmf.known_cmfs import pi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def cartesian_to_spherical(x):
    """
    Convert n-dimensional Cartesian coordinates to spherical coordinates.
    Supports both single points and batches of points.

    Parameters:
    x : tensor, shape (..., n) or (n,)
        Cartesian coordinates [..., x1, x2, ..., xn]

    Returns:
    spherical : tensor, shape (..., n) or (n,)
        Spherical coordinates [..., r, θ1, θ2, ..., θn-1]
        where r is the radius and θi are the angular coordinates

    Note:
    - For n=1: returns [r] where r = |x|
    - For n=2: returns [r, θ] where θ is the angle from x-axis
    - For n>=3: returns [r, θ1, ..., θn-1] following standard spherical coordinates
    """
    x = torch.as_tensor(x, dtype=torch.float32)

    # Handle single point case
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_shape = x.shape[:-1]
    n = x.shape[-1]

    if n == 0:
        result = torch.empty((*batch_shape, 0), device=x.device, dtype=x.dtype)
        return result.squeeze(0) if squeeze_output else result

    # Calculate radius
    r = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))

    if n == 1:
        # For 1D, preserve the sign by returning the original value
        result = x
        return result.squeeze(0) if squeeze_output else result

    # Initialize spherical coordinates tensor
    spherical = torch.zeros((*batch_shape, n), device=x.device, dtype=x.dtype)
    spherical[..., 0] = r.squeeze(-1)

    # Handle case where r is zero
    zero_mask = r.squeeze(-1) < 1e-10

    if n == 2:
        # 2D case: [r, theta]
        spherical[..., 1] = torch.atan2(x[..., 1], x[..., 0])
    else:
        # n-dimensional case (n >= 3)
        # Calculate cumulative sums from the end
        x_squared = x**2
        cum_sum_squares = torch.cumsum(torch.flip(x_squared, [-1]), dim=-1)
        cum_sum_squares = torch.flip(cum_sum_squares, [-1])

        # Calculate angles θ1, θ2, ..., θn-2 (polar angles: 0 to π)
        for i in range(1, n-1):
            sum_squares = cum_sum_squares[..., i-1]
            safe_sum_squares = torch.clamp(sum_squares, min=1e-10)
            cos_val = torch.clamp(x[..., i-1] / torch.sqrt(safe_sum_squares), -1, 1)
            spherical[..., i] = torch.acos(cos_val)

        # Last angle θn-1 (azimuthal angle: -π to π)
        if n > 2:
            spherical[..., n-1] = torch.atan2(x[..., n-1], x[..., n-2])

    # Set zero entries for points with zero radius
    spherical[zero_mask] = 0.0

    return spherical.squeeze(0) if squeeze_output else spherical


def spherical_to_cartesian(spherical):
    """
    Convert n-dimensional spherical coordinates back to Cartesian.
    Supports both single points and batches of points.

    Parameters:
    spherical : tensor, shape (..., n) or (n,)
        Spherical coordinates [..., r, θ1, θ2, ..., θn-1]

    Returns:
    x : tensor, shape (..., n) or (n,)
        Cartesian coordinates [..., x1, x2, ..., xn]

    Note:
    - For n=1: input [r] returns [r] (or [-r] if r < 0)
    - For n=2: input [r, θ] returns [r*cos(θ), r*sin(θ)]
    - For n>=3: follows standard spherical coordinate conversion
    """
    spherical = torch.as_tensor(spherical, dtype=torch.float32)

    # Handle single point case
    if spherical.dim() == 1:
        spherical = spherical.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_shape = spherical.shape[:-1]
    n = spherical.shape[-1]

    if n == 0:
        result = torch.empty((*batch_shape, 0), device=spherical.device, dtype=spherical.dtype)
        return result.squeeze(0) if squeeze_output else result

    if n == 1:
        # For 1D, the spherical coordinate is just the signed distance
        result = spherical
        return result.squeeze(0) if squeeze_output else result

    r = spherical[..., 0:1]  # Keep dimension for broadcasting
    x = torch.zeros((*batch_shape, n), device=spherical.device, dtype=spherical.dtype)

    if n == 2:
        # 2D case
        theta = spherical[..., 1]
        x[..., 0] = (r * torch.cos(theta)).squeeze(-1)
        x[..., 1] = (r * torch.sin(theta)).squeeze(-1)
    else:
        # n-dimensional case (n >= 3)
        # Pre-compute sin and cos values
        angles = spherical[..., 1:]  # All angles
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)

        # Calculate cumulative products of sines
        sin_products = torch.ones((*batch_shape, n), device=spherical.device, dtype=spherical.dtype)

        for i in range(n):
            if i == 0:
                # x1 = r * cos(θ1)
                x[..., 0] = (r * cos_vals[..., 0]).squeeze(-1) if n > 1 else r.squeeze(-1)
            elif i < n - 1:
                # xi = r * sin(θ1) * sin(θ2) * ... * sin(θi-1) * cos(θi)
                sin_products[..., i] = torch.prod(sin_vals[..., :i], dim=-1)
                x[..., i] = (r.squeeze(-1) * sin_products[..., i] * cos_vals[..., i])
            else:
                # xn = r * sin(θ1) * sin(θ2) * ... * sin(θn-2) * sin(θn-1)
                sin_products[..., i] = torch.prod(sin_vals[..., :i], dim=-1)
                x[..., i] = r.squeeze(-1) * sin_products[..., i]

    return x.squeeze(0) if squeeze_output else x


class deltaDataset(Dataset):
    def __init__(self, path: str):
        import pandas as pd
        # load csv file
        self.data = torch.tensor(pd.read_csv(path).values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][:-1]
        delta = self.data[idx][-1]

        return state, delta


def find_best_traj(model: nn.Module, num_steps: int):
    '''
    find the best trajectory using the trained model

    Args:
    - model (nn.Module): The trained model to use for predictions.
    - num_steps (int): The number of steps to take in the search.

    Returns:
    - best_trajectory (tuple[float,...]): The direction of the best trajectory found.
    - best_delta (float): The delta value of the best trajectory found.
    - trajectory_history (list[tuple]): A list of all trajectories evaluated during the search, and their deltas.
    '''
    # Initialize as a 2D direction vector (will be normalized)
    initial_direction = torch.tensor([1.0, 0.5], requires_grad=True)
    trajectory_history = []
    optimizer = torch.optim.Adam([initial_direction], lr=0.01, maximize=True)

    def delta_prediction(trajectory: torch.Tensor):
        normalized_traj = trajectory / torch.norm(trajectory)
        return model(normalized_traj)

    for step in range(num_steps):
        optimizer.zero_grad()
        predicted_delta = delta_prediction(initial_direction)
        predicted_delta.backward()
        optimizer.step()

        # Store normalized trajectory for history
        normalized_traj = initial_direction.detach() / torch.norm(initial_direction.detach())
        trajectory_history.append((normalized_traj.clone(), predicted_delta.item()))

    best_trajectory, best_delta = max(trajectory_history, key=lambda x: x[1])

    return best_trajectory, best_delta, trajectory_history


if __name__ == "__main__":
    class DeltaEstimator(nn.Module):
        def __init__(self, state_dim: int, num_layers: int = 16, hidden_dim: int = 256):
            super().__init__()

            self.linear_start = nn.Linear(state_dim-1, hidden_dim)  # -1 to remove radius
            self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-2)])
            self.linear_end = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # Handle batch of points
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension if missing

            x = cartesian_to_spherical(x)
            x = x[:, 1:]  # Remove radius, keep only angles (theta)

            x = F.relu(self.linear_start(x))
            for layer in self.linear_layers:
                x = F.relu(layer(x))
            x = self.linear_end(x)

            return x

    model = DeltaEstimator(2, 16, 256)

    num_epochs = 400
    batch_size = 256

    # Load full dataset and split into train and validation
    full_dataset = deltaDataset('states_deltas.csv')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn(model, x, y):
        y = y.unsqueeze(1)
        y_pred = model(x)

        return F.mse_loss(y_pred, y)

    def accuracy_fn(model, x, y):
        y_pred = model(x)
        y = y.unsqueeze(1)
        mae = F.l1_loss(y_pred, y)

        return mae

    model_path = 'delta_estimator_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        losses_train, losses_val = [], []
    else:
        model, losses_train, losses_val = train_model(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, accuracy_fn)
        torch.save(model.state_dict(), 'delta_estimator_model.pth')

    # Load the actual data for interpolation fitting
    data_for_spline = pd.read_csv('states_deltas.csv')
    states_for_spline = torch.tensor(data_for_spline.iloc[:, :-1].values, dtype=torch.float32)
    deltas_for_spline = torch.tensor(data_for_spline.iloc[:, -1].values, dtype=torch.float32)

    # Convert to spherical coordinates and extract angles
    spherical_coords = cartesian_to_spherical(states_for_spline)
    angles_for_spline = spherical_coords[:, 1].numpy()  # theta component

    # Sort by angle for proper interpolation fitting
    sorted_indices = np.argsort(angles_for_spline)
    sorted_angles = angles_for_spline[sorted_indices]
    sorted_deltas = deltas_for_spline[sorted_indices].numpy()

    # graph the neural network estimations by the angle, and compare against the actual deltas in states_deltas.csv
    # Load the actual data for comparison
    data = pd.read_csv('states_deltas.csv')
    actual_states = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    actual_deltas = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    # Get training and validation indices to separate the data points
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # Convert states to angles for plotting
    actual_spherical = cartesian_to_spherical(actual_states)
    actual_angles = actual_spherical[:, 1]  # Get theta (angle) component

    # Split data into training and validation for plotting
    train_angles = actual_angles[train_indices]
    train_deltas = actual_deltas[train_indices]
    val_angles = actual_angles[val_indices]
    val_deltas = actual_deltas[val_indices]

    # Generate a range of angles for neural network predictions
    test_angles = torch.linspace(0, np.pi/2, 300)
    test_states = torch.stack([torch.cos(test_angles), torch.sin(test_angles)], dim=1)

    # Get neural network predictions
    model.eval()
    with torch.no_grad():
        nn_predictions = model(test_states).squeeze()

    # Get predictions from all interpolation methods
    test_angles_np = test_angles.numpy()

    # Create the comparison plot
    plt.figure(figsize=(14, 9))

    # Plot training data points
    plt.scatter(train_angles.numpy(), train_deltas.numpy(),
                alpha=0.6, c='blue', s=20, label='Training Data', zorder=2)

    # Plot validation data points
    plt.scatter(val_angles.numpy(), val_deltas.numpy(),
                alpha=0.6, c='green', s=20, label='Validation Data', zorder=2)

    # Plot neural network predictions
    plt.plot(test_angles_np, nn_predictions.numpy(),
                'r-', linewidth=2, label='Neural Network Predictions', zorder=3)

    plt.xlabel('Angle (radians)')
    plt.ylabel('Delta')
    plt.title('Neural Network Predictions vs Training/Validation Data by Angle')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.pi/2)

    plt.tight_layout()
    plt.show()

    # Calculate R² score for the neural network predictions on actual data
    model.eval()
    with torch.no_grad():
        actual_predictions = model(actual_states).squeeze()

    # find the best trajectory using the trained model
    best_traj, best_delta, traj_history = find_best_traj(model, 1000)

    print(f"Best trajectory direction: {best_traj.numpy()}")
    print(f"Predicted delta for best trajectory: {best_delta:.6f}")

    quantized_traj = quantize(best_traj[0].item(), best_traj[1].item(), 100)
    print(f"Quantized trajectory: {quantized_traj}")

    cmf = pi()
    constant = 'pi'

    x, y = sp.symbols('x, y')
    variables = (x, y)
    starting_point = {x: 1, y: 1}

    cost_fn = get_delta_func(cmf, starting_point, constant, 5000, variables)
    delta = cost_fn(quantized_traj)
    print(f"Actual delta for quantized trajectory: {delta:.6f}")
