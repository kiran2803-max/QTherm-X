import numpy as np
import torch


def detect_hotspot(model, device):

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)

    X, T = np.meshgrid(x, t)

    grid = np.column_stack((X.flatten(), T.flatten()))

    grid_tensor = torch.tensor(
        grid,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        prediction = model(grid_tensor)

    temperature = prediction.cpu().numpy()

    index = np.argmax(temperature)

    hotspot_temperature = temperature[index][0]

    hotspot_x = grid[index][0]

    hotspot_t = grid[index][1]

    print("\n==============================")
    print(" HOTSPOT ANALYSIS")
    print("==============================")

    print(f"Maximum Temperature : {hotspot_temperature:.6f}")
    print(f"Position (x)        : {hotspot_x:.3f}")
    print(f"Time (t)            : {hotspot_t:.3f}")

    if hotspot_temperature > 0.5:
        print("\n⚠ WARNING : Potential Hotspot Detected")
    else:
        print("\n✅ Temperature Within Safe Limit")

    return hotspot_temperature