# FBKAN Navier Stokes
Mainly inspired by [Neuromancer, Part 6: PINN for Navier-Stokes Steady-State Cavity Flow](https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/PDEs/Part_6_PINN_NavierStokesCavitySteady_KAN.ipynb)

This package is managed by UV.
```bash
uv sync
```

## Running
```bash
uv run 2d_function_fitting_w_noise.py
```

This problem instead does a 2D navier stokes approximation using randomly selected 
Boundary Points and Centerpoints.
```bash
uv run fbkan_navier_stokes.py
```
