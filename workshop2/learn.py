"""
Teacher-student perceptron learning, vanilla JAX.

Notes:

* did anyone try the challenge from last week?
* new dependency `pip install plotille` for easy-ish plotting
* workshop 2 demo: stochastic gradient descent
* challenge 2: multi-layer perceptron!
"""

import time

import plotille
import tqdm
import tyro

import jax
import jax.numpy as jnp


# # # 
# Training loop


def main(
    num_steps: int = 200,
    learning_rate: float = 0.01,
    seed: int = 0,
):
    key = jax.random.key(seed)

    # initialise weights
    key, key_init_w = jax.random.split(key)
    w = init_params(key_init_w)

    key, key_init_w_true = jax.random.split(key)
    w_true = init_params(key_init_w_true)
    print(f"true weights: {w_true}")
    print(vis(w=w, w_true=w_true, overwrite=False))

    for t in tqdm.trange(num_steps):
        key, key_sample = jax.random.split(key)
        x = jax.random.normal(key_sample)
        y = forward_pass(w_true, x)

        # compute loss
        loss = loss_fn(w, x, y)
        grad = jax.grad(loss_fn)(w, x, y)

        # update weights
        w = (w[0] - learning_rate * grad[0], w[1] - learning_rate * grad[1])

        # visualise
        figs = vis(x=x, w=w, w_true=w_true, overwrite=True)
        tqdm.tqdm.write(figs)
        tqdm.tqdm.write(f"x: {x:.2f}, y: {y:.2f}, loss: {loss:.2f}")
        time.sleep(0.01)

def init_params(key):
    key_weight, key_bias = jax.random.split(key)
    a = jax.random.normal(key_weight)
    b = jax.random.normal(key_bias)
    return (a, b)


def forward_pass(w, x):
    return w[0] * x + w[1]

def loss_fn(w, x, y):
    return jnp.mean((forward_pass(w, x) - y) ** 2)


# # # 
# Perceptron architecture


# TODO!


# # # 
# Visualisation


def vis(x=None, overwrite=True, **models):
    # configure plot
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-4, 4)
    fig.set_y_limits(-3, 3)
    
    # compute data and add to plot
    xs = jnp.linspace(-4, 4)
    for (label, w), color in zip(models.items(), ['cyan', 'magenta']):
        ys = forward_pass(w, xs)
        fig.plot(xs, ys, label=label, lc=color)
    
    # add a marker for the input batch
    if x is not None:
        fig.text([x], [0], ['x'], lc='yellow')
    
    # render to string
    figure_str = str(fig.show(legend=True))
    reset = f"\x1b[{len(figure_str.splitlines())+1}A" if overwrite else ""
    return reset + figure_str


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
