# Multi-agent Game with Jumps
This is the code for three numerical experiments in the paper "Multi-Agent Relative Investment Games in a Jump Diffusion Market with Deep Reinforcement Learning Algorithm". For more information, please refer to the [paper](https://arxiv.org/abs/2404.11967). The code is written in PyTorch.

![](./diagram/Stochastic%20Control.png)

![](./diagram/Parallel%20Compute.png)

It is recommended to use the following environment:
```
Python=3.9.12
torch=1.13.0
matplotlib=3.5.1
numpy=1.22.3
scipy=1.7.3
```
***
# Code Structure
Files:
- `Merton problem.py`: The numerical experiment ''Merton’s problem under a jump-diffusion model''.
- `Linear quadratic regulator.py`: The numerical experiment ''Stochastic linear quadratic regulator problem''.
- `Multi-agent game.py`: The numerical experiment ''The multi-agent portfolio game''.
- `output`: The training and visualization of the model are separated. The results of training are stored in the `output`, and the model is read from the `output` during visualization. If only visualization is required, you can comment out the line of code `model.train_all()`.

Parameters：
- `D`: Dimension.
- `T`: Terminal time.
- `M`: Number of trajectories.
- `N`: Number of intervals $0=t_0<t_1<\cdots<t_N=T$.
- `mu, r, sigma, p, lambda, z, a, b, q, nu, alpha, beta, delta, theta`: Parameters in exponential, power and logarithmic utilities.
- `Xi`: Initial point.
- `terminal`: Terminal points in the first iteration

***
# Citation
```
@article{lu2024multi,
  title={Multi-Agent Relative Investment Games in a Jump Diffusion Market with Deep Reinforcement Learning Algorithm},
  author={Lu, Liwei and Hu, Ruimeng and Yang, Xu and Zhu, Yi},
  journal={arXiv preprint arXiv:2404.11967},
  year={2024}
}
```