import numpy as np
import time
import torch
import psutil
import json
import os
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm


class Block(nn.Module):

    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(Block, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(torch.sin(self.L1(x)))) + x
        else:
            a = self.act(self.L1(self.act(self.L2(x)))) + x
        return a


class Network(nn.Module):

    def __init__(self, params, penalty=None):
        super(Network, self).__init__()
        self.params = params
        self.first = nn.Linear(self.params["inputs"], self.params["width"])
        self.last = nn.Linear(self.params["width"], self.params["output"])
        self.network = nn.Sequential(*[
            self.first,
            *[Block(self.params["width"], self.params["params_act"], self.params["activation"])] * self.params["depth"],
            self.last
        ])
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        if self.penalty is not None:
            if self.penalty == "Sigmoid":
                sigmoid = nn.Sigmoid()
                return sigmoid(self.network(x)) * self.bound
            elif self.penalty == "Tanh":
                tanh = nn.Tanh()
                return tanh(self.network(x)) * self.bound
            else:
                raise RuntimeError("Other penalty has not bee implemented!")
        else:
            return self.network(x)


class FBSNN(nn.Module): # Forward-Backward Stochastic Neural Network
    def __init__(self, net_value, net_control, params):
        super().__init__()
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        self.params_net = params["net"]
        self.device = self.params_train["device"]
        self.Xi = self.params_equ["Xi"]    # initial point
        self.T = self.params_equ["T"]              # terminal time
        self.M = self.params_equ["M"]             # number of trajectories
        self.N = self.params_equ["N"]              # number of time snapshots
        self.D = self.params_equ["D"]              # number of players
        manager = mp.Manager()
        self.dict = manager.dict()
        for i in range(self.D):
            self.dict[f'net_critic_{i}'] = net_value[i].to(self.device)
            self.dict[f'net_actor_{i}'] = net_control[i].to(self.device)
            self.dict[f'optimizer_critic_{i}'] = optim.Adam(self.dict[f'net_critic_{i}'].parameters(), lr=self.params_train["lr_critic"])
            self.dict[f'optimizer_actor_{i}'] = optim.Adam(self.dict[f'net_actor_{i}'].parameters(), lr=self.params_train["lr_actor"])
            self.dict[f'scheduler_critic_{i}'] = optim.lr_scheduler.MultiStepLR(self.dict[f'optimizer_critic_{i}'],
                milestones=self.params_train["milestones_critic"], gamma=self.params_train["gamma_critic"])
            self.dict[f'scheduler_actor_{i}'] = optim.lr_scheduler.MultiStepLR(self.dict[f'optimizer_actor_{i}'],
                milestones=self.params_train["milestones_actor"], gamma=self.params_train["gamma_actor"])
        self.terminal = self.params_equ["terminal"].to(self.device)  # terminal points in the first iteration
        self.Y_error_list, self.control_error_list, self.training_cost_list = [], [], []
        self.dt = self.T / self.N

    def compute_exact(self):
        D = self.D
        mu, nu, sigma = self.params_equ["mu"], self.params_equ["nu"], self.params_equ["sigma"]
        alpha, beta = self.params_equ["alpha"], self.params_equ["beta"]
        lam, lam0 = self.params_equ["lambda"], self.params_equ["lambda0"]
        delta, theta = self.params_equ["delta"], self.params_equ["theta"]
        def optimal_control(pi):
            hat_pisigma = [(sum([a * b for a, b in zip(pi, sigma)]) - pi[i] * sigma[i]) / D for i in range(D)]
            hat_pibeta = [(sum([a * b for a, b in zip(pi, beta)]) - pi[i] * beta[i]) / D for i in range(D)]
            no_jump = [mu[i] + sigma[i]*hat_pisigma[i]*theta[i]/delta[i] - pi[i]*(nu[i]**2+sigma[i]**2)*(1-theta[i]/D)/delta[i] for i in range(D)]
            jump = [-lam0 * beta[i] - lam[i] * alpha[i] + lam0 * beta[i] * np.exp(
                -(1 - theta[i] / D) * pi[i] * beta[i] / delta[i] + theta[i] * hat_pibeta[i] / delta[i]) + lam[i] *
                    alpha[i] * np.exp(-(1 - theta[i] / D) * pi[i] * alpha[i] / delta[i]) for i in range(D)]
            return [no_jump[i] + jump[i] for i in range(D)]
        self.control_exact = fsolve(optimal_control, [1.0 for _ in range(D)])
        A = self.control_exact
        hat_pimu = [(sum([a * b for a, b in zip(list(A), mu)]) - A[i] * mu[i]) / D for i in range(D)]
        hat_pisigma = [(sum([a * b for a, b in zip(list(A), sigma)]) - A[i] * sigma[i]) / D for i in range(D)]
        hat_pibeta = [(sum([a * b for a, b in zip(list(A), beta)]) - A[i] * beta[i]) / D for i in range(D)]
        hat_pinu2 = [(sum([a**2 * b**2 for a, b in zip(list(A), nu)]) - A[i]**2 * nu[i]**2) / D for i in range(D)]
        K1 = [-(1 - theta[i] / D) * A[i] * mu[i] / delta[i] + hat_pimu[i] * theta[i] / delta[i] + 0.5 * (
                    nu[i] ** 2 + sigma[i] ** 2) * A[i] ** 2 * (1 - theta[i] / D) ** 2 / delta[i] ** 2 + 0.5 * (
                          hat_pisigma[i] ** 2 + hat_pinu2[i] / self.D) * theta[i] ** 2 / delta[i] ** 2 - A[i] * sigma[
                  i] * hat_pisigma[i] * theta[i] * (1 - theta[i] / D) / delta[i] ** 2 for i in range(D)]
        K2 = [lam0 * (np.exp(-(1 - theta[i] / D) * A[i] * beta[i] / delta[i] + theta[i] * hat_pibeta[i] / delta[i]) - 1 +
                    A[i] * beta[i] * (1 - theta[i] / D) / delta[i] - hat_pibeta[i] * theta[i] / delta[i]) for i in range(D)]
        K3 = [lam[i]*(np.exp(-(1-theta[i]/D)*A[i]*alpha[i]/delta[i])-1+A[i]*alpha[i]*(1-theta[i]/D)/delta[i]) for i in range(D)]
        K4 = [sum([lam[k] * (
                    np.exp(theta[i] * A[k] * alpha[k] / delta[i] / D) - 1 - A[k] * alpha[k] / D * theta[i] / delta[i])
                   for k in range(D) if k != i]) for i in range(D)]
        self.K = [K1[i] + K2[i] + K3[i] + K4[i] for i in range(D)]
        print("Exact control:", self.control_exact)
        print("Exact Y0:", self.exact_value(torch.zeros(self.M, self.N + 1, 1, device=self.device),
            torch.cat([self.Xi.to(self.device)] * (self.M * (self.N + 1)), dim=0).view(self.M, self.N + 1, self.D))[0, 0, :].cpu().numpy())
        print("K:", self.K)

    def exact_value(self, t, x):
        K_list = [torch.tensor(self.K, device=self.device) for _ in range(self.M * (self.N + 1))]
        delta_list = [torch.tensor(self.params_equ["delta"], device=self.device) for _ in range(self.M * (self.N+1))]
        theta_list = [torch.tensor(self.params_equ["theta"], device=self.device) for _ in range(self.M * (self.N+1))]
        K = torch.stack(K_list, dim=0).view(self.M, self.N + 1, self.D)
        delta = torch.stack(delta_list, dim=0).view(self.M, self.N+1, self.D)
        theta = torch.stack(theta_list, dim=0).view(self.M, self.N+1, self.D)
        T = torch.exp((self.T - t) * K)
        mean = torch.mean(x, dim=-1, keepdim=True)
        X = -torch.exp(-(x-theta*mean)/delta)
        return T * X

    def existence_and_uniqueness(self):
        mu, nu, sigma, alpha = self.params_equ["mu"], self.params_equ["nu"], self.params_equ["sigma"], self.params_equ[
            "alpha"]
        beta, delta, theta = self.params_equ["beta"], self.params_equ["delta"], self.params_equ["theta"],
        lam, lam0 = self.params_equ["lambda"], params_equ["lambda0"]
        control_bound = 1
        condition1 = (max(nu) ** 2 + max(sigma) ** 2 + lam0 * max(beta) ** 2 * np.exp(
            2 * control_bound * max(beta) / min(delta)) + max(lam) * max(alpha) ** 2 * np.exp(
            control_bound * max(alpha) / min(delta))) / min(delta)
        condition2 = min(nu) ** 2 - max(sigma) * (max(sigma) - min(sigma)) - lam0 * max(beta) ** 2 * np.exp(
            2 * control_bound * max(beta) / min(delta))
        print("Existence and uniqueness condition in the paper:",
              "Satisfy" if condition1 < 1 and condition2 > 0 else "Dissatisfy")

    def g(self, X, k):
        mean = torch.mean(X, dim=1, keepdim=True)
        return -torch.exp(-(X[:, k].unsqueeze(-1)-self.params_equ["theta"][k]*mean)/self.params_equ["delta"][k])

    def mu(self, t, X, control):
        return control * torch.tensor(self.params_equ["mu"], device=self.device).unsqueeze(0)
        
    def sigma_i(self, t, X, control):
        return control * torch.tensor(self.params_equ["nu"], device=self.device).unsqueeze(0)

    def sigma_0(self, t, X, control):
        return control * torch.tensor(self.params_equ["sigma"], device=self.device).unsqueeze(0)

    def xi_i(self, t, X, control):
        return control * torch.tensor(self.params_equ["alpha"], device=self.device).unsqueeze(0)

    def xi_0(self, t, X, control):
        return control * torch.tensor(self.params_equ["beta"], device=self.device).unsqueeze(0)

    def net_u_Du(self, net_critic, t, X, k, compute_Du=True, compute_all_critic=True):
        if compute_Du:
            u = net_critic[k](torch.cat([t, X], dim=1))
            Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True, create_graph=True)[0]
            return u, Du
        elif compute_all_critic:
            u_list = [net_critic[i](torch.cat([t, X], dim=1)) for i in range(self.D)]
            return torch.cat(u_list, dim=1), 0
        else:
            return net_critic[k](torch.cat([t, X], dim=1)), 0

    def get_memory(self, type):
        process = psutil.Process()
        mem_info = process.memory_info()
        if type == "MB":
            print("Memory of physical: %.1f MB, virtual: %.1f MB, allocated: %.1f MB" % (
                mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
        if type == "GB":
            print("Memory of physical: %.3f GB, virtual: %.3f GB, allocated: %.3f GB" % (
                mem_info.rss / 1024 / 1024 / 1024, mem_info.vms / 1024 / 1024 / 1024,
                torch.cuda.memory_allocated() / 1024 / 1024 / 1024))

    def fetch_minibatch(self):
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D))
        DB = np.zeros((self.M, self.N + 1, 1))
        Dt[:, 1:, :] = self.dt
        DW[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, self.D))
        DB[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, 1))
        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        B = np.cumsum(DB, axis=1)
        P = np.cumsum(np.random.exponential(scale=[1 / self.params_equ["lambda"][i] for i in range(self.D)],
                                            size=(self.M, 30, self.D)), axis=1)
        P[np.where(P > self.T)] = 0
        count = np.count_nonzero(P, axis=1).max()
        P = P[:, 0:count, :]
        P0 = np.cumsum(np.random.exponential(scale=1 / self.params_equ["lambda0"], size=(self.M, 30)), axis=1)
        P0[np.where(P0 > self.T)] = 0
        count0 = np.count_nonzero(P0, axis=1).max()
        P0 = P0[:, 0:count0]
        self.jump_index = torch.unique(torch.nonzero(torch.tensor(P))[:, 0])
        self.jump_index0 = torch.unique(torch.nonzero(torch.tensor(P0))[:, 0])
        return torch.from_numpy(t).float().to(self.device), torch.from_numpy(W).float().to(
            self.device), torch.from_numpy(B).repeat(1, 1, self.D).float().to(self.device), torch.from_numpy(
            P).float().to(self.device), torch.from_numpy(P0).float().unsqueeze(-1).to(self.device)

    def plot_list(self, list, ylabel, legend, color=None, title=None, ylabel_log=False, xticks=None, baseline=None, filename=None):
        plt.figure(dpi=300)
        for i in range(len(list)):
            if color is not None:
                plt.plot(list[i], label=legend[i], color=color[i])
            else:
                plt.plot(list[i], label=legend[i])
        if baseline is not None:
            plt.plot(torch.ones(self.params_train["iteration"]) * baseline, '--', color=color[-1])
        if len(list) > 1:
            plt.legend()
        plt.xlabel("iteration")
        plt.ylabel(ylabel)
        if xticks is not None:
            plt.xticks(xticks[0], xticks[1])
        if title is not None:
            plt.title(title)
        if ylabel_log:
            plt.gca().set_yscale("log")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(os.path.join(self.output_folder, '%s.png' % filename), dpi=300, bbox_inches='tight')
        plt.show()

    def loss_function(self, net_critic, net_actor, t, W, B, P, P0, X0, i, n, k, traj_only=False, X_only=False, control_exact=False):
        loss1 = torch.zeros(1, device=self.device)
        X_buffer, Y_buffer = [], []
        t0, W0, B0 = t[:, i, :], W[:, i, :], B[:, i, :]
        X0.requires_grad = True
        if not X_only:
            Y0, Z0 = self.net_u_Du(net_critic, t0, X0, k, compute_Du=not traj_only)
        reward = 0
        if i == 0:
            X_buffer.append(X0)
            if not X_only:
                Y_buffer.append(Y0)
        for j in range(0, n):
            t1, W1, B1 = t[:, i+j+1, :], W[:, i+j+1, :], B[:, i+j+1, :]
            P1 = torch.where(P < t1[0, 0], P, torch.zeros_like(P, dtype=torch.float))
            P2 = torch.where(P1 > t0[0, 0], P1, torch.zeros_like(P1, dtype=torch.float))
            P3 = torch.where(torch.sum(P2, dim=1) != 0, 1, 0)   # M x D
            P01 = torch.where(P0 < t1[0, 0], P0, torch.zeros_like(P0, dtype=torch.float))
            P02 = torch.where(P01 > t0[0, 0], P01, torch.zeros_like(P01, dtype=torch.float))
            P03 = torch.where(torch.sum(P02, dim=1) != 0, 1, 0)  # M x 1
            Mi = P3 - torch.tensor(self.params_equ["lambda"], device=self.device).unsqueeze(0)*self.dt
            M0 = P03 - self.params_equ["lambda0"] * self.dt     # M x 1
            control = torch.cat([net_actor[d](torch.cat([t0, X0], dim=1)) for d in range(self.D)], dim=1)
            if control_exact:
                control_list = [torch.tensor(self.control_exact, device=self.device) for _ in range(self.M)]
                control = torch.stack(control_list, dim=0).view(self.M, self.D)
                control = control.to(X0.dtype)
            X1 = X0 + self.mu(t0, X0, control) * self.dt + self.sigma_i(t0, X0, control) * (W1 - W0) + self.sigma_0(
                t0, X0, control) * (B1 - B0) + self.xi_i(t0, X0, control) * Mi + self.xi_0(t0, X0, control) * M0
            if not X_only:
                Y1, Z1 = self.net_u_Du(net_critic, t1, X1, k, compute_Du=not traj_only)
            if not traj_only:
                reward = reward - torch.sum(Z0*(self.sigma_i(t0, X0, control) * (W1 - W0)
                                                + self.sigma_0(t0, X0, control) * (B1 - B0)), dim=1, keepdim=True)
                nonlocal_term0, _ = self.net_u_Du(net_critic, t0, X0 + self.xi_0(t0, X0, control), k, compute_Du=False, compute_all_critic=False)
                reward = reward - (nonlocal_term0 - Y0) * M0
                for j in range(self.D):
                    newX = torch.zeros_like(control)
                    newX[:, j] = self.xi_0(t0, X0, control)[:, j]
                    nonlocal_term, _ = self.net_u_Du(net_critic, t0, X0 + newX, k, compute_Du=False, compute_all_critic=False)
                    reward = reward - (nonlocal_term - Y0) * Mi[:, j].unsqueeze(1)
                loss1 = loss1 + torch.mean((Y0 - reward - Y1)**2)
            if not X_only:
                t0, W0, B0, X0, Y0, Z0 = t1, W1, B1, X1, Y1, Z1
                Y_buffer.append(Y0)
            else:
                t0, W0, B0, X0 = t1, W1, B1, X1
            X_buffer.append(X0)

        if not traj_only:
            if (i + n) == self.N:
                self.dict[f'XT_{k}'] = X1.detach()
            YT, Z1 = self.net_u_Du(net_critic, torch.ones_like(Y0) * self.T, self.dict[f'XT_{k}'], k,
                                   compute_Du=False, compute_all_critic=False)
            loss2 = torch.mean((YT - self.g(self.dict[f'XT_{k}'], k))**2) / self.N * n
        loss_critic = loss1 + loss2 if not traj_only else 0
        X = torch.stack(X_buffer, dim=1)
        if not X_only:
            Y = torch.stack(Y_buffer, dim=1)
        loss_actor = -torch.mean(self.g(X[:, -1, :], k))
        return loss_critic, loss_actor, X, Y if not X_only else 0

    def Critic_step(self, k, net_critic, net_actor, optimizer_critic):
        n = self.params_train["critic_step"]
        t, W, B, P, P0 = self.fetch_minibatch()
        for i in range(int(self.N / n)):
            X0 = torch.cat([self.Xi.to(self.device)] * self.M) if i == 0 else X1.detach()
            loss_critic, _, X_pred, _ = self.loss_function(net_critic, net_actor, t, W, B, P, P0, X0, i=i * n, n=n, k=k)
            X1 = X_pred[:, -1, :]
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
        return loss_critic

    def Actor_step(self, k, net_critic, net_actor, optimizer_actor):
        n = self.params_train["critic_step"]
        t, W, B, P, P0 = self.fetch_minibatch()
        X0 = torch.cat([self.Xi.to(self.device)] * self.M)
        _, loss_actor, _, _ = self.loss_function(net_critic, net_actor, t, W, B, P, P0, X0, i=0, n=self.N, k=k, traj_only=True, X_only=True)
        optimizer_actor.zero_grad()
        loss_actor.backward()
        for _ in range(n):
            optimizer_actor.step()
        return loss_actor

    def compute_error(self):
        net_critic = [self.dict[f'net_critic_{k}'] for k in range(D)]
        net_actor = [self.dict[f'net_actor_{k}'] for k in range(D)]
        t, W, B, P, P0 = self.fetch_minibatch()
        X0 = torch.cat([self.Xi.to(self.device)] * self.M)
        _, _, X_pred, Y_pred = self.loss_function(net_critic, net_actor, t, W, B, P, P0, X0, i=0, n=self.N, k=0, traj_only=True)
        Y_exact = self.exact_value(t, X_pred)
        Y_error = torch.mean(
            torch.sum(torch.sqrt(torch.mean((Y_pred - Y_exact) ** 2, dim=0) / torch.mean(Y_exact ** 2, dim=0))[0:-1, :],
                      dim=0) * self.dt)
        control_list = [torch.tensor(self.control_exact, device=self.device) for _ in range(self.M * (self.N + 1))]
        control_exact = torch.stack(control_list, dim=0).view(self.M, self.N + 1, self.D)
        control_pred = torch.cat([self.dict[f'net_actor_{i}'](torch.cat([t, X_pred], dim=2)) for i in range(self.D)], dim=2)
        control_error = torch.mean(torch.sum(
            torch.sqrt(torch.mean((control_pred - control_exact) ** 2, dim=0) / torch.mean(control_exact ** 2, dim=0))[
            0:-1, :], dim=0) * self.dt)
        return Y_error.squeeze(), control_error.squeeze()

    def train_players(self, it, k):
        if it == 0:
            self.dict[f'XT_{k}'] = torch.cat([self.terminal.to(self.device)] * self.M)
        net_critic = [Network(self.params_net) for _ in range(self.D)]
        net_actor = [Network(self.params_net, penalty=self.params_net["penalty"]) for _ in range(self.D)]
        for d in range(self.D):
            net_critic[d].load_state_dict(self.dict[f'net_critic_{d}'].state_dict())
            net_actor[d].load_state_dict(self.dict[f'net_actor_{d}'].state_dict())
        if it < self.params_train["iteration"] * 0.6:
            lr_critic, lr_actor = self.params_train["lr_critic"], self.params_train["lr_actor"]
        elif it < self.params_train["iteration"] * 0.8:
            lr_critic = self.params_train["lr_critic"]*self.params_train["gamma_critic"]
            lr_actor = self.params_train["lr_actor"]*self.params_train["gamma_actor"]
        else:
            lr_critic = self.params_train["lr_critic"] * self.params_train["gamma_critic"]**2
            lr_actor = self.params_train["lr_actor"] * self.params_train["gamma_actor"]**2
        optimizer_critic = optim.Adam(net_critic[k].parameters(), lr=lr_critic)
        optimizer_actor = optim.Adam(net_actor[k].parameters(), lr=lr_actor)
        for j in range(self.params_train["epochs"]):
            if j % 50 == 0:
                print("         it:", it, "player:", k, "epochs:", j, "time: %.2f min" % ((time.time() - self.start_time) / 60))
            loss_critic = self.Critic_step(k, net_critic, net_actor, optimizer_critic)
            loss_actor = self.Actor_step(k, net_critic, net_actor, optimizer_actor)
            self.dict[f"loss_critic_{k}"] = loss_critic.detach()
            self.dict[f"loss_actor_{k}"] = loss_actor.detach()
        self.dict[f'net_critic_{k}'].load_state_dict(net_critic[k].state_dict())
        self.dict[f'net_actor_{k}'].load_state_dict(net_actor[k].state_dict())

    def train_all(self):
        self.start_time = time.time()
        # self.save_and_read_model("read")
        self.existence_and_uniqueness()
        self.compute_exact()
        for it in range(self.params_train["iteration"]):
            processes = []
            for k in range(self.D):
                p = mp.Process(target=self.train_players, args=(it, k))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            with torch.no_grad():
                Y_error, control_error = self.compute_error()
            self.Y_error_list.append(Y_error.detach())
            self.control_error_list.append(control_error.detach())
            self.training_cost_list.append(torch.tensor(time.time() - self.start_time) / 60)
            if it % 1 == 0:
                elapsed = time.time() - self.start_time
                print(
                    'It: %d, Time: %.2f min, Y: %.3f%%, control: %.3f%%, loss_critic: %.3e, %.3e, loss_actor: %.4f, %.4f' % (
                        it, elapsed / 60, Y_error.cpu().detach().numpy() * 100,
                        control_error.cpu().detach().numpy() * 100, self.dict["loss_critic_0"].item(),
                        self.dict["loss_critic_1"].item(), self.dict["loss_actor_0"].item(), self.dict["loss_actor_1"].item()))
            if it % 50 == 0:
                self.get_memory("GB")
        print("The training is over!")
        self.save_and_read_model(type="save")

    def save_and_read_model(self, type="save"):
        filename = os.path.basename(os.path.abspath(__file__))
        self.output_folder = os.path.join('output', filename)
        name = ["Y_error.json", "control_error.json", "training_cost.json"]
        save_list = [self.Y_error_list, self.control_error_list, self.training_cost_list]
        if type == "save":
            os.makedirs(self.output_folder, exist_ok=True)
            for k in range(self.D):
                torch.save(self.dict[f'net_critic_{k}'].state_dict(), os.path.join(self.output_folder, f'critic_{k}.pth'))
                torch.save(self.dict[f'net_actor_{k}'].state_dict(), os.path.join(self.output_folder, f'actor_{k}.pth'))
            for i in range(len(name)):
                with open(os.path.join(self.output_folder, name[i]), 'w') as file:
                    json.dump([tensor.tolist() for tensor in save_list[i]], file)
        elif type == "read":
            for k in range(self.D):
                self.dict[f'net_critic_{k}'].load_state_dict(torch.load(os.path.join(self.output_folder, f'critic_{k}.pth')))
                self.dict[f'net_actor_{k}'].load_state_dict(torch.load(os.path.join(self.output_folder, f'actor_{k}.pth')))
            for i in range(len(name)):
                with open(os.path.join(self.output_folder, name[i]), 'r') as file:
                    save_list[i][:] = [torch.tensor(j) for j in json.load(file)]
            print(f"The training costs: %.2f min" % self.training_cost_list[-1])
        else:
            raise RuntimeError("You can only save or read the model!")

    def predict(self, t, W, B, P, P0, X0):
        self.compute_exact()
        self.save_and_read_model(type="read")
        print("Y error: %.3f%%, control error: %.3f%%" % (self.Y_error_list[-1].numpy() * 100, self.control_error_list[-1].numpy() * 100))
        self.plot_list([self.Y_error_list, self.control_error_list], "Error_value/Error_control", ["value function", "control"],
                       color=["#E7483D", "#6565AE", "#8FC751"], title=None, ylabel_log=True, baseline=0.01, filename="game_train")
        net_critic = [self.dict[f'net_critic_{d}'] for d in range(D)]
        net_actor = [self.dict[f'net_actor_{d}'] for d in range(D)]
        _, _, X_pred, Y_pred = self.loss_function(net_critic, net_actor, t, W, B, P, P0, X0, i=0, n=self.N, k=0, traj_only=True)
        _, _, X_test, _ = self.loss_function(net_critic, net_actor, t, W, B, P, P0, X0, i=0, n=self.N, k=0, traj_only=True, control_exact=True)
        control_pred = torch.cat([self.dict[f'net_actor_{i}'](torch.cat([t, X_pred], dim=2)) for i in range(self.D)], dim=2)
        return X_pred, Y_pred, control_pred, X_test


if __name__ == '__main__':
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    device = torch.device('cpu')
    print("device:", device)
    parameters = {"axes.labelsize": 20, "axes.titlesize": 20,
                  "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}
    plt.rcParams.update(parameters)

    D = 20
    params = {
        "equation": {
            "mu": [0.05]+[0.04]*(D-1), "nu": [0.4]+[0.3]*(D-1), "sigma": [0.35]+[0.25]*(D-1), "alpha": [0.3]+[0.2]*(D-1),
            "beta": [0.25]+[0.15]*(D-1), "delta": [2]+[1]*(D-1), "theta": [0.8]+[0.7]*(D-1), "lambda": [0.3]+[0.2]*(D-1),
            "lambda0": 0.25, "D": D, "T": 1, "M": 500, "N": 50,
            "Xi": torch.tensor([10.]+[9.]*(D-1)).unsqueeze(0), "terminal": torch.ones(1, D)
        },
        "train": {
            "lr_critic": 1e-3, "gamma_critic": 0.1, "milestones_critic": [150000, 200000],
            "lr_actor": 1e-3, "gamma_actor": 0.1, "milestones_actor": [30000, 40000],
            "iteration": 100, "epochs": 100, "actor_step": 10, "critic_step": 1, "device": device,
        },
        "net": {
            "inputs": D+1, "width": D+10, "depth": 2, "output": 1, "activation": "Tanh", "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
        }
    }
    params_equ, params_train, params_net = params["equation"], params["train"], params["net"]
    net_value = [Network(params_net) for _ in range(D)]
    net_control = [Network(params_net, penalty=params_net["penalty"]) for _ in range(D)]
    model = FBSNN(net_value, net_control, params)
    # If the model has already been trained and saved, just read and visualize the results.
    # Please comment out the following line.
    model.train_all()

    # --------------------- Below is the code for result visualization --------------------- #

    t_test, W_test, B_test, P_test, P0_test = model.fetch_minibatch()
    t_test, W_test, B_test, P_test, P0_test = t_test.to(device), W_test.to(device), B_test.to(device), P_test.to(
        device), P0_test.to(device)
    X0 = torch.cat([params_equ["Xi"]] * params_equ["M"])
    X_pred, Y_pred, control_pred, X_test = model.predict(t_test, W_test, B_test, P_test, P0_test, X0.to(device))
    Y_test = model.exact_value(t_test, X_pred)
    t_test = t_test.cpu().detach().numpy()
    X_pred = X_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    Y_pred = Y_pred.cpu().detach().numpy()
    Y_test = Y_test.cpu().detach().numpy()
    P_test = P_test.cpu().detach().numpy()
    P0_test = P0_test.cpu().detach().numpy()
    control_test_list = [torch.tensor(model.control_exact) for _ in range(model.M*(model.N+1))]
    control_test = torch.stack(control_test_list, dim=0).view(model.M, model.N+1, model.D)
    control_pred = control_pred.cpu().detach().numpy()
    control_test = control_test.cpu().detach().numpy()

    samples = min(model.M, 5)
    jump_num = 5
    samples_index = torch.randperm(model.M)[0:samples]
    samples_index = torch.unique(torch.cat([samples_index, model.jump_index[0:jump_num]], dim=0))

    def plot_trajectory(t_test, f_pred, f_test, approximate, exact, start, end, ylabel, filename=None, ylim=None):
        plt.figure(dpi=300)
        plt.plot(t_test[samples_index[0:1], :, 0].T, f_pred[samples_index[0:1], :, 0].T, color='darkorange',
                 label='Approximate {}'.format(approximate), zorder=2)
        plt.plot(t_test[samples_index[0:1], :, 0].T, f_test[samples_index[0:1], :, 0].T, '--', color='darkviolet',
                 label='Exact {}'.format(exact), zorder=3)
        first_legend = None
        for i in range(len(samples_index)):
            for j in range(P_test.shape[1]):
                for k in range(P_test.shape[2]):
                    if P_test[samples_index[i], j, k] > 0:
                        index = torch.searchsorted(torch.tensor(t_test)[samples_index[i], :, 0],
                                                   P_test[samples_index[i], j, k])
                        if first_legend is None:
                            plt.plot(t_test[samples_index[i], index - 1:index + 1, 0],
                                     f_test[samples_index[i], index - 1:index + 1, 0], color='red', linewidth=4,
                                     label='Jump', zorder=1)
                            first_legend = 1
                        else:
                            plt.plot(t_test[samples_index[i], index - 1:index + 1, 0],
                                     f_test[samples_index[i], index - 1:index + 1, 0], color='red', linewidth=4, zorder=1)
            for j in range(P0_test.shape[1]):
                if P0_test[samples_index[i], j, 0] > 0:
                    index = torch.searchsorted(torch.tensor(t_test)[samples_index[i], :, 0],
                                               P0_test[samples_index[i], j, 0])
                    if first_legend is None:
                        plt.plot(t_test[samples_index[i], index - 1:index + 1, 0],
                                 f_test[samples_index[i], index - 1:index + 1, 0], color='red', linewidth=4,
                                 label='Jump', zorder=1)
                        first_legend = 1
                    else:
                        plt.plot(t_test[samples_index[i], index - 1:index + 1, 0],
                                 f_test[samples_index[i], index - 1:index + 1, 0], color='red', linewidth=4, zorder=1)
        plt.plot(t_test[samples_index[0:1], -1, 0], f_test[samples_index[0:1], -1, 0], 'ko',
                 label=end, zorder=4)
        plt.plot(t_test[samples_index[1:(samples + jump_num)], :, 0].T,
                 f_pred[samples_index[1:(samples + jump_num)], :, 0].T, color='darkorange', zorder=2)
        plt.plot(t_test[samples_index[1:(samples + jump_num)], :, 0].T,
                 f_test[samples_index[1:(samples + jump_num)], :, 0].T, '--', color='darkviolet', zorder=3)
        plt.plot(t_test[samples_index[1:(samples + jump_num)], -1, 0],
                 f_test[samples_index[1:(samples + jump_num)], -1, 0], 'ko', zorder=4)
        plt.plot([0], f_test[0, 0, 0], 'ks', label=start, zorder=4)
        plt.xlabel('$t$')
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(os.path.join(model.output_folder, '%s.png' % filename), dpi=300, bbox_inches='tight')
        plt.show()

    # ylim = [[0.6, 1.4], [0.1, 0.9]]
    # for k in range(model.D):
    #     plot_trajectory(t_test, np.expand_dims(Y_pred[:, :, k], axis=2), np.expand_dims(Y_test[:, :, k], axis=2),
    #                     approximate=f"$u_{k}(t,X_t)$", exact=f"$u_{k}(t,X_t)$", start=f"$Y_0 = u_{k}(0,X_0)$",
    #                     end=f"$Y_T = u_{k}(T,X_T)$", ylabel=f"$Y_t=u_{k}(t,X_t)$", filename=f"traj_Y{k}")
    #     plot_trajectory(t_test, np.expand_dims(control_pred[:, :, k], axis=2), np.expand_dims(control_test[:, :, k], axis=2),
    #                     approximate=f"$a_{k}(t,X_t)$", exact=f"$a_{k}(t,X_t)$", start=f"$a_0 = a_{k}(0,X_0)$",
    #                     end=f"$a_T = a_{k}(T,X_T)$", ylabel=f"$a_t=a_{k}(t,X_t)$", filename=f"traj_a{k}")
    #     plot_trajectory(t_test, np.expand_dims(X_pred[:, :, k], axis=2), np.expand_dims(X_test[:, :, k], axis=2),
    #                     approximate=f"$\hat{{X}}_t^{k}=f(t,W_t,M_t,\hat{{a}})$", exact=f"$X_t^{k}=f(t,W_t,M_t,a)$",
    #                     start=f"$X_0^{k}$", end=f"$X_T^{k}$", ylabel=f"$X_t^{k}=f(t,W_t,M_t,a)$", filename=f"traj_X^{k}")
    
    Y_errors = np.sqrt(np.mean((Y_test-Y_pred)**2, 0) / np.mean(Y_test**2, 0))
    plt.figure(dpi=300)
    k = 0
    plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#6565AE', label=f"agent {k+1}")
    for k in range(model.D):
        if k == 1:
            plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#E7483D', label=f"agent 2~10")
        if 2 <= k <= 9:
            plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#E7483D')
        if k == 10:
            plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#FAC074', label=f"agent 11~20")
        if 11 <= k <= 19:
            plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#FAC074')
    plt.xlabel('$t$')
    plt.ylabel('$L^2$ relative error of $e_t^v$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model.output_folder, "game_error_value.png"), dpi=300, bbox_inches='tight')
    plt.show()

    control_errors = np.sqrt(np.mean((control_test - control_pred) ** 2, 0) / np.mean(control_test ** 2, 0))
    plt.figure(dpi=300)
    k = 0
    plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#6565AE', label=f"agent {k + 1}")
    for k in range(model.D):
        if k == 1:
            plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#E7483D', label=f"agent 2~10")
        if 2 <= k <= 9:
            plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#E7483D')
        if k == 10:
            plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#FAC074', label=f"agent 11~20")
        if 11 <= k <= 19:
            plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#FAC074')
    plt.xlabel('$t$')
    plt.ylabel('$L^2$ relative error of $e_t^u$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model.output_folder, "game_error_control.png"), dpi=300, bbox_inches='tight')
    plt.show()

