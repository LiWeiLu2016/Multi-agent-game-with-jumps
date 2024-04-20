import numpy as np
import time
import torch
import psutil
import json
import os
import torch.nn as nn
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

    def __init__(self, inputs: int, width: int, depth: int, output: int, params: dict, activation="Tanh", penalty=None):
        super(Network, self).__init__()
        self.first = nn.Linear(inputs, width)
        self.last = nn.Linear(width, output)
        self.network = nn.Sequential(*[
            self.first,
            *[Block(width, params, activation)] * depth,
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
        self.device = self.params_train["device"]
        self.Xi = self.params_equ["Xi"].to(self.device)    # initial point
        self.T = self.params_equ["T"]              # terminal time
        self.M = self.params_equ["M"]             # number of trajectories
        self.N = self.params_equ["N"]              # number of time snapshots
        self.D = self.params_equ["D"]              # number of dimensions
        self.net_critic = net_value.to(self.device)
        self.net_actor = net_control.to(self.device)
        self.terminal = self.params_equ["terminal"].to(self.device)  # terminal points in the first iteration
        self.terminal.requires_grad = True
        self.optimizer_critic = optim.Adam(self.net_critic.parameters(), lr=self.params_train["lr_critic"])
        self.optimizer_actor = optim.Adam(self.net_actor.parameters(), lr=self.params_train["lr_actor"], weight_decay=self.params_train["weight_decay"])
        self.scheduler_critic = optim.lr_scheduler.MultiStepLR(self.optimizer_critic,
                                                               milestones=self.params_train["milestones_critic"],
                                                               gamma=self.params_train["gamma_critic"])
        self.scheduler_actor = optim.lr_scheduler.MultiStepLR(self.optimizer_actor,
                                                               milestones=self.params_train["milestones_actor"],
                                                               gamma=self.params_train["gamma_actor"])
        self.best_critic_loss, self.best_actor_loss = 10000, 10000
        self.Y_error_list, self.control_error_list, self.bound_list = [], [], []
        self.CriticLoss_list, self.ActorLoss_list = [], []
        self.dt = self.T / self.N

    def exact_critic(self, t, x):
        a, b, q = self.params_equ["a"], self.params_equ["b"], self.params_equ["q"]
        sigma, lam, z = self.params_equ["sigma"], self.params_equ["lambda"], self.params_equ["z"]
        quadratic = np.sqrt(b * q) * (-1 + 2 * (a + np.sqrt(b * q)) / (
                (-a + np.sqrt(b * q)) * torch.exp(-2 * np.sqrt(b / q) * (self.T - t)) + (a + np.sqrt(b * q))))
        zeroth = ((2*self.D-2)*sigma ** 2 + sum([L*Z**2 for L, Z in zip(lam, z)])) * (-np.sqrt(b * q) * (self.T - t) + q * torch.log(
            (-a + np.sqrt(b * q) + (a + np.sqrt(b * q)) * torch.exp(2 * np.sqrt(b / q) * (self.T - t))) / (
                    2 * np.sqrt(b * q))))
        return quadratic * torch.sum(x ** 2, dim=-1, keepdim=True) + zeroth

    def exact_actor(self, t, x):
        a, b, q = self.params_equ["a"], self.params_equ["b"], self.params_equ["q"]
        quadratic = np.sqrt(b * q) * (-1 + 2 * (a + np.sqrt(b * q)) / (
                (-a + np.sqrt(b * q)) * torch.exp(-2 * np.sqrt(b / q) * (self.T - t)) + (a + np.sqrt(b * q))))
        return -quadratic*x/q

    def f_torch(self, t, X):
        control = self.net_actor(torch.cat([t, X], dim=-1))
        q, b = self.params_equ["q"], self.params_equ["b"]
        return q*torch.sum(control**2, dim=-1, keepdim=True) + b*torch.sum(X**2, dim=-1, keepdim=True)

    def g_torch(self, X):
        return self.params_equ["a"]*torch.sum(X**2, dim=-1, keepdim=True)

    def mu_torch(self, t, X):
        control = self.net_actor(torch.cat([t, X], dim=1))
        return control
        
    def sigma_torch(self, t, X):
        s = torch.stack([(torch.eye(self.D) + torch.diag_embed(torch.ones(self.D - 1), offset=-1)) * self.params_equ[
            "sigma"]] * self.M)
        return s

    def xi(self, t, X):
        return torch.tensor(self.params_equ["z"], device=self.device).unsqueeze(0)

    def net_u_Du(self, t, X, compute_Du=True):
        inputs = torch.cat([t, X], dim=1)
        u = self.net_critic(inputs)
        Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True, create_graph=True)[0] if compute_Du else 0
        return u, Du

    def Dg_torch(self, X):
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0]  # M x D

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
        T = self.T
        M = self.M
        N = self.N
        D = self.D
        Dt = np.zeros((M, N + 1, 1))
        DW = np.zeros((M, N + 1, D))
        dt = T / N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        P = np.cumsum(np.random.exponential(scale=[1 / self.params_equ["lambda"][i] for i in range(self.D)],
                                            size=(M, 30, self.D)), axis=1)
        P[np.where(P > T)] = 0
        count = np.count_nonzero(P, axis=1).max()
        P = P[:, 0:count, :]
        self.jump_index = torch.unique(torch.nonzero(torch.tensor(P))[:, 0])
        self.jump_index_D = torch.unique(torch.nonzero(torch.tensor(P)[:, :, self.D-1])[:, 0])
        return torch.from_numpy(t).float().to(self.device), torch.from_numpy(W).float().to(
            self.device), torch.from_numpy(P).float().to(self.device)

    def plot_list(self, list, ylabel, legend, color=None, title=None, ylabel_log=False, xticks=None, baseline=None, filename=None):
        plt.figure(dpi=300)
        for i in range(len(list)):
            if color is not None:
                plt.plot(list[i], label=legend[i], color=color[i])
            else:
                plt.plot(list[i], label=legend[i])
        if baseline is not None:
            plt.plot(torch.ones(self.params_train["iteration"]) * baseline, '--')
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

    def plot_loss(self, CriticLoss, ActorLoss, exact_Y0, filename):
        fig, ax1 = plt.subplots(dpi=300)
        ax1.set_xlabel('iteration')
        # ax1.set_ylabel('CriticLoss', color='#E7483D')
        plt.gca().set_yscale("log")
        ax2 = ax1.twinx()
        # ax2.set_ylabel('ActorLoss', color='#6565AE')
        ax2.plot(ActorLoss, color='#6565AE', label='ActorLoss', zorder=1)
        ax2.plot(torch.ones(len(ActorLoss)) * exact_Y0, '--', color='#8FC751', zorder=1)
        ax2.tick_params(axis='y', labelcolor='#6565AE')
        ax2.set_ylim(27.5, 35)
        ax2.set_yticks([28, 30, 30.6, 32, 34])
        ax1.plot(CriticLoss, color='#E7483D', label='CriticLoss', zorder=2)
        ax1.tick_params(axis='y', labelcolor='#E7483D')
        ax1.set_ylim(0, 1)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best', framealpha=0.5, edgecolor='gray')
        if filename is not None:
            plt.savefig(os.path.join(self.output_folder, '%s.png' % filename), dpi=300, bbox_inches='tight')
        plt.show()

    def loss_function(self, t, W, P, X0, i, n, traj_only=False, control_exact=False):
        self.loss1 = torch.zeros(1, device=self.device)
        X_buffer, Y_buffer = [], []
        t0 = t[:, i, :]             # M x 1
        W0 = W[:, i, :]
        X0.requires_grad = True
        Y0, Z0 = self.net_u_Du(t0, X0, compute_Du=not traj_only)
        reward = 0
        if i == 0:
            X_buffer.append(X0)
            Y_buffer.append(Y0)
        for k in range(0, n):
            t1 = t[:, i+k+1, :]
            W1 = W[:, i+k+1, :]
            P1 = torch.where(P < t1[0, 0], P, torch.zeros_like(P, dtype=torch.float))
            P2 = torch.where(P1 > t0[0, 0], P1, torch.zeros_like(P1, dtype=torch.float))
            P3 = torch.sum(P2, dim=1)
            P0 = torch.where(P3 != 0, 1, 0)
            diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1))
            dMt = P0 - torch.tensor(self.params_equ["lambda"], device=self.device).unsqueeze(0)*self.dt
            if not control_exact:
                X1 = X0 + self.mu_torch(t0, X0) * self.dt + diffusion.squeeze(2) + self.xi(t0, X0)*dMt
            else:
                mu_torch = self.exact_actor(t0, X0)
                X1 = X0 + mu_torch * self.dt + diffusion.squeeze(2) + self.xi(t0, X0)*dMt
            Y1, Z1 = self.net_u_Du(t1, X1, compute_Du=not traj_only)
            if not traj_only:
                reward = reward + self.f_torch(t0, X0)*self.dt - torch.bmm(Z0.unsqueeze(1), diffusion).squeeze(1)
                for j in range(self.D):
                    control = self.net_actor(torch.cat([t0, X0], dim=-1))
                    newX = torch.zeros_like(control)
                    newX[:, j] = self.xi(t0, X0)[:, j]
                    nonlocal_term, _ = self.net_u_Du(t0, X0 + newX, compute_Du=False)
                    reward = reward - (nonlocal_term - Y0) * dMt[:, j].unsqueeze(1)
                self.loss1 = self.loss1 + torch.mean((Y0 - reward - Y1)**2)
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1
            X_buffer.append(X0)
            Y_buffer.append(Y0)

        if not traj_only:
            if (i + n) == self.N:
                self.X1 = X1.detach()
                self.X1.requires_grad = True
            YT, Z1 = self.net_u_Du(torch.ones(self.M, 1) * self.T, self.X1)
            self.loss2 = torch.mean((YT - self.g_torch(self.X1))**2) / self.N * n
        loss_critic = self.loss1 + self.loss2 if not traj_only else 0
        X = torch.stack(X_buffer, dim=1)
        Y = torch.stack(Y_buffer, dim=1)
        loss_actor = torch.mean(torch.sum(self.f_torch(t, X)[:, 0:-1, :], dim=1)*self.dt + self.g_torch(X[:, -1, :])) if n == self.N else 0
        return loss_critic, loss_actor, X, Y, Y[0,0,0]

    def Critic_step(self, n):
        t_batch, W_batch, P_batch = self.fetch_minibatch()
        for i in range(int(self.N / n)):
            X0 = torch.cat([self.Xi] * self.M) if i == 0 else X1.detach()
            loss_critic, _, X_pred, _, _ = self.loss_function(t_batch, W_batch, P_batch, X0, i=i * n, n=n)
            X1 = X_pred[:, -1, :]
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()
            self.scheduler_critic.step()
        return loss_critic

    def Actor_step(self, n):
        t_batch, W_batch, P_batch = self.fetch_minibatch()
        _, loss_actor, _, _, _ = self.loss_function(t_batch, W_batch, P_batch, torch.cat([self.Xi] * self.M), i=0, n=self.N, traj_only=True)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        for _ in range(n):
            self.optimizer_actor.step()
            self.scheduler_actor.step()
        self.bound_list.append(self.net_actor.bound.detach().clone())
        return loss_actor

    def compute_error(self):
        t, W, P = model.fetch_minibatch()
        _, _, X_pred, Y_pred, _ = self.loss_function(t, W, P, torch.cat([self.Xi] * self.M), i=0, n=self.N, traj_only=True)
        Y_exact = self.exact_critic(t, X_pred)
        Y_error = torch.sum(torch.sqrt(torch.mean((Y_pred - Y_exact) ** 2, dim=0) / torch.mean(Y_exact ** 2, dim=0))[0:-1]) * self.dt
        control_pred = self.net_actor(torch.cat([t, X_pred], dim=2))
        control_exact = self.exact_actor(t, X_pred)
        control_error = torch.mean(torch.sum(
            torch.sqrt(torch.mean((control_pred - control_exact) ** 2, dim=0) / torch.mean(control_exact ** 2, dim=0))[0:-1, :], dim=0) * self.dt)
        return Y_error.squeeze(), control_error.squeeze()

    def train_all(self, n, N_Iter=10):
        start_time = time.time()
        t0 = torch.zeros(self.M, 1, device=self.device)
        self.X1 = self.terminal
        Y_temp, Z_temp = self.net_u_Du(t0, self.terminal)
        self.Y1, self.Z1 = Y_temp.detach(), Z_temp.detach()
        self.bound_list.append(self.net_actor.bound.detach().clone())
        a, b, q = self.params_equ["a"], self.params_equ["b"], self.params_equ["q"]
        condition = q > 0 and b > 0 and a > -np.sqrt(b*q)*(1+2/(np.exp(2*self.T*np.sqrt(b/q))-1))
        print("Condition {q<0, b<0, a>-sqrt(b*q)*(1+2/(e^{2*sqrt(b/q)*T}-1)}:", "Satisfy" if condition else "Dissatisfy")
        print("Dimension:", self.D)
        print("Exact Y0:", self.exact_critic(torch.tensor([[0.]]), self.Xi).squeeze().numpy())
        for it in range(N_Iter):
            self.it = it
            loss_critic = self.Critic_step(n)
            self.CriticLoss_list.append(loss_critic.detach().clone())
            with torch.no_grad():
                Y_error, _ = self.compute_error()
            loss_actor = self.Actor_step(self.params_train["actor_step"])
            self.ActorLoss_list.append(loss_actor.detach().clone())
            with torch.no_grad():
                _, control_error = self.compute_error()
            self.Y_error_list.append(Y_error.detach())
            self.control_error_list.append(control_error.detach())
            if it % 2 == 0:
                elapsed = time.time() - start_time
                print(
                    'It: %d, Time: %.2f, Loss_critic: %.3e, Loss_actor: %.3e, Y: %.3f%%, control: %.3f%%' % (
                    it, elapsed, loss_critic.item(), loss_actor.item(), Y_error.detach().numpy() * 100,
                    control_error.detach().numpy() * 100))
            if it % 50 == 0:
                self.get_memory("GB")
        self.training_cost = time.time() - start_time
        print("The training is over!")
        self.save_and_read_model(type="save")

    def save_and_read_model(self, type="save"):
        filename = os.path.basename(os.path.abspath(__file__))
        filename_without_extension, extension = os.path.splitext(filename)
        filename = f"{filename_without_extension}_{self.D}{extension}"
        self.output_folder = os.path.join('output', filename)
        name = ["Y_error.json", "control_error.json", "bound.json", "CriticLoss.json", "ActorLoss.json"]
        save_list = [self.Y_error_list, self.control_error_list, self.bound_list, self.CriticLoss_list, self.ActorLoss_list]
        if type == "save":
            os.makedirs(self.output_folder, exist_ok=True)
            torch.save(self.net_critic.state_dict(), os.path.join(self.output_folder, 'critic.pth'))
            torch.save(self.net_actor.state_dict(), os.path.join(self.output_folder, 'actor.pth'))
            for i in range(len(name)):
                with open(os.path.join(self.output_folder, name[i]), 'w') as file:
                    json.dump([tensor.tolist() for tensor in save_list[i]], file)
            with open(os.path.join(self.output_folder, 'training_cost.txt'), 'w') as file:
                file.write(str(self.training_cost))
        elif type == "read":
            self.net_critic.load_state_dict(torch.load(os.path.join(self.output_folder, 'critic.pth')))
            self.net_actor.load_state_dict(torch.load(os.path.join(self.output_folder, 'actor.pth')))
            for i in range(len(name)):
                with open(os.path.join(self.output_folder, name[i]), 'r') as file:
                    save_list[i][:] = [torch.tensor(j) for j in json.load(file)]
            with open(os.path.join(self.output_folder, 'training_cost.txt'), 'r') as file:
                saved_execution_time = float(file.read())
                print(f"The training costs: %.2f min" % (saved_execution_time/60))
        else:
            raise RuntimeError("You can only save or read the model!")

    def predict(self, Xi_star, t_star, W_star, P_star):
        Y0 = self.exact_critic(torch.zeros(1, 1), self.Xi).squeeze()
        self.save_and_read_model(type="read")
        print("Y error: %.3f%%, control error: %.3f%%, Y0: %.3f" % (
        self.Y_error_list[-1].numpy() * 100, self.control_error_list[-1].numpy() * 100, Y0.numpy()))
        self.plot_list([self.Y_error_list, self.control_error_list], "Error_value/Error_control", ["value function", "control"],
                       color=["#E7483D", "#6565AE"], title=None, ylabel_log=True, baseline=None, filename="LQR_train")
        self.plot_loss(self.CriticLoss_list, self.ActorLoss_list, Y0.numpy(), filename="LQR_loss")
        _, _, X_star, Y_star, _ = self.loss_function(t_star, W_star, P_star, Xi_star, i=0, n=self.N, traj_only=True)
        _, _, X_test, _, _ = self.loss_function(t_star, W_star, P_star, Xi_star, i=0, n=self.N, traj_only=True, control_exact=True)
        control_star = self.net_actor(torch.cat([t_star, X_star], dim=2))
        return X_star, Y_star, control_star, X_test


if __name__ == '__main__':
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    device = torch.device('cpu')
    print("device:", device)
    parameters = {"axes.labelsize": 20, "axes.titlesize": 20,
                  "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}
    plt.rcParams.update(parameters)

    D = 25
    params = {
        "equation": {
            "sigma": 0.4, "lambda": np.linspace(0.2, 0.3, D).tolist(), "z": np.linspace(0.3, 0.2, D).tolist(),
            "a": 1, "b": 0.1, "q": 5, "D": D, "T": 1, "M": 500, "N": 50
        },
        "train": {
            "lr_critic": 1e-3, "gamma_critic": 0.1, "milestones_critic": [30000, 40000],
            "lr_actor": 1e-3, "gamma_actor": 0.1, "milestones_actor": [6000, 8000],
            "iteration": 1000, "actor_step": 10, "device": device, "weight_decay": 0, "plot_control_dim": D
        }
    }
    params_equ, params_train = params["equation"], params["train"]
    params["equation"]["Xi"] = torch.ones(1, D) * 1
    params["equation"]["terminal"] = torch.ones(params_equ["M"], D)
    params_act = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()}

    net_value = Network(inputs=D+1, width=D+10, depth=3, output=1, params=params_act, activation="Tanh")
    net_control = Network(inputs=D+1, width=D+10, depth=3, output=D, params=params_act, activation="Tanh", penalty="Tanh")
    model = FBSNN(net_value, net_control, params)
    # If the model has already been trained and saved, just read and visualize the results.
    # Please comment out the following line.
    model.train_all(n=1, N_Iter=params_train["iteration"])

    # --------------------- Below is the code for result visualization --------------------- #

    t_test, W_test, P_test = model.fetch_minibatch()
    t_test, W_test = t_test.to(device), W_test.to(device)
    X0 = torch.cat([params_equ["Xi"]] * params_equ["M"])
    X_pred, Y_pred, control_pred, X_test = model.predict(X0.to(device), t_test, W_test, P_test)
    Y_test = model.exact_critic(t_test, X_pred)
    control_test = model.exact_actor(t_test, X_pred)
    t_test = t_test.cpu().detach().numpy()
    X_pred = X_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    Y_pred = Y_pred.cpu().detach().numpy()
    Y_test = Y_test.cpu().detach().numpy()
    control_test = control_test.cpu().detach().numpy()
    control_pred = control_pred.cpu().detach().numpy()

    samples = min(model.M, 5)
    jump_num = 5
    samples_index = torch.randperm(model.M)[0:samples]
    samples_index = torch.unique(torch.cat([samples_index, model.jump_index[0:jump_num]], dim=0))
    samples_index_D = torch.unique(torch.cat([torch.randperm(model.M)[0:samples], model.jump_index_D[0:jump_num]], dim=0))

    def plot_trajectory(t_test, f_pred, f_test, control_dim, approximate, exact, start, end, ylabel, ylim, filename):
        plt.figure(dpi=300)
        plt.plot(t_test[samples_index[0:1], :, 0].T, f_pred[samples_index[0:1], :, 0].T, color='darkorange',
                 label='Approx. {}'.format(approximate), zorder=2)
        plt.plot(t_test[samples_index[0:1], :, 0].T, f_test[samples_index[0:1], :, 0].T, '--', color='darkviolet',
                 label='Exact {}'.format(exact), zorder=3)
        first_legend = None
        for i in range(len(samples_index)):
            for j in range(P_test.shape[1]):
                for k in range(D):
                    if k in control_dim:
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
        plt.legend(ncol=2, framealpha=0.5, edgecolor='gray', handletextpad=0.5, handlelength=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(model.output_folder, '%s.png' % filename), dpi=300, bbox_inches='tight')
        plt.show()

    plot_trajectory(t_test, Y_pred, Y_test, control_dim=[i for i in range(D)], approximate="$\hat{v}(t,X_t)$", exact="$v(t,X_t)$",
                    start="$v(0,X_0)$", end="$v(T,X_T)$", ylabel=None, ylim=[5, 40], filename="LQR_traj_value")
    plot_trajectory(t_test, control_pred[:, :, -1][..., np.newaxis], control_test[:, :, -1][..., np.newaxis],
                    control_dim=[D - 1], approximate="$\hat{u}_{25}(t,X_t)$", exact="$u^*_{25}(t,X_t)$",
                    start="$u_{25}(0,X_0)$", end="$u_{25}(T,X_T)$", ylabel=None,
                    ylim=[-0.35, 0.3], filename="LQR_traj_control")
    
    Y_errors = np.sqrt(np.mean((Y_test-Y_pred)**2, 0) / np.mean(Y_test**2, 0))
    control_errors = np.mean(np.sqrt(np.mean((control_test - control_pred) ** 2, 0) / np.mean(control_test ** 2, 0)), 1, keepdims=True)

    plt.figure(dpi=300)
    plt.plot(t_test[0, :, 0], Y_errors, color='#E7483D', label='value function')
    plt.plot(t_test[0, :, 0], control_errors, color='#6565AE', label='control')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$L^2$ relative errors $e_t^v$/$e_t^u$')
    plt.tight_layout()
    plt.savefig(os.path.join(model.output_folder, "LQR_error.png"), dpi=300, bbox_inches='tight')
    plt.show()

