import os
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.agent import AgentPPO, AgentBase
from elegantrl import logger
from elegantrl.net import CriticTwin, CriticAdv
from elegantrl.net_residual import ActorResidualPPO, ActorResidualIntegratorModularPPO, \
                                    ActorResidualIntegratorModularSinglePPO, \
                                    ActorResidualIntegratorModularLinearPPO,\
                                    ActorResidualIntegratorModularLinearOutPPO

class Residual:
    def init_residual(self, residual_kwarg):
        # self.act.K = nn.Parameter(-torch.Tensor(SCN_kwarg['init_K']).to(self.device), requires_grad=True)
        self.act.priorK = nn.Parameter(-torch.Tensor(residual_kwarg['init_K']).to(self.device), requires_grad=False)
        self.init_actor_zero()
        print(f'K: {self.act.priorK}')
        self.priorK = -residual_kwarg['init_K']

    def fix_K(self):
        self.act.priorK.requires_grad = False   

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class AgentResidualPPO(AgentPPO, Residual):

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorResidualPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def init_actor_zero(self):
        # set last layer's weights and bias to 0.
        self.act.net[-1].bias.data.fill_(0.)
        self.act.net[-1].weight.data.fill_(0.)
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        buffer.empty_buffer_before_explore()  # NOTICE! necessary for on-policy
        # assert target_step == buffer.max_len - max_step

        actual_step = 0
        while actual_step < target_step:
            state = env.reset()
            for _ in range(env.max_step):
                action, noise = self.select_action(state)
                next_state, reward, done, _ = env.step(np.tanh(action)+state@self.priorK)
                actual_step += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_buffer(state, other)
                if done:
                    break
                state = next_state
        return actual_step

    def frozen_transfer(self):
        self.act.frozen_transfer()
        self.cri.frozen_transfer()


class AgentResidualIntegratorModularPPO(AgentResidualPPO):
    def init(self, net_dim, state_dim, action_dim, integrator_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorResidualIntegratorModularPPO(net_dim, state_dim, action_dim, integrator_dim ,self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def frozen_integrator(self):
        self.act.frozen_integrator()
        self.cri.frozen_transfer()

    def frozen_transfer(self):
        self.act.frozen_transfer()
        self.cri.frozen_transfer()

class AgentResidualIntegratorModularSinglePPO(AgentResidualIntegratorModularPPO):
    def init(self, net_dim, state_dim, action_dim, integrator_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorResidualIntegratorModularSinglePPO(net_dim, state_dim, action_dim, integrator_dim ,self.if_use_dn).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER


class AgentResidualIntegratorModularLinearPPO(AgentResidualPPO):
    def init(self, net_dim, state_dim, action_dim, integrator_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorResidualIntegratorModularLinearPPO(net_dim, state_dim, action_dim, integrator_dim ,self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def init_actor_zero(self):
        # set last layer's weights and bias to 0.
        self.act.other_net[-1].bias.data.fill_(0.)
        self.act.other_net[-1].weight.data.fill_(0.)
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])

    def frozen_integrator(self):
        self.act.frozen_integrator()
        self.cri.frozen_transfer()
        
    def frozen_transfer(self):
        self.act.frozen_transfer()
        self.cri.frozen_transfer()

class AgentResidualIntegratorModularLinearOutPPO(AgentResidualPPO):
    def init(self, net_dim, state_dim, action_dim, integrator_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorResidualIntegratorModularLinearOutPPO(net_dim, state_dim, action_dim, integrator_dim ,self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER
        self.integrator_dim = integrator_dim

    def init_actor_zero(self):
        # set last layer's weights and bias to 0.
        self.act.other_net[-1].bias.data.fill_(0.)
        self.act.other_net[-1].weight.data.fill_(0.)
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])

    def frozen_integrator(self):
        self.act.frozen_integrator()
        self.cri.frozen_transfer()

    def frozen_transfer(self):
        self.act.frozen_transfer()
        self.cri.frozen_transfer()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        buffer.empty_buffer_before_explore()  # NOTICE! necessary for on-policy
        # assert target_step == buffer.max_len - max_step

        actual_step = 0
        while actual_step < target_step:
            state = env.reset()
            for _ in range(env.max_step):
                action, noise = self.select_action(state)
                next_state, reward, done, _ = env.step(action+state@self.priorK)
                actual_step += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_buffer(state, other)
                if done:
                    break
                state = next_state
        return actual_step


