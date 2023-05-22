import torch
import torch.nn as nn
import numpy as np
from elegantrl.net import layer_norm

class ActorResidualPPO(nn.Module):
	def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
		super().__init__()
		if isinstance(state_dim, int):
			if if_use_dn:
				nn_dense = DenseNet(mid_dim // 2)
				inp_dim = nn_dense.inp_dim
				out_dim = nn_dense.out_dim

				self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.Tanh(),
										 nn_dense,
										 nn.Linear(out_dim, action_dim), )
			else:
				self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, action_dim), )
		else:
			def set_dim(i):
				return int(12 * 1.5 ** i)

			self.net = nn.Sequential(NnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
									 nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
									 nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.Tanh(),
									 nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.Tanh(),
									 nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.Tanh(),
									 nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.Tanh(),
									 nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.Tanh(),
									 NnReshape(-1),
									 nn.Linear(set_dim(5), mid_dim), nn.Tanh(),
									 nn.Linear(mid_dim, action_dim), )

		self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		self.priorK = nn.Parameter(torch.randn(state_dim, action_dim)*0.01, requires_grad=False)
		layer_norm(self.net[-1], std=0.1)  # output layer for action

	def forward(self, state):
		return self.net(state).tanh() + state@self.priorK   # action

	def get_action_noise(self, state):
		a_avg = self.net(state)
		a_std = self.a_std_log.exp()

		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def compute_logprob(self, state, action):
		a_avg = self.net(state)
		a_std = self.a_std_log.exp()
		delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
		return logprob.sum(1)

	def frozen_transfer(self):
		for param in self.net.parameters():
			param.requires_grad = False
		for param in self.net[-1].parameters():
			param.requires_grad = True
		

class ActorResidualIntegratorModularSinglePPO(nn.Module):
	def __init__(self, mid_dim, state_dim, action_dim, integrator_dim, if_use_dn=False):
		super().__init__()
		# print(f'integrator_dim {integrator_dim}')
		other_dim = state_dim -  integrator_dim
		self.other_dim = other_dim
		if if_use_dn:
			nn_dense = DenseNet(mid_dim // 2)
			inp_dim = nn_dense.inp_dim
			out_dim = nn_dense.out_dim

			self.other_net = nn.Sequential(nn.Linear(other_dim, inp_dim), nn.Tanh(),
									 nn_dense,
									 nn.Linear(out_dim, action_dim), )
		else:
			self.other_net = nn.Sequential(nn.Linear(other_dim, 3*mid_dim//4 ), nn.Tanh(),
										 nn.Linear(3*mid_dim//4 , 3*mid_dim//4 ), nn.Tanh(),
										 nn.Linear(3*mid_dim//4, mid_dim//2), nn.Tanh())

		self.integrator_net = nn.Sequential(nn.Linear(integrator_dim, 3*mid_dim//4), nn.Tanh(),
										 nn.Linear(3*mid_dim//4, 3*mid_dim//4), nn.Tanh(),
										 nn.Linear(3*mid_dim//4, mid_dim // 2), nn.Tanh())

		self.net = nn.Sequential(nn.Linear(mid_dim//2 *2, action_dim))

		self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		self.priorK = nn.Parameter(torch.randn(state_dim, action_dim)*0.01, requires_grad=False)
		layer_norm(self.net[-1], std=0.1)  # output layer for action

	def forward(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		return self.net(torch.cat([tmp1, tmp2], dim=-1)).tanh() + state@self.priorK   # action

	def get_action_noise(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		a_avg = self.net(torch.cat([tmp1, tmp2], dim=-1))
		a_std = self.a_std_log.exp()

		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def compute_logprob(self, state, action):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		a_avg = self.net(torch.cat([tmp1, tmp2], dim=-1))

		a_std = self.a_std_log.exp()
		delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
		return logprob.sum(1)

	def frozen_integrator(self):
		for param in self.integrator_net.parameters():
			param.requires_grad = False

	def frozen_transfer(self):
		for param in self.integrator_net.parameters():
			param.requires_grad = False
		for param in self.other_net.parameters():
			param.requires_grad = False
			
		for param in self.net.parameters():
			param.requires_grad = False
		for param in self.net[-1].parameters():
			param.requires_grad = True

class ActorResidualIntegratorModularPPO(nn.Module):
	def __init__(self, mid_dim, state_dim, action_dim, integrator_dim, if_use_dn=False):
		super().__init__()
		# print(f'integrator_dim {integrator_dim}')
		other_dim = state_dim -  integrator_dim
		self.other_dim = other_dim
		if if_use_dn:
			nn_dense = DenseNet(mid_dim // 2)
			inp_dim = nn_dense.inp_dim
			out_dim = nn_dense.out_dim

			self.other_net = nn.Sequential(nn.Linear(other_dim, inp_dim), nn.Tanh(),
									 nn_dense,
									 nn.Linear(out_dim, action_dim), )
		else:
			self.other_net = nn.Sequential(nn.Linear(other_dim, mid_dim ), nn.Tanh(),
										 nn.Linear(mid_dim , mid_dim // 2 ), nn.Tanh())

		self.integrator_net = nn.Sequential(nn.Linear(integrator_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim // 2), nn.Tanh())

		self.net = nn.Sequential(nn.Linear(mid_dim//2 *2, mid_dim), nn.Tanh(),
								nn.Linear(mid_dim, action_dim),)

		self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		self.priorK = nn.Parameter(torch.randn(state_dim, action_dim)*0.01, requires_grad=False)
		layer_norm(self.net[-1], std=0.1)  # output layer for action

	def forward(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		return self.net(torch.cat([tmp1, tmp2], dim=-1)).tanh() + state@self.priorK   # action

	def get_action_noise(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		a_avg = self.net(torch.cat([tmp1, tmp2], dim=-1))
		a_std = self.a_std_log.exp()

		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def compute_logprob(self, state, action):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = self.integrator_net(state[:, self.other_dim:])
		a_avg = self.net(torch.cat([tmp1, tmp2], dim=-1))

		a_std = self.a_std_log.exp()
		delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
		return logprob.sum(1)

	def frozen_integrator(self):
		for param in self.integrator_net.parameters():
			param.requires_grad = False

	def frozen_transfer(self):
		for param in self.integrator_net.parameters():
			param.requires_grad = False
		for param in self.other_net.parameters():
			param.requires_grad = False

		for param in self.net.parameters():
			param.requires_grad = False
		for param in self.net[-1].parameters():
			param.requires_grad = True
		
class ActorResidualIntegratorModularLinearPPO(nn.Module):
	def __init__(self, mid_dim, state_dim, action_dim, integrator_dim, if_use_dn=False):
		super().__init__()
		# print(f'integrator_dim {integrator_dim}')s
		other_dim = state_dim -  integrator_dim
		self.other_dim = other_dim
		if if_use_dn:
			nn_dense = DenseNet(mid_dim // 2)
			inp_dim = nn_dense.inp_dim
			out_dim = nn_dense.out_dim

			self.other_net = nn.Sequential(nn.Linear(other_dim, inp_dim), nn.Tanh(),
									 nn_dense,
									 nn.Linear(out_dim, action_dim), )
		else:
			self.other_net = nn.Sequential(nn.Linear(other_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, action_dim), )


		self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		self.priorK = nn.Parameter(torch.randn(state_dim, action_dim)*0.01, requires_grad=False)
		self.integratorK = nn.Parameter(torch.randn(integrator_dim, action_dim)*0.01, requires_grad=True)

		layer_norm(self.net[-1], std=0.1)  # output layer for action

	def forward(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = state[:, self.other_dim:]@self.integratorK
		return (tmp1+tmp2).tanh() + state@self.priorK   # action

	def get_action_noise(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = state[:, self.other_dim:]@self.integratorK
		a_avg = tmp1+tmp2

		a_std = self.a_std_log.exp()
		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def compute_logprob(self, state, action):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = state[:, self.other_dim:]@self.integratorK
		a_avg = tmp1+tmp2

		a_std = self.a_std_log.exp()
		delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
		return logprob.sum(1)

	def frozen_integrator(self):
		self.integratorK.requires_grad = False

	def frozen_transfer(self):
		self.integratorK.requires_grad = False
		for param in self.other_net.parameters():
			param.requires_grad = False
		for param in self.other_net[-1].parameters():
			param.requires_grad = True

		

class ActorResidualIntegratorModularLinearOutPPO(nn.Module):
	def __init__(self, mid_dim, state_dim, action_dim, integrator_dim, if_use_dn=False):
		super().__init__()
		# print(f'integrator_dim {integrator_dim}')
		other_dim = state_dim -  integrator_dim
		self.other_dim = other_dim
		if if_use_dn:
			nn_dense = DenseNet(mid_dim // 2)
			inp_dim = nn_dense.inp_dim
			out_dim = nn_dense.out_dim

			self.other_net = nn.Sequential(nn.Linear(other_dim, inp_dim), nn.Tanh(),
									 nn_dense,
									 nn.Linear(out_dim, action_dim), )
		else:
			self.other_net = nn.Sequential(nn.Linear(other_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, mid_dim), nn.Tanh(),
										 nn.Linear(mid_dim, action_dim), )


		self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		self.priorK = nn.Parameter(torch.randn(state_dim, action_dim)*0.01, requires_grad=False)
		self.integratorK = nn.Parameter(torch.randn(integrator_dim, action_dim)*0.01, requires_grad=True)

		layer_norm(self.net[-1], std=0.1)  # output layer for action

	def forward(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		tmp2 = state[:, self.other_dim:]@self.integratorK
		return (tmp1).tanh() + tmp2 + state@self.priorK   # action

	def get_action_noise(self, state):
		tmp1 = self.other_net(state[:, :self.other_dim])
		a_avg =(tmp1+ state[:, self.other_dim:]@self.integratorK).tanh()

		a_std = self.a_std_log.exp()
		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def compute_logprob(self, state, action):
		tmp1 = self.other_net(state[:, :self.other_dim])
		a_avg = (tmp1 + state[:, self.other_dim:]@self.integratorK).tanh()

		a_std = self.a_std_log.exp()
		delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
		return logprob.sum(1)

	def frozen_integrator(self):
		self.integratorK.requires_grad = False

	def frozen_transfer(self):
		self.integratorK.requires_grad = False
		for param in self.other_net.parameters():
			param.requires_grad = False
		for param in self.other_net[-1].parameters():
			param.requires_grad = True


