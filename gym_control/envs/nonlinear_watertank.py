import gym
from gym import error, spaces, utils
from gym import Wrapper
from gym.utils import seeding
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import control.matlab
from collections import deque 

class StackingHistoryPreprocessing(gym.Wrapper):
	def __init__(
		self,
		env: gym.Env,
		num_stack: int,
	):
		"""Wrapper for Atari 2600 preprocessing.
			Args:
				env (Env): The environment to apply the preprocessing
				history_size: The history as observation
		"""
		super().__init__(env)
		assert num_stack > 0
		self.num_stack = num_stack
		self.env = env
		self.frames = deque(maxlen=num_stack)

		low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack)
		high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack)
		
		self.observation_space = spaces.Box(
				low=low, high=high, dtype=self.observation_space.dtype
			)

	def observation(self, observation):
		"""Converts the wrappers current frames to lazy frames.
		Args:
			observation: Ignored
		"""
		assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
		return np.array(self.frames)

	def step(self, action):
		observation, reward, terminated, info = self.env.step(action)
		self.frames.append(observation)
		return self.observation(None), reward, terminated, info


	def reset(self, **kwargs):
		"""Reset the environment with kwargs.
		Args:
			**kwargs: The kwargs for the environment reset
		Returns:
			The stacked observations
		"""
		obs = self.env.reset()
		info = None  # Unused
		[self.frames.append(obs) for _ in range(self.num_stack)]

		return self.observation(None)


def goal_distance(goal_a, goal_b):
	# if isinstance(goal_a, np.ndarray):
	# 	assert goal_a.shape == goal_b.shape
	# 	return np.linalg.norm((goal_a - goal_b).reshape(-1,1), axis=1).reshape(-1,1)
	# return np.linalg.norm(goal_a - goal_b)**2
	if isinstance(goal_a, np.ndarray):
		return np.linalg.norm(goal_a-goal_b).reshape(-1,1)
	else:
		return np.linalg.norm(goal_a-goal_b)

def goal_distance_tensor(goal_a, goal_b):
	# with torch.no_grad():
	# 	return (goal_a-goal_b).reshape(-1,1)**2
	# 	assert goal_a.shape == goal_b.shape
	# 	# print(f"goal_a: {goal_a}")
	# 	# exit()
	# 	return torch.norm((goal_a-goal_b).reshape(-1,1), dim=1).reshape(-1,1).to(device=goal_a.device, dtype=goal_a.dtype)**2
	return (goal_a-goal_b).reshape(-1,1)

class NonLinearWaterTank(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self,
				 a1=1,
				 a2=1,
				 A1=2,
				 A2=2,
				 Kp=2,
				 G=9.8,
				 z1 = 1,
				 z2 = 0.1,
				 max_step = 500,
				 noise_scale = 0.01,
				 gamma=0.99,
				 seed=None,
				 r = 9.0,
				 N = 100,
				 overflow_cost=-10,
				 n_discrete = 1,
				 sample_t = 0.02,
				 reward_type='distance',
				 controller_type='linearized',
				 distance_threshold=0.05,
				 linearize_r=9.0,
				 reset_from_last_state=True,
				 P_control_K = np.array([0., 0.4]),
				 P_control_L = np.array([-0.4]),
				 P_max_action = 10.0,
				 ):
		"""
		Param:
		a1: cross-sectional area of the outflow orifice at the bottom of first tank
		Ai: cross-sectional area of the inflow
		Vp: voltage to the pump
		Kp: pump constant
		z1: punishment on second tank
		z2: punishment on action
		noise_scale: process noise_scale
		"""
		super().__init__()
		self.max_step = max_step
		self.reward_type = reward_type
		self.distance_threshold = distance_threshold

		self.a1 = a1
		self.a2 = a2
		self.A1 = A1
		self.A2 = A2
		self.Kp = Kp
		self.G = G

		self.m = self._get_m()
		self.n = 1


		self.gamma = gamma
		self.noise_scale = noise_scale
		self.noise_cov = np.eye(self.m)*noise_scale
		self.r = r
		self.z1 = z1
		self.z2 = z2
		self.overflow_cost = overflow_cost

		self.h1 = None
		self.h2 = None

		self.sample_t = sample_t
		self.n_discrete = n_discrete
		self.delta_t = sample_t / n_discrete
		self.reset_from_last_state = reset_from_last_state

		self.P_control_K = P_control_K
		self.P_control_L = P_control_L
		self.P_max_action = P_max_action

		if controller_type == 'linearized':
			self.linearize(linearize_r)
			self._observationSpace = self._observationSpace_linearized
			self._actionSpace = self._actionSpace_linearized
			self._get_observe = self._get_observe_linearized
			self.action = self.action_linearized
			self.get_linear_action = self.get_lqr_action
			self.get_linear_acion_tensor = self.get_lqr_action_tensor
		elif controller_type == 'P':
			self._observationSpace = self._observationSpace_P
			self._actionSpace = self._actionSpace_P
			self._get_observe = self._get_observe_P
			self.action = self.action_P
			
			# K L here is something just tuning
			self.K, self.L = self.P_KL()
			# self.qv, self.qvc = self.qvalue_weights()
			self.get_linear_action = self.get_P_action
			self.get_linear_action_tensor = self.get_P_action_tensor
		else:
			print('wrong option for controller_type')


		self.observation_space = self._observationSpace()
		self.action_space = self._actionSpace()
		self.seed(seed)

		self.last_h1 = None
		self.last_h2 = None


	def _get_m(self):
		return 2

	def reset(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self._episode_steps = 0
		return self._get_observe()

	def set_state(self, h1, h2):
		self.h1 = h1
		self.h2 = h2
		return self._get_observe()

	def set_r(self, r):
		self.r = r
		return self._get_observe()
	
	# Linearized controller method
	def _actionSpace_linearized(self):
		self.a = -1
		self.b = 1
		self.oa = -self.eq_action
		self.ob = self.eq_action
		low_action = np.ones(self.n)*self.a
		high_action = np.ones(self.n)*self.b
		return spaces.Box(	low=low_action,
							high=high_action,
							dtype=np.float32)

	def _observationSpace_linearized(self):
		low_state = -np.ones(self.m)*0
		high_state = np.ones(self.m)*np.inf
		return spaces.Box(low=low_state,
							high=high_state,
							dtype=np.float32)

	def action_linearized(self, action):
		return float(action)*self.eq_action + self.eq_action

	def _get_observe_linearized(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2]) - self.eq_state
		return self.state

	# P controller method
	def _actionSpace_P(self):
		self.a = -1
		self.b = 1
		low_action = np.ones(self.n)*self.a
		high_action = np.ones(self.n)*self.b
		return spaces.Box(	low=low_action,
							high=high_action,
							dtype=np.float32)

	def _observationSpace_P(self):
		low_state = -np.ones(self.m)*0
		high_state = np.ones(self.m)*np.inf
		return spaces.Box(low=low_state,
							high=high_state,
							dtype=np.float32)

	def action_P(self, action):
		# 6.64 is the eq_action of linearized controller
		return float(action)*self.P_max_action/2. + self.P_max_action/2.

	def _get_observe_P(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2])
		return self.state
		
	def P_KL(self):
		K, L = self.P_control_K, self.P_control_L
		return K, L

	def get_noise(self):
		return np.random.normal(loc=0., scale=self.noise_scale)

	def step(self, action):
		self._episode_steps += 1
		# print(f"a {action}")
		# reward = -float(self.z2*action**2)
		action = self.action(action)

		for _ in range(self.n_discrete):
			h1 = self.h1 + (- self.a1/self.A1*np.sqrt(2*self.G*self.h1) + self.Kp/self.A1*action)*self.delta_t
			h2 = self.h2 + (self.a1/self.A2*np.sqrt(2*self.G*self.h1) - self.a2/self.A2*np.sqrt(2*self.G*self.h2))*self.delta_t
			self.h1 = np.clip(h1, self.observation_space.low[0], self.observation_space.high[0])
			self.h2 = np.clip(h2, self.observation_space.low[1], self.observation_space.high[1])
		self.h1 += self.get_noise()
		self.h2 += self.get_noise()
		self.h1 = np.clip(self.h1, self.observation_space.low[0], self.observation_space.high[0])
		self.h2 = np.clip(self.h2, self.observation_space.low[1], self.observation_space.high[1])
		
		reward = self.compute_reward(self.h2, self.r)
		if self._episode_steps < self.max_step:
			done = False
		else:
			done = True
			self.last_h1 = self.h1
			self.last_h2 = self.h2
		return self._get_observe(), reward, done, {}

	def close(self):
		self.h1 = None
		self.h2 = None
		self.r = None
		self.qv = None
		self.qvc = None
		self.K = None
		self.L = None

	def seed(self, seed=None):
		self._episode_steps = 0
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	# @title LQR Controller
	def lqr_KL(self,A,B,Z1,Z2, gamma=1.0):
		# print(f"A: {self.A}")
		# print(f"B: {self.B}")
		a = np.sqrt(gamma)*A
		b = np.sqrt(gamma)*B
		r = Z2
		q = Z1
		P = scipy.linalg.solve_discrete_are(a,b,q,r)

		K = gamma* (1/(gamma*B.T@P@B+Z2))@B.T@P@A
		aans = np.array([0,1]) @ scipy.linalg.inv(np.eye(A.shape[0]) - A + B@K)@B
		L= 1./aans
		# print(f"L: {L}")
		return K, L


	def linearize(self, r):
		A = np.zeros((self.m, self.m))
		h1s = (self.a2/self.a1)**2*r
		A[0][0] = -self.a1/self.A1*np.sqrt(self.G/2.0/h1s)
		A[1][0] = self.a1/self.A2*np.sqrt(self.G/2.0/h1s)
		A[1][1] = -self.a2/self.A2*np.sqrt(self.G/2.0/r)

		self.eq_state = np.array([h1s, r])
		self.eq_action = self.a1/self.Kp*np.sqrt(2*9.8*h1s)
		self.eq_r = r

		B = np.zeros((self.m,self.n))
		B[0][0] = self.Kp/self.A1
		
		# self.A = A
		# self.B = B

		self.sys = sys = control.ss(A, B, np.array([[0,1]]), np.zeros((1,1)))

		# discrete the system 
		self.d_sys = d_sys = control.c2d(sys, self.sample_t)
		# # # print(f"d_sys: {d_sys}")
		self.A = d_sys.A.A
		self.B = d_sys.B.A
		

		self.Z1 = np.zeros((2,2))
		self.Z1[1][1] = self.z1
		self.Z2 = self.z2
		self.K, self.L = self.lqr_KL(A=self.A, B=self.B, Z1=self.Z1, Z2=self.Z2)

		self.qv, self.qvc = self.qvalue_weights()
		# change action range
		self.K /= self.eq_action
		return self.K, self.L

	def get_linearize_controller(self, r):
		A = np.zeros((self.m, self.m))
		h1s = (self.a2/self.a1)**2*r
		A[0][0] = -self.a1/self.A1*np.sqrt(self.G/2.0/h1s)
		A[1][0] = self.a1/self.A2*np.sqrt(self.G/2.0/h1s)
		A[1][1] = -self.a2/self.A2*np.sqrt(self.G/2.0/r)
		eq_state = np.array([h1s, r])
		eq_action = self.a1/self.Kp*np.sqrt(2*9.8*h1s)
		eq_r = r
		B = np.zeros((self.m,self.n))
		B[0][0] = self.Kp/self.A1

		# self.A = A
		# self.B = B

		sys = control.ss(A, B, np.array([[0,1]]), np.zeros((1,1)))
		# discrete the system 
		self.d_sys = d_sys = control.c2d(sys, self.sample_t)
		# # print(f"d_sys: {d_sys}")
		A = d_sys.A.A
		B = d_sys.B.A
		
		Z1 = np.zeros((2,2))
		Z1[1][1] = self.z1
		Z2 = self.z2
		K, L = self.lqr_KL(A=A, B=B, Z1=Z1, Z2=Z2)
		return K, L, eq_state, eq_action, eq_r

	def qvalue_weights(self):
		gamma = self.gamma
		Z1 = self.Z1
		# no punishment for action
		Z2 = 0.
		# Z2 = 10.
		A = self.A
		B = self.B
		K = self.K

		q = Z1 + K.T*Z2@K
		a = np.sqrt(gamma)*(A-B@K).T
		P = scipy.linalg.solve_discrete_lyapunov(a, q)
		self.P = P
		
		w_s = Z1 + gamma*A.T@P@A
		w_a = Z2 + gamma*B.T@P@B
		w_sa = gamma*A.T@P@B
		qv = np.zeros((self.m+self.n, self.m+self.n))
		# print(f"qv: {qv.shape}")
		# print(f"w_s: {w_s.shape}")
		qv[:self.m, :self.m] = w_s
		qv[self.m:, self.m:] = w_a
		qv[:self.m, self.m:] = w_sa
		qv[self.m:, :self.m] = w_sa.T
		c = gamma/(1-gamma)*np.trace(self.noise_cov*P)
		return qv, c

	def lqr_value(self, s, a=None):
		s = s.reshape(-1,self.m)
		return -np.diag(s@self.P@s.T + self.qvc).reshape(-1,1)

	def lqr_value_tensor(self, s, a=None):
		with torch.no_grad():
			P = torch.tensor(self.P, device=s.device)
			s = s.reshape(-1,self.m).double()
			return -torch.diag(s@P@s.T + self.qvc).reshape(-1,1)

	def lqr_Qvalue(self, s, a): 
		a = a*self.eq_action

		sa = np.hstack((s.reshape(-1,self.m),a.reshape(-1,self.n)))
		return -np.diag(sa@self.qv@sa.T + self.qvc).reshape(-1,1)

	def lqr_Qvalue_tensor(self, s, a):
		with torch.no_grad():
			a = a*self.eq_action
			qv = torch.tensor(self.qv, device=s.device)
			# a = torch.clip(a,self.action_space.low[0], self.action_space.high[0])
			sa = torch.hstack((s.reshape(-1,self.m),a.reshape(-1,self.n))).double()
			return -torch.diag(sa@qv@sa.T + self.qvc).reshape(-1,1)

	def lqr_Advalue(self,s,a):
		return self.lqr_Qvalue(s,a) - self.lqr_value(s)

	def lqr_Advalue_tensor(self, s,a):
		return self.lqr_Qvalue_tensor(s,a) - self.lqr_value_tensor(s)

	# for linearized controller
	def get_lqr_action(self, state):
		lqr_action = - state@self.K.T \
						 + self.L*(self.r - self.eq_r)
		return  np.clip(lqr_action, self.action_space.low, self.action_space.high)

	def get_lqr_action_tensor(self, state):
		with torch.no_grad():
			K = torch.tensor(self.K, device=state.device, dtype=state.dtype)
			L = torch.tensor(self.L, device=state.device, dtype=state.dtype)
			lqr_action = - state@self.K.T \
						 + self.L*(self.r - self.eq_r)
			return torch.clip(lqr_action, self.action_space.low[0], self.action_space.high[0])

	# for P controller
	def get_P_action(self, state):
		lqr_action = - state@self.K.T \
						 + self.L*self.r
		return  np.clip(lqr_action, self.action_space.low, self.action_space.high)

	def get_P_action_tensor(self, state):
		with torch.no_grad():
			K = torch.tensor(self.K, device=state.device, dtype=state.dtype)
			L = torch.tensor(self.L, device=state.device, dtype=state.dtype)
			lqr_action = - state@self.K.T \
						 + self.L*self.r
			return torch.clip(lqr_action, self.action_space.low[0], self.action_space.high[0])


	def set_reward_type(self, tp):
		if tp in ['sparse','distance','square_distance']:
			self.reward_type = tp

	def compute_reward(self, achieved_goal, desired_goal, info=None):
		"""Compute the step reward. This externalizes the reward function and makes
		it dependent on a desired goal and the one that was achieved. If you wish to include
		additional rewards that are independent of the goal, you can include the necessary values
		to derive it in 'info' and compute it accordingly.
		Args:
			achieved_goal (object): the goal that was achieved during execution
			desired_goal (object): the desired goal that we asked the agent to attempt to achieve
			info (dict): an info dictionary with additional information
		Returns:
			float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
			goal. Note that the following should always hold true:
				ob, reward, done, info = env.step()
				assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
		"""
		# print(f"achieved_goal: {achieved_goal}")
		# print(f"desired_goal: {desired_goal}")
		# print(f"info: {info}")
		d = goal_distance(achieved_goal, desired_goal)
		if self.reward_type =='sparse':
			return -(d > self.distance_threshold).astype(np.float32)
			# print("sparse ===========")
		elif self.reward_type =='distance':
			# print("distance ===========")
			return -d*self.z1
		elif self.reward_type=='square_distance':
			return -d**2*self.z1
		else:
			assert True, "no such reward type for watertank"

	def compute_reward_tensor(self, achieved_goal, desired_goal, info=None):
		# print(f"achieved_goal: {achieved_goal}")
		# print(f"desired_goal: {desired_goal}")
		# print(f"info: {info}")

		with torch.no_grad():
			d = goal_distance_tensor(achieved_goal, desired_goal)
			if self.reward_type =='sparse':
				return -(d > self.distance_threshold).to(dtype=achieved_goal.dtype, device=achieved_goal.device)
			elif self.reward_type =='distance':
				return -d*self.z1
			elif self.reward_type=='square_distance':
				return -d**2*self.z1
			else:
				assert True, "no such reward type for watertank"

class NonLinearWaterTankFixedGoal(NonLinearWaterTank):

	def P_KL(self):
		K, L = self.P_control_K, None
		return K, L

	def _get_m(self):
		return 3

	def _get_observe_linearized(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2, self.r]) - self.eq_state
		return self.state

	def _get_observe_P(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2, self.r])
		return self.state

	def get_lqr_action(self, state):
		# print(f'state: {state}, r: {r}')
		state = state[:self.m]
		lqr_action = - state@self.K.T
		return  np.clip(lqr_action, self.action_space.low, self.action_space.high)

	def get_lqr_action_tensor(self, state):
		state = state[:,:self.m]
		with torch.no_grad():
			K = torch.tensor(self.K, device=state.device, dtype=state.dtype)
			lqr_action = - state@K.T 
			return torch.clip(lqr_action, self.action_space.low[0], self.action_space.high[0])

	# for P controller
	def get_P_action(self, state):
		# print(f'state: {state}, r: {r}')
		state = state[:self.m]
		lqr_action = - state@self.K.T
		return  np.clip(lqr_action, self.action_space.low, self.action_space.high)

	def get_P_action_tensor(self, state):
		state = state[:,:self.m]
		with torch.no_grad():
			K = torch.tensor(self.K, device=state.device, dtype=state.dtype)
			lqr_action = - state@K.T 
			return torch.clip(lqr_action, self.action_space.low[0], self.action_space.high[0])


	def linearize(self, r):
		A = np.zeros((self.m-1, self.m-1))
		h1s = (self.a2/self.a1)**2*r
		A[0][0] = -self.a1/self.A1*np.sqrt(self.G/2.0/h1s)
		A[1][0] = self.a1/self.A2*np.sqrt(self.G/2.0/h1s)
		A[1][1] = -self.a2/self.A2*np.sqrt(self.G/2.0/r)

		self.eq_state = np.array([h1s, r, r])
		self.eq_action = self.a1/self.Kp*np.sqrt(2*9.8*h1s)
		self.eq_r = r

		B = np.zeros((self.m-1,self.n))
		B[0][0] = self.Kp/self.A1
		# print(A.shape)
		self.sys = sys = control.ss(A, B, np.array([[0,1]]), np.zeros((1,1)))
		self.d_sys = d_sys = control.c2d(sys, self.sample_t)

		Atilde = np.zeros((self.m,self.m))
		Atilde[0:self.m-1, 0:self.m-1] = d_sys.A.A
		Atilde[-1][-1] = 1
		self.A = Atilde

		Btilde = np.zeros((self.m,1))
		Btilde[0:self.m-1,:] = d_sys.B.A
		# print("Btilde:\n", Btilde)
		self.B = Btilde

		Z1 = np.zeros((2,2))
		Z1[1][1] = self.z1
		C = np.append(np.diag(Z1), -1).reshape(1,-1)
		# print(C)
		self.Z1 = C.T@C

		self.Z2 = self.z2

		self.K = self.estimate_linear_policy_K(gamma=self.gamma, A=self.A, B=self.B, Z1=self.Z1, Z2=self.Z2)
		# bad controller!
		# self.K[-1] = 0.8*self.K[-1]

		self.qv, self.qvc = self.qvalue_weights()
		return self.K

	def get_linearize_controller(self, r):
		A = np.zeros((self.m-1, self.m-1))
		h1s = (self.a2/self.a1)**2*r
		A[0][0] = -self.a1/self.A1*np.sqrt(self.G/2.0/h1s)
		A[1][0] = self.a1/self.A2*np.sqrt(self.G/2.0/h1s)
		A[1][1] = -self.a2/self.A2*np.sqrt(self.G/2.0/r)
		eq_state = np.array([h1s, r, r])
		eq_action = self.a1/self.Kp*np.sqrt(2*9.8*h1s)
		# eq_r = r
		B = np.zeros((self.m-1,self.n))
		B[0][0] = self.Kp/self.A1
		sys = control.ss(A, B, np.array([[0,1]]), np.zeros((1,1)))
		d_sys = control.c2d(sys, self.sample_t)

		Atilde = np.zeros((self.m,self.m))
		Atilde[0:self.m-1, 0:self.m-1] = d_sys.A.A
		Atilde[-1][-1] = 1

		A = Atilde

		Btilde = np.zeros((self.m,1))
		Btilde[0:self.m-1,:] = d_sys.B.A
		B = Btilde

		Z1 = np.zeros((2,2))
		Z1[1][1] = self.z1
		C = np.append(np.diag(Z1), -1).reshape(1,-1)
		# print(C)
		Z1 = C.T@C
		Z2 = self.z2
		K = self.estimate_linear_policy_K(gamma=self.gamma,A=A, B=B, Z1=Z1, Z2=Z2)
		
		lqr_func = lambda state: np.clip(-(state-eq_state)@K.T + eq_action,
											 self.action_space.low, self.action_space.high)
		return lqr_func

	def get_linearize_controller_tensor(self, r):
		A = np.zeros((self.m-1, self.m-1))
		h1s = (self.a2/self.a1)**2*r
		A[0][0] = -self.a1/self.A1*np.sqrt(self.G/2.0/h1s)
		A[1][0] = self.a1/self.A2*np.sqrt(self.G/2.0/h1s)
		A[1][1] = -self.a2/self.A2*np.sqrt(self.G/2.0/r)
		eq_state = np.array([h1s, r, r])
		eq_action = self.a1/self.Kp*np.sqrt(2*9.8*h1s)
		# eq_r = r
		B = np.zeros((self.m-1,self.n))
		B[0][0] = self.Kp/self.A1
		sys = control.ss(A, B, np.array([[0,1]]), np.zeros((1,1)))
		# d_sys = control.c2d(sys, self.delta_t)

		Atilde = np.zeros((self.m,self.m))
		Atilde[0:self.m-1, 0:self.m-1] = sys.A.A
		Atilde[-1][-1] = 1

		A = Atilde

		Btilde = np.zeros((self.m,1))
		Btilde[0:self.m-1,:] = sys.B.A
		B = Btilde

		Z1 = np.zeros((2,2))
		Z1[1][1] = self.z1
		C = np.append(np.diag(Z1), -1).reshape(1,-1)
		# print(C)
		Z1 = C.T@C
		Z2 = self.z2
		K = self.estimate_linear_policy_K(gamma=self.gamma,A=A, B=B, Z1=Z1, Z2=Z2)

		lqr_func = lambda state: torch.clip(-(state[:,:self.m]-torch.tensor(eq_state, device=state.device, dtype=state.dtype))@torch.tensor(K, device=state.device, dtype=state.dtype).T 
																+ torch.tensor(eq_action, device=state.device, dtype=state.dtype),
											self.action_space.low[0], self.action_space.high[0])
		return lqr_func

	def estimate_linear_policy_K(self, gamma, A, B, Z1, Z2):
		a = np.sqrt(gamma)*A
		b = np.sqrt(gamma)*B
		r = Z2
		q = Z1
		P = scipy.linalg.solve_discrete_are(a,b,q,r)

		K = gamma* (1/(gamma*B.T@P@B+Z2))@B.T@P@A
		return K

	def _observationSpace(self):
		low_state = -np.ones(3)*0
		high_state = np.ones(3)*np.inf
		return spaces.Box(low=low_state,
							high=high_state,
							dtype=np.float32)


class NonLinearWaterTankUniformGoal(NonLinearWaterTankFixedGoal):
	def reset(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		return self._get_observe()


class NonLinearWaterTankUniformGoalIntegrator(NonLinearWaterTankUniformGoal):
	"""docstring for ClassName"""
	n_integrator = 1
	integral_max = 25.
	integral_punish = 0.0
	
	def _observationSpace_P(self):
		low_state = -np.ones(self.m)*0
		high_state = np.ones(self.m)*np.inf
		low_state = np.append(low_state, -self.integral_max)
		high_state = np.append(high_state, self.integral_max)
		return spaces.Box(low=low_state,
							high=high_state,
							dtype=np.float32)

	def _observationSpace_linearized(self):
		low_state = -np.ones(self.m)*0
		high_state = np.ones(self.m)*np.inf
		low_state = np.append(low_state, -self.integral_max)
		high_state = np.append(high_state, self.integral_max)

		return spaces.Box(low=low_state,
							high=high_state,
							dtype=np.float32)

	# for P controller
	def get_P_action(self, state):
		# print(f'state: {state}, r: {r}')
		state = state[:self.m+1]
		lqr_action = - state@self.K.T
		return  np.clip(lqr_action, self.action_space.low, self.action_space.high)

	def get_P_action_tensor(self, state):
		state = state[:,:self.m+1]
		with torch.no_grad():
			K = torch.tensor(self.K, device=state.device, dtype=state.dtype)
			lqr_action = - state@K.T 
			return torch.clip(lqr_action, self.action_space.low[0], self.action_space.high[0])



	def reset(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.integrator = 0.
		return self._get_observe()


	def _get_observe_linearized(self):
		assert self.h1 is not None, "Please reset the env first"
		state = np.array([self.h1, self.h2, self.r]) - self.eq_state
		self.state = np.append(state, self.integrator)
		return self.state

	def _get_observe_P(self):
		assert self.h1 is not None, "Please reset the env first"
		state = np.array([self.h1, self.h2, self.r])
		self.state = np.append(state, self.integrator)
		return self.state


	def step(self, action):
		self._episode_steps += 1
		# print(f"a {action}")
		# reward = -float(self.z2*action**2)
		action = self.action(action)
		for _ in range(self.n_discrete):
			h1 = self.h1 + (- self.a1/self.A1*np.sqrt(2*self.G*self.h1) + self.Kp/self.A1*action)*self.delta_t
			h2 = self.h2 + (self.a1/self.A2*np.sqrt(2*self.G*self.h1) - self.a2/self.A2*np.sqrt(2*self.G*self.h2))*self.delta_t
			self.h1 = np.clip(h1, self.observation_space.low[0], self.observation_space.high[0])
			self.h2 = np.clip(h2, self.observation_space.low[1], self.observation_space.high[1])
		self.h1 += self.get_noise()
		self.h2 += self.get_noise()
		self.h1 = np.clip(self.h1, self.observation_space.low[0], self.observation_space.high[0])
		self.h2 = np.clip(self.h2, self.observation_space.low[1], self.observation_space.high[1])
		
		reward = self.compute_reward(self.h2, self.r)
		if self._episode_steps < self.max_step:
			done = False
		else:
			done = True
			self.last_h1 = self.h1
			self.last_h2 = self.h2
		delta_integrator_after = self.r - self.h2
		integrator = self.integrator + delta_integrator_after
		reward += -self.integral_punish*np.abs(integrator)
		self.integrator = np.clip(integrator, -self.integral_max, self.integral_max)
		return self._get_observe(), reward, done, {}

class NonLinearWaterTankChangingParamUniformGoalIntegrator(NonLinearWaterTankUniformGoalIntegrator):
	metadata = {'render.modes': ['human']}
	def __init__(self,
				 a1=[1,2],	# must be a range!
				 a2=[1,2],	# must be a range!
				 A1=2,
				 A2=2,
				 Kp=[1,2],	# must be a range!
				 G=9.8,
				 z1 = 1,
				 z2 = 0.1,
				 max_step = 500,
				 noise_scale = 0.01,
				 gamma=0.99,
				 seed=None,
				 r = 9.0,
				 N = 100,
				 overflow_cost=-10,
				 n_discrete = 1,
				 sample_t = 0.02,
				 reward_type='distance',
				 controller_type='P',
				 distance_threshold=0.05,
				 linearize_r=9.0,
				 reset_from_last_state=True,
				 P_control_K = np.array([0., 0.4]),
				 P_control_L = np.array([-0.4]),
				 P_max_action = 10.0,
				 ):
		self.a1_range = a1
		self.a2_range = a2
		self.Kp_range = Kp

		self.if_reset_all = True
		# init the value and env
		a1_init, a2_init, Kp_init = self.sample_parameters()
		super().__init__(a1=a1_init,
						 a2=a2_init,
						 A1=A1,
						 A2=A2,
						 Kp=Kp_init,
						 G=G,
						 z1 = z1,
						 z2 = z2,
						 max_step = max_step,
						 noise_scale = noise_scale,
						 gamma=gamma,
						 seed=seed,
						 r = r,
						 N = N,
						 overflow_cost=overflow_cost,
						 n_discrete = n_discrete,
						 sample_t = sample_t,
						 reward_type=reward_type,
						 controller_type=controller_type,
						 distance_threshold=distance_threshold,
						 linearize_r=linearize_r,
						 reset_from_last_state=reset_from_last_state,
						 P_control_K = P_control_K,
						 P_control_L = P_control_L,
						 P_max_action = P_max_action)

	def sample_parameters(self):
		a1 = np.random.uniform(self.a1_range[0], self.a1_range[1])
		a2 = np.random.uniform(self.a2_range[0], self.a2_range[1])
		Kp = np.random.uniform(self.Kp_range[0], self.Kp_range[1])
		return a1, a2, Kp

	def get_changable_parameters(self):
		return self.a1, self.a2, self.Kp

	def reset_changable_parameters(self, a1, a2, Kp):
		self.a1, self.a2, self.Kp = a1, a2, Kp

	def reset_all(self):
		self.a1, self.a2, self.Kp = self.sample_parameters()
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.integrator = 0.
		return self._get_observe()

	def reset_r(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.integrator = 0.
		return self._get_observe()

	def reset(self):
		if self.if_reset_all:
			return self.reset_all()
		else:
			return self.reset_r()

	
class NonLinearWaterTankChangingParamUniformGoal(NonLinearWaterTankUniformGoal):
	metadata = {'render.modes': ['human']}
	def __init__(self,
				 a1=[1,2],	# must be a range!
				 a2=[1,2],	# must be a range!
				 A1=2,
				 A2=2,
				 Kp=[1,2],	# must be a range!
				 G=9.8,
				 z1 = 1,
				 z2 = 0.1,
				 max_step = 500,
				 noise_scale = 0.01,
				 gamma=0.99,
				 seed=None,
				 r = 9.0,
				 N = 100,
				 overflow_cost=-10,
				 n_discrete = 1,
				 sample_t = 0.02,
				 reward_type='distance',
				 controller_type='P',
				 distance_threshold=0.05,
				 linearize_r=9.0,
				 reset_from_last_state=True,
				 P_control_K = np.array([0., 0.4]),
				 P_control_L = np.array([-0.4]),
				 P_max_action = 10.0,
				 ):
		self.a1_range = a1
		self.a2_range = a2
		self.Kp_range = Kp

		self.if_reset_all = True
		# init the value and env
		a1_init, a2_init, Kp_init = self.sample_parameters()
		super().__init__(a1=a1_init,
						 a2=a2_init,
						 A1=A1,
						 A2=A2,
						 Kp=Kp_init,
						 G=G,
						 z1 = z1,
						 z2 = z2,
						 max_step = max_step,
						 noise_scale = noise_scale,
						 gamma=gamma,
						 seed=seed,
						 r = r,
						 N = N,
						 overflow_cost=overflow_cost,
						 n_discrete = n_discrete,
						 sample_t = sample_t,
						 reward_type=reward_type,
						 controller_type=controller_type,
						 distance_threshold=distance_threshold,
						 linearize_r=linearize_r,
						 reset_from_last_state=reset_from_last_state,
						 P_control_K = P_control_K,
						 P_control_L = P_control_L,
						 P_max_action = P_max_action)

	def sample_parameters(self):
		a1 = np.random.uniform(self.a1_range[0], self.a1_range[1])
		a2 = np.random.uniform(self.a2_range[0], self.a2_range[1])
		Kp = np.random.uniform(self.Kp_range[0], self.Kp_range[1])
		return a1, a2, Kp

	def get_changable_parameters(self):
		return self.a1, self.a2, self.Kp

	def reset_changable_parameters(self, a1, a2, Kp):
		self.a1, self.a2, self.Kp = a1, a2, Kp

	def reset_all(self):
		self.a1, self.a2, self.Kp = self.sample_parameters()
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.integrator = 0.
		return self._get_observe()

	def reset_r(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.integrator = 0.
		return self._get_observe()

	def reset(self):
		if self.if_reset_all:
			return self.reset_all()
		else:
			return self.reset_r()


class NonLinearWaterTankChangingParamUniformGoalStacking(NonLinearWaterTankChangingParamUniformGoal):
	"""docstring for NonLinearWaterTankChangingParamUniformGoalStacking"""
	def __init__(self,
				 a1=[1,2],	# must be a range!
				 a2=[1,2],	# must be a range!
				 A1=2,
				 A2=2,
				 Kp=[1,2],	# must be a range!
				 G=9.8,
				 z1 = 1,
				 z2 = 0.1,
				 max_step = 500,
				 noise_scale = 0.01,
				 gamma=0.99,
				 seed=None,
				 r = 9.0,
				 N = 100,
				 overflow_cost=-10,
				 n_discrete = 1,
				 sample_t = 0.02,
				 reward_type='distance',
				 controller_type='P',
				 distance_threshold=0.05,
				 linearize_r=9.0,
				 reset_from_last_state=True,
				 P_control_K = np.array([0., 0.4]),
				 P_control_L = np.array([-0.4]),
				 P_max_action = 10.0,
				 num_stack = 4,
				 ):
		self.num_stack = num_stack
		super(NonLinearWaterTankChangingParamUniformGoalStacking, self).__init__(
						 a1=a1,
						 a2=a2,
						 A1=A1,
						 A2=A2,
						 Kp=Kp,
						 G=G,
						 z1 = z1,
						 z2 = z2,
						 max_step = max_step,
						 noise_scale = noise_scale,
						 gamma=gamma,
						 seed=seed,
						 r = r,
						 N = N,
						 overflow_cost=overflow_cost,
						 n_discrete = n_discrete,
						 sample_t = sample_t,
						 reward_type=reward_type,
						 controller_type=controller_type,
						 distance_threshold=distance_threshold,
						 linearize_r=linearize_r,
						 reset_from_last_state=reset_from_last_state,
						 P_control_K = P_control_K,
						 P_control_L = P_control_L,
						 P_max_action = P_max_action)
		self.frames = deque(maxlen=num_stack)
		if controller_type == 'linearized':
			self.update_state = self.update_state_linearized
		elif controller_type == 'P':
			self.update_state = self.update_state_P

	def _get_m(self):
		return 3*self.num_stack

	def step(self, action):
		self._episode_steps += 1
		# print(f"a {action}")
		# reward = -float(self.z2*action**2)
		action = self.action(action)
		for _ in range(self.n_discrete):
			h1 = self.h1 + (- self.a1/self.A1*np.sqrt(2*self.G*self.h1) + self.Kp/self.A1*action)*self.delta_t
			h2 = self.h2 + (self.a1/self.A2*np.sqrt(2*self.G*self.h1) - self.a2/self.A2*np.sqrt(2*self.G*self.h2))*self.delta_t
			self.h1 = np.clip(h1, self.observation_space.low[0], self.observation_space.high[0])
			self.h2 = np.clip(h2, self.observation_space.low[1], self.observation_space.high[1])
		self.h1 += self.get_noise()
		self.h2 += self.get_noise()
		self.h1 = np.clip(self.h1, self.observation_space.low[0], self.observation_space.high[0])
		self.h2 = np.clip(self.h2, self.observation_space.low[1], self.observation_space.high[1])
		
		reward = self.compute_reward(self.h2, self.r)
		if self._episode_steps < self.max_step:
			done = False
		else:
			done = True
			self.last_h1 = self.h1
			self.last_h2 = self.h2

		self.update_state()
		self.frames.append(self.state)


		return self._get_observe(), reward, done, {}

	def update_state_P(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2, self.r])

	def update_state_linearized(self):
		assert self.h1 is not None, "Please reset the env first"
		self.state = np.array([self.h1, self.h2, self.r]) - self.eq_state
		

	def _get_observe_linearized(self):
		assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
		return np.array(self.frames).reshape(1,-1)[0]

	def _get_observe_P(self):
		assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
		return np.array(self.frames).reshape(1,-1)[0]

	def reset_all(self):
		self.a1, self.a2, self.Kp = self.sample_parameters()
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.update_state()
		[self.frames.append(self.state) for _ in range(self.num_stack)]
		return self._get_observe()

	def reset_r(self):
		if self.reset_from_last_state:
			if self.last_h1 is None:
				self.h1 = np.abs(np.random.randn())*0.1
				self.h2 = np.abs(np.random.randn())*0.1
			else:
				self.h1 = self.last_h1
				self.h2 = self.last_h2
		else:
			self.h1, self.h2 = tuple(np.random.uniform(0.,10., 2))
		self.r = np.random.uniform(0.,10.)

		self._episode_steps = 0
		self.update_state()
		[self.frames.append(self.state) for _ in range(self.num_stack)]
		
		return self._get_observe()

	def reset(self):
		if self.if_reset_all:
			return self.reset_all()
		else:
			return self.reset_r()
		