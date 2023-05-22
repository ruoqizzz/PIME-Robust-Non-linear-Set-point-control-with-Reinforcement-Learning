import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.optimize import fsolve
import control

def goal_distance(goal_a, goal_b):
	if isinstance(goal_a, np.ndarray):
		return np.linalg.norm(goal_a-goal_b).reshape(-1,1)
	else:
		return np.linalg.norm(goal_a-goal_b)

class PH1D(gym.Env):
	metadata = {'render.modes': ['human']}
	dim=1
	'''
	The pH neutralization process is a highly nonlinear process. 
		It concers neutralization titration of waste water, 
		containing ammonia (NH3) and sodium hydroxide (NaOH) by hydrocholoric acid (HCI) 

	The model is highly nonlinear due to the implicit output equation (titration curve)

	More details: [paper] Recursive identification based on the nonlinear wiener model
	'''
	def __init__(self, 
				 # V = 0.5,			# volume of tank [m*m*m]
				 # qww = 0.005,		# process flow [m*m*m/s]
				 # qc = 0.0001,		# control flow [m*m*m/s]
				 qww_V = 0.01,
				 qc_V = 0.0002,
				 kw = 1e-14,		# ion product of water
				 kchem = 5.6e-10,	# dissociation constant of NH3
				 ka = 0.5e-5,		# dissosiation constant of weak acid
				 MNaOH = 0.01,		# concentration of sodium hydroxide before reaction
				 MHA = 0.005,		# concentration of weak acid before reaction
				 MNH3 = 0.01,		# concentration of ammonia before reaction
				 MHCl=np.arange(0.,0.06,step=0.00001),	# amount of hydrocloric acid used
				 r=7.0,		# goal
				 n_discrete = 200,
				 sample_t = 20,
				 reset_from_last_state=False,
				 reward_type='distance',
				 distance_threshold=0.05,
				 P_control_K = np.array([1, 1]),
				 P_control_L = np.array([-0.4]),
				 action_punishment=0.,
				 action_change_punishment=0.,
				 max_episode_steps=200,
				 seed=None,):
		super(PH1D, self).__init__()
		# self.V = V
		# self.qww = qww
		# self.qc = qc
		self.qww_V = qww_V
		self.qc_V = qc_V
		
		self.sample_t = sample_t
		self.n_discrete = n_discrete
		self.delta_t = sample_t / n_discrete

		self.update_system()

		self.kw = kw
		self.kchem = kchem
		self.ka = ka
		self.MNaOH = MNaOH
		self.MHA = MHA
		self.MNH3 = MNH3
		self.MHCl = MHCl

		# cubic equation for [H+].
		# See more details in thesis A.2.10
		pH=0.*MHCl
		H=1e-14/MNaOH
		for i in range(len(MHCl)):
			ak = MNH3-MHCl[i]+MNaOH+kchem+ka
			bk = (kchem+ka)*MNaOH-(kchem+ka)*MHCl[i]-kw+MNH3*ka+kchem*ka-ka*MHA
			ck = MNaOH*kchem*ka-kw*(ka+kchem)-MHCl[i]*kchem*ka-ka*kchem*MHA
			dk = -kchem*ka*kw
			for j in range(5):
				H=np.abs(H-(H**4+ak*H**3+bk*H**2+ck*H+dk)/(4*H**3+3*ak*H**2+2*bk*H+ck))
				pH[i]=-1*np.log10(H)
		self.pH = pH

		self.r = r


		self.m = self._get_m() 	# [ph, r]
		self.n = 1
		self.observation_space = self._observationSpace()
		self.action_space = self._actionSapce()



		self.reset_from_last_state  = reset_from_last_state

		self.max_episode_steps = max_episode_steps
		self.reward_type = reward_type
		self.distance_threshold = distance_threshold
		self.reset_from_last_state = reset_from_last_state
		self.last_state = None

		self.K = P_control_K
		self.L = P_control_L
		self.action_punishment = action_punishment
		self.action_change_punishment = action_change_punishment
		self.seed(seed)
		self.if_reset_all = True

	def set_reset_all(self, if_reset_all):
		self.if_reset_all  = if_reset_all

	def update_system(self):
		self.F_con = self.a0con = np.array([1, self.qww_V])	# continuous time system denominator
		self.B_con = self.b0con = np.array([0, self.qc_V])	# continuous time system numerator

		self.sys = control.tf2ss(self.b0con, self.a0con)
		self.dsys = control.c2d(self.sys, self.sample_t)

		self.gain = self.qc_V/(1+self.qww_V)

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _get_m(self):
		return 1

	def _get_observe(self):
		return self.y

	def _observationSpace(self):
		low_state = -np.ones(self.m)*np.inf
		high_state = np.ones(self.m)*np.inf
		return spaces.Box(low=low_state,
						  high=high_state,
						  dtype=np.float32)
	
	def _actionSapce(self):
		self.min_action = -1
		self.max_action = 1

		# self.low = np.array([0., 0.])
		# self.high = np.array([25, 1])
		self.low = 0.
		self.high = 1.5
		low_action = np.ones(self.n)*-1.
		high_action = np.ones(self.n)*1.
		return spaces.Box(low=low_action,
								high=high_action,
								dtype=np.float32)


	def action(self, action):
		action = self.low + (self.high - self.low) * (
			(action - self.min_action) / (self.max_action - self.min_action)
		)
		return action

	def rescale_action(self, action):
		action = self.min_action + (self.max_action - self.min_action) * (
			(action - self.low) / (self.high - self.low)
		)
		return action

	def step(self, action):
		action = np.clip(action, self.min_action, self.max_action)
		delta_u = action - self.last_action if self._episode_steps!=0 else 0
		self.last_action = action
		self._episode_steps += 1
		action = self.action(action)
		u = action
		A = self.dsys.A.item()
		B = self.dsys.B.item()
		self.state = A*self.state + B*action
		# output
		y = self.observe_state(self.state)
		self.y = y
		reward = self.compute_reward(y, self.r)
		reward -= self.action_punishment*np.linalg.norm(action, ord=2)
		reward -= self.action_change_punishment*np.linalg.norm(delta_u, ord=2)
		if self._episode_steps >= self.max_episode_steps:
			self.last_state = self.state
		return self._get_observe(), reward, False, {}

	def observe_state(self, state):
		idx = np.argwhere(self.MHCl>=np.around(self.dsys.C.item()*state,5))[0]
		return self.pH[idx].item()

	def reset(self):
		# state: [Wa, Wb] reaction invariants for the effluent solution (charge balance, balance on the carbonate ion)
		if self.reset_from_last_state and self.last_state is not None:
			self.state = self.last_state
		else:
			self.state = self.np_random.uniform(low=0, high=50)

		self._episode_steps = 0
		self.y = self.observe_state(self.state)
		return self._get_observe()

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
		d = goal_distance(achieved_goal, desired_goal)
		if self.reward_type =='sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		elif self.reward_type =='distance':
			return -d
		elif self.reward_type == 'square_distance':
			return -d**2
		else:
			assert True, "no such reward type"

	def get_linear_action(self, state=None):
		if state is None:
			return -self._get_observe()@self.K.T
		else:
			return -state@self.K.T

	def set_state(self, state):
		self.state = state
		self.y = self.observe_state(self.state)
		return self._get_observe()

	def set_r(self, r):
		self.r = r
		return self._get_observe()

	# def set_V(self, V):
	# 	self.V = V

	# def set_qc(self, qc):
	# 	self.qc = qc

	# def set_qww(self, qww):
	# 	self.qww = qww

	def set_qww_V(self, qww_V):
		self.qww_V = qww_V

	def set_qc_V(self, qc_V):
		self.qc_V = qc_V

	# def set_params(self, qww, qc, V):
	# 	self.qc = qc
	# 	self.qww = qww
	# 	self.V = V
	# 	return self._get_observe()

	def set_params(self, qww_V, qc_V):
		self.qww_V = qww_V
		self.qc_V = qc_V


	def get_changable_parameters(self):
		# return self.qww, self.qc, self.V
		return self.qww_V, self.qc_V

	def set_max_episode_steps(self, step):
		self.set_max_episode_steps = step
		return step
		
class PH1DFixedGoal(PH1D):
	def _get_m(self):
		return 2

	def _get_observe(self):
		return np.array([self.y, self.r])


class PH1DUniformGoal(PH1DFixedGoal):
	def reset(self):
		# state: [Wa, Wb] reaction invariants for the effluent solution (charge balance, balance on the carbonate ion)
		if self.reset_from_last_state and self.last_state is not None:
			self.state = self.last_state
		else:
			self.state = self.np_random.uniform(low=0, high=50)
		self.r = self.np_random.uniform(3., 11.)
		self._episode_steps = 0
		self.y = self.observe_state(self.state)
		return self._get_observe()


class PH1DUniformGoalIntegrator(PH1DUniformGoal):
	n_integrator = 1
	integral_max = 25.
	integral_punish = 0.0

	def _get_m(self):
		return 3

	def _get_observe(self):
		return np.array([self.y, self.r, self.integrator])

	def reset(self):
		# state: [Wa, Wb] reaction invariants for the effluent solution (charge balance, balance on the carbonate ion)
		if self.reset_from_last_state and self.last_state is not None:
			self.state = self.last_state
		else:
			self.state = self.np_random.uniform(low=0, high=50)
		self.r = self.np_random.uniform(3., 11.)
		self._episode_steps = 0
		self.y = self.observe_state(self.state)
		self.integrator = 0.
		return self._get_observe()

	def step(self, action):
		action = np.clip(action, self.min_action, self.max_action)
		delta_u = action - self.last_action if self._episode_steps!=0 else 0
		# print(f"delta_u: {delta_u}")
		self.last_action = action
		self._episode_steps += 1
		action = self.action(action)
		u = action
		A = self.dsys.A.item()
		B = self.dsys.B.item()
		self.state = A*self.state + B*action
		# output
		y = self.observe_state(self.state)
		self.y = y
		reward = self.compute_reward(y, self.r)
		# print(action)
		reward -= self.action_punishment*np.abs(float(action))
		reward -= self.action_change_punishment*np.linalg.norm([delta_u], ord=2)
		# print(f"delta_u punishment: {delta_u}")
		delta_integrator_after = self.r - y
		integrator = self.integrator + delta_integrator_after
		self.integrator = np.clip(integrator, -self.integral_max, self.integral_max)
		
		reward += -self.integral_punish*np.abs(integrator)
		
		if self._episode_steps >= self.max_episode_steps:
			self.last_state = self.state

		return self._get_observe(), reward, False, {}


class PH1DChangingParamUniformGoalIntegrator(PH1DUniformGoalIntegrator):
	"""docstring for PHChangingParamUniformGoalIntegrator"""
	def __init__(self, 
	# 			 V = [0.4,0.6],			# volume of tank [m*m*m]
	# 			 qww = [0.002,0.008],		# process flow [m*m*m/s]
	# 			 qc = [0.00005,0.00015],		# control flow [m*m*m/s]
				 qww_V=[0.005,0.015],		# qww=0.005; V=0.5;qc=0.0001; as defualt
				 qc_V=[0.0015, 0.0025],	# gain range: 0.0025/(1+0.005)=0.00249 ~ 0.0015/1.015 = 0.00148 (max/min=1.68)
				 kw = 1e-14,		# ion product of water
				 kchem = 5.6e-10,	# dissociation constant of NH3
				 ka = 0.5e-5,		# dissosiation constant of weak acid
				 MNaOH = 0.01,		# concentration of sodium hydroxide before reaction
				 MHA = 0.005,		# concentration of weak acid before reaction
				 MNH3 = 0.01,		# concentration of ammonia before reaction
				 MHCl=np.arange(0.,0.2,step=0.00001),	# amount of hydrocloric acid used
				 r=7.0,		# goal
				 n_discrete = 200,
				 sample_t = 20,
				 reset_from_last_state=False,
				 reward_type='distance',
				 distance_threshold=0.05,
				 P_control_K = np.array([1, 1]),
				 P_control_L = np.array([-0.4]),
				 action_punishment=0.,
				 action_change_punishment=0.,
				 max_episode_steps=200,
				 seed=None):
		# self.V_range = V
		# self.qww_range = qww
		# self.qc_range = qc
		self.qww_Vrange = qww_V
		self.qc_Vrange = qc_V
		
		qww_V_init, qc_V_init = self.sample_parameters()
		super().__init__(qww_V = qww_V_init,			
						 qc_V = qc_V_init,
		# 				 qww = qww_init,		# process flow [m*m*m/s]
		# 				 qc = qc_init,		# control flow [m*m*m/s]
						 kw = kw,		# ion product of water
						 kchem = kchem,	# dissociation constant of NH3
						 ka = ka,		# dissosiation constant of weak acid
						 MNaOH = MNaOH,		# concentration of sodium hydroxide before reaction
						 MHA = MHA,		# concentration of weak acid before reaction
						 MNH3 = MNH3,		# concentration of ammonia before reaction
						 MHCl=MHCl,	# amount of hydrocloric acid used
						 r=r,		# goal
						 n_discrete = n_discrete,
						 sample_t = sample_t,
						 reset_from_last_state=reset_from_last_state,
						 reward_type=reward_type,
						 distance_threshold=distance_threshold,
						 P_control_K = P_control_K,
						 P_control_L = P_control_L,
						 action_punishment = action_punishment,
						 action_change_punishment=action_change_punishment,
						 max_episode_steps=max_episode_steps,
						 seed=seed)
		
	def sample_parameters(self):
		return np.random.uniform(*self.qww_Vrange), np.random.uniform(*self.qc_Vrange)

	def reset_all(self):
		self.qww_V, self.qc_V = self.sample_parameters()
		self.update_system()

		# self.state: 1xm numpy array
		if self.reset_from_last_state and self.last_state is not None:
			self.state = self.last_state
		else:
			self.state = self.np_random.uniform(low=0, high=50)

		self.y = self.observe_state(self.state)
		self._episode_steps = 0
		self.r = self.np_random.uniform(3., 11.)
		self.integrator = 0.
		return self._get_observe()

	def reset_r(self):
		# self.state: 1xm numpy array
		if self.reset_from_last_state and self.last_state is not None:
			self.state = self.last_state
		else:
			self.state = self.np_random.uniform(low=0, high=50)

		self.y = self.observe_state(self.state)
		self._episode_steps = 0
		self.r = self.np_random.uniform(3., 11.)
		self.integrator = 0.
		return self._get_observe()

	def reset(self):
		if self.if_reset_all:
			return self.reset_all()
		else:
			return self.reset_r()


class PH1DChangingParamUniformGoalIntegrator_NoBound(PH1DChangingParamUniformGoalIntegrator):
	def step(self, action):
		action = np.clip(action, self.min_action, self.max_action)
		delta_u = action - self.last_action if self._episode_steps!=0 else 0
		# print(f"delta_u: {delta_u}")
		self.last_action = action
		self._episode_steps += 1
		action = self.action(action)
		u = action
		A = self.dsys.A.item()
		B = self.dsys.B.item()
		self.state = A*self.state + B*action
		# output
		y = self.observe_state(self.state)
		self.y = y
		reward = self.compute_reward(y, self.r)
		# print(action)
		reward -= self.action_punishment*np.abs(float(action))
		reward -= self.action_change_punishment*np.linalg.norm([delta_u], ord=2)
		# print(f"delta_u punishment: {delta_u}")
		delta_integrator_after = self.r - y
		# print(f'delta_integrator_after: {delta_integrator_after}')
		self.integrator = self.integrator + delta_integrator_after
		# self.integrator = np.clip(integrator, -self.integral_max, self.integral_max)
		
		reward += -self.integral_punish*np.abs(self.integrator)
		
		if self._episode_steps >= self.max_episode_steps:
			self.last_state = self.state

		return self._get_observe(), reward, False, {}
class PH1DChangingParamUniformGoal(PH1DChangingParamUniformGoalIntegrator):
	"""docstring for PHChangingParamUniformGoalIntegrator"""
	def _get_m(self):
		return 2

	def _get_observe(self):
		return np.array([self.y, self.r])
