import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pickle
import time

import gym
import gym_control

from elegantrl import logger
from elegantrl.logger import Figure

from copy import deepcopy

def save_pickle(obj, fn):
	f = open(fn, 'wb')
	pickle.dump(obj, f)
	f.close()

def test_policy(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	state = env_eval.set_state(0.,0.)
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_


	# reverse_actions = lambda actions: env_eval.action(actions.reshape(-1, env_eval.action_space.shape[0])).flatten()
	# if if_lqr:
	# 	return np.array(xs), np.array(refs), reverse_actions(np.array(actions)+np.array(lqr_actions)), totals
	# else:
	# 	# print("no adding")
	# 	return np.array(xs), np.array(refs), reverse_actions(np.array(actions)), totals
	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions), totals
		

def test_policy_uniform(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_state(0.,0.)
	state = env_eval.set_r(2.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
		
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(6.)
	# here assume full observation of state
	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(9)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(4)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(1.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_


	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions), totals


def test_policy_uniform_integrator(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	integrators = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_state(0.,0.)
	state = env_eval.set_r(3.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(6.)
	# here assume full observation of state
	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(9)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(4)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(2.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(integrators), np.array(actions), totals
	
def test_watertanklqr(env_eval, agent, log_path, if_uniform=False):
	env_eval2 = deepcopy(env_eval)
	# print(f'state{state}')
	test_ = test_policy_uniform if if_uniform else test_policy
	# print(f"test_ {test_}")
	deterministic_act = lambda x: agent.select_action(x ,if_deterministic=True)
	xs_agent, refs_agent, actions_agent, totals_agent = test_(env_eval, deterministic_act, if_lqr=True)
	xs_linear, refs_linear, actions_linear, totals_linear = test_(env_eval2, lambda x: [0.], if_lqr=True)

	plt.clf()
	plt.plot(xs_agent[:,0], label='agent')
	plt.plot(xs_linear[:,0], label='linear')
	plt.plot(refs_linear, label='ref-linear')
	plt.plot(refs_agent, label='ref-agent')
	plt.legend()
	pltfn = os.path.join(log_path, 'first_tank.png')
	plt.savefig(pltfn,dpi=300)
	plt.close()

	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/first_tank": figure}, key_excluded={"figure": ()})

	plt.clf()
	plt.plot(xs_agent[:,1], label='agent')
	plt.plot(xs_linear[:,1], label='linear')
	plt.plot(refs_linear, label='ref-linear')
	plt.plot(refs_agent, label='ref-agent')
	plt.legend()
	pltfn = os.path.join(log_path, 'second_tank.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close()
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig2, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/second_tank": figure}, key_excluded={"figure": ()})

	plt.clf()
	# plt.plot(refs_agent, label='agent_ref')
	plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close()
	
	plt.clf()
	
	plt.plot(totals_agent, label='agent')
	plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)
	plt.close()
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig3, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/total_rewards": figure}, key_excluded={"figure": ()})
	plt.close('all')


def test_policy_realtank_uniform(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_state_zero()

	state = env_eval.set_r(3.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(6.)
	# here assume full observation of state	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(9.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(4.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(2.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions), totals

def test_policy_realtank_uniform_integrator(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	integrators = []
	actions = []
	env_eval.reset()
	env_eval.set_state_zero()

	state = env_eval.set_r(3.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:2])
		integrators.append(env_eval.state[-1])
		refs.append(env_eval.state[2])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(6.)
	# here assume full observation of state	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(9.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(4.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(2.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:2])
		refs.append(env_eval.state[2])
		integrators.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(integrators), np.array(actions), totals

def test_observer_policy_realtank_uniform(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_state_zero()

	state = env_eval.set_r(3.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(6.)
	# here assume full observation of state	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_r(9.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(4.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_r(2.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions), totals

def test_realwatertank_integrator(env_eval, agent, log_path, if_uniform=False):
	# print(f"env_eval {env_eval}")
	# env_eval2 = deepcopy(env_eval)
	# print(f'state{state}')
	agent.save_load_model(log_path, if_save=True)
	deterministic_act = lambda x: agent.select_action(x ,if_deterministic=True)
	test_ = test_policy_realtank_uniform_integrator
	xs_agent, refs_agent, integrators_agent, actions_agent, totals_agent = test_(env_eval, deterministic_act)
	# xs_linear, refs_linear, integrators_linear, actions_linear, totals_linear = test_(env_eval, env_eval.get_linear_action)

	plt.clf()
	plt.plot(xs_agent[:,0], label='agent')
	plt.plot(refs_agent, label='agent_ref')
	# plt.plot(xs_linear[:,0], label='linear')
	# plt.plot(refs_agent, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'first_tank.png')
	plt.savefig(pltfn,dpi=300)

	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/first_tank": figure}, key_excluded={"figure": ()})
	plt.close('all')

	plt.clf()
	plt.plot(xs_agent[:,1], label='agent')
	plt.plot(refs_agent, label='ref')
	# plt.plot(xs_linear[:,1], label='linear')
	# plt.plot(refs_linear, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'second_tank.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig2, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/second_tank": figure}, key_excluded={"figure": ()})
	save_pickle(xs_agent, os.path.join(log_path, 'xs_agent.pkl'))
	save_pickle(refs_agent, os.path.join(log_path, 'refs_agent.pkl'))
	# save_pickle(xs_linear, os.path.join(log_path, 'xs_linear.pkl'))

	plt.clf()
	plt.plot(integrators_agent, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(integrators_linear, label='linear')
	plt.plot([0.]*len(integrators_agent), 'k--')
	plt.legend()
	pltfn = os.path.join(log_path, 'integrator.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig2, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/second_tank": figure}, key_excluded={"figure": ()})
	save_pickle(integrators_agent, os.path.join(log_path, 'integrators_agent.pkl'))
	# save_pickle(xs_linear, os.path.join(log_path, 'xs_linear.pkl'))


	plt.clf()
	plt.plot(actions_agent, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	save_pickle(actions_agent, os.path.join(log_path, 'actions_agent.pkl'))
	
	plt.clf()
	
	plt.plot(totals_agent, label='agent')
	# plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)
	plt.close('all')
	save_pickle(totals_agent, os.path.join(log_path, 'totals_agent.pkl'))
	# save_pickle(totals_linear, os.path.join(log_path, 'totals_linear.pkl'))

def test_realwatertank(env_eval, agent, log_path, if_uniform=False):
	# print(f"env_eval {env_eval}")
	# env_eval2 = deepcopy(env_eval)
	# print(f'state{state}')
	agent.save_load_model(log_path, if_save=True)
	deterministic_act = lambda x: agent.select_action(x ,if_deterministic=True)
	test_ = test_policy_realtank_uniform
	xs_agent, refs_agent, actions_agent, totals_agent = test_(env_eval, deterministic_act)
	# xs_linear, refs_linear, actions_linear, totals_linear = test_(env_eval, env_eval.get_linear_action)

	plt.clf()
	plt.plot(xs_agent[:,0], label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(xs_linear[:,0], label='linear')
	plt.plot(refs_agent, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'first_tank.png')
	plt.savefig(pltfn,dpi=300)

	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/first_tank": figure}, key_excluded={"figure": ()})
	plt.close('all')

	plt.clf()
	plt.plot(xs_agent[:,1], label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(xs_linear[:,1], label='linear')
	plt.plot(refs_agent, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'second_tank.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig2, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/second_tank": figure}, key_excluded={"figure": ()})
	save_pickle(xs_agent, os.path.join(log_path, 'xs_agent.pkl'))
	# save_pickle(xs_linear, os.path.join(log_path, 'xs_linear.pkl'))

	plt.clf()
	plt.plot(actions_agent, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	save_pickle(actions_agent, os.path.join(log_path, 'actions_agent.pkl'))
	
	plt.clf()
	
	plt.plot(totals_agent, label='agent')
	# plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)
	plt.close('all')
	save_pickle(totals_agent, os.path.join(log_path, 'totals_agent.pkl'))
	# save_pickle(totals_linear, os.path.join(log_path, 'totals_linear.pkl'))


def test_policy_uniform_observer(env_eval, policy, if_lqr=False):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_state(0.,0.)
	state = env_eval.set_r(3.)

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(6.)
	# here assume full observation of state
	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(9)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(4)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	old_h1, old_h2 = env_eval.h1, env_eval.h2
	env_eval.reset()
	env_eval.set_state(old_h1, old_h2)
	state = env_eval.set_r(2.)
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.state[:1])
		refs.append(env_eval.state[-1])
		lqr_actions.append(env_eval.get_linear_action(state).item())
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	if if_lqr:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[1:2], np.array(refs)+env_eval.eq_r, np.array(actions)+np.array(lqr_actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions)+np.array(lqr_actions), totals
	else:
		if hasattr(env_eval, 'eq_state'):
			return np.array(xs)+env_eval.eq_state[1:2], np.array(refs)+env_eval.eq_r, np.array(actions), totals
		else:
			return np.array(xs), np.array(refs), np.array(actions), totals



def test_watertank(env_eval_original, agent, log_path, if_uniform=False):
	# print(f"env_eval {env_eval}")
	env_eval = deepcopy(env_eval_original)
	env_eval2 = deepcopy(env_eval_original)
	if hasattr(env_eval_original, 'integral_punish'):
		env_eval.integral_punish = 0.
		env_eval2.integral_punish = 0.
	# print(f'state{state}')
	deterministic_act = lambda x: agent.select_action(x ,if_deterministic=True)
	test_ = test_policy_uniform if if_uniform else test_policy
	xs_agent, refs_agent, actions_agent, totals_agent = test_(env_eval, deterministic_act)
	xs_linear, refs_linear, actions_linear, totals_linear = test_(env_eval2, env_eval2.get_linear_action)

	plt.clf() 
	plt.plot(xs_agent[:,0], label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	plt.plot(xs_linear[:,0], label='linear')
	plt.plot(refs_linear, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'first_tank.png')
	plt.savefig(pltfn,dpi=300)

	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/first_tank": figure}, key_excluded={"figure": ()})
	plt.close('all')

	plt.clf()
	plt.plot(xs_agent[:,1], label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	plt.plot(xs_linear[:,1], label='linear')
	plt.plot(refs_linear, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, 'second_tank.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	# if tensorboard_log is not None:
	# 	figure = Figure(figure=fig2, close=True)
	# 	writer = logger.make_output_format("tensorboard", tensorboard_log)
	# 	writer.write({"figure/second_tank": figure}, key_excluded={"figure": ()})

	plt.clf()
	plt.plot(actions_agent, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	
	plt.clf()
	
	plt.plot(totals_agent, label='agent')
	plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)
	plt.close('all')


def test_policy_reacher_uniform(env_eval, policy, if_lqr=False):
	refs = []
	diffs = []
	xs = []
	integrators = []
	totals = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	env_eval.reset()
	env_eval.set_goal(np.array([0,0.14]))
	state = env_eval._get_obs()

	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.get_fingertip_pos())
		refs.append(env_eval.get_goal())
		diffs.append(state[8:10])
		integrators.append(state[-2:])
		lqr_actions.append(env_eval.get_linear_action(state))
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_goal(np.array([0.07, 0.09]))
	# here assume full observation of state
	
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.get_fingertip_pos())
		refs.append(env_eval.get_goal())
		diffs.append(state[8:10])
		integrators.append(state[-2:])
		lqr_actions.append(env_eval.get_linear_action(state))
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_
	env_eval.reset()
	state = env_eval.set_goal(np.array([-0.14, 0]))
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.get_fingertip_pos())
		refs.append(env_eval.get_goal())
		diffs.append(state[8:10])
		integrators.append(state[-2:])
		lqr_actions.append(env_eval.get_linear_action(state))
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	env_eval.reset()
	state = env_eval.set_goal(np.array([0.09, 0.07]))
	for n in range(env_eval.max_step):
		states_his.append(state)
		action =  policy(state)[0]
	#     print(action)
		actions.append(action)
		xs.append(env_eval.get_fingertip_pos())
		refs.append(env_eval.get_goal())
		diffs.append(state[8:10])
		integrators.append(state[-2:])
		lqr_actions.append(env_eval.get_linear_action(state))
		# print(action)
		state_, reward, done, _ = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
	#     print(done)
		# if done:
		#   break
		state = state_

	return np.array(diffs), np.array(refs), np.array(integrators), np.array(actions), totals

params_ph = {
	# qc qww V
	1: [[0.005, 0.0025],
		[0.005, 0.0015],
		[0.015, 0.0025],
		[0.015, 0.0015],
		[0.001, 0.002],
		[0.001, 0.0022],
		[0.001, 0.0018],
		[0.0007, 0.002],
		[0.0013, 0.002]
		 ],

	2: [],

}

def test_ph_policy(env_eval, policy):
	refs = []
	xs = []
	totals = []
	total_reward = 0.
	totals_cost = []
	total_cost = 0.
	# actions = []
	states_his = []
	actions = []
	lqr_actions = []

	
	m = env_eval.dim
	last_state = np.zeros(m)

	env_eval.reset()
	state = env_eval.set_state(last_state)

	for n in range(env_eval.max_episode_steps):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(state[:m])
		refs.append(state[-1])
		state_, reward, done, info = env_eval.step(action)
		# print('reward: ',reward)
		total_reward += reward
		# print(total_reward)
		totals.append(total_reward)
		
		total_cost += info.get('cost', 0.)
		totals_cost.append(total_cost)
		# if done:
		#   break
		state = state_
	
	return np.array(xs), np.array(ys), np.array(refs), np.array(actions), totals, totals_cost


def test_ph_policy_uniform(env_eval, policy):
	refs = []
	xs = []
	ys = []
	totals = []
	total_reward = 0.
	totals_cost = []
	total_cost = 0.
	# actions = []
	states_his = []
	actions = []
	lqr_actions = []

	
	m = env_eval.dim
	last_state = np.zeros(m)

	for r in [10., 6, 3, 8, 5]:
	# for r in [8,8,8,8,8,8,8]:
		env_eval.reset()
		print(env_eval.get_changable_parameters())
		env_eval.set_state(last_state)
		state = env_eval.set_r(r)
		for n in range(env_eval.max_episode_steps):
			states_his.append(state)
			action =  policy(state)[0]
			actions.append(action)
			xs.append(state[:m])
			refs.append(env_eval.r)
			ys.append(env_eval.y)
			state_, reward, done, info = env_eval.step(action)
			# print('reward: ',reward)
			total_reward += reward
			# print(total_reward)
			totals.append(total_reward)
			
			total_cost += info.get('cost', 0.)
			totals_cost.append(total_cost)
			# if done:
			#   break
			state = state_
		
		last_state = env_eval.state

	return np.array(xs), np.array(ys), np.array(refs), np.array(actions), totals, totals_cost



def test_ph_policy_integrator(env_eval, policy):
	refs = []
	xs = []
	ys = []
	totals = []
	integrators = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	lqr_actions = []
	totals_cost = []
	total_cost = 0
	m = env_eval.dim
	last_state = np.zeros(m)

	env_eval.reset()
	state = env_eval.set_state(last_state)
	# print(f"state {state}")
	lqr_actions = []
	for n in range(env_eval.max_episode_steps):
		states_his.append(state)
		action =  policy(state)[0]
		actions.append(action)
		xs.append(state[:m])
		refs.append(env_eval.r)	# for water tank or 2-state env
		ys.append(env_eval.y)	# for water tank or 2-state env
		integrators.append(env_eval.integrator)
		lqr_actions.append(env_eval.get_linear_action(state).item())
		state_, reward, done, info = env_eval.step(action)
		total_reward += float(reward)
		total_cost += info.get('cost', 0.)
		totals.append(total_reward)
		# if done:
		#   break
		state = state_
	return np.array(ys), np.array(refs), np.array(integrators), np.array(actions), totals, totals_cost


def test_ph_policy_uniform_integrator(env_eval, policy):
	refs = []
	xs = []
	ys = []
	totals = []
	integrators = []
	total_reward = 0.
	# actions = []
	states_his = []
	actions = []
	totals_cost = []
	total_cost = 0.

	m = env_eval.dim
	last_state = np.zeros(m)

	for r in [10., 6, 3, 8, 5]:
		env_eval.reset()
		env_eval.set_state(last_state)
		state = env_eval.set_r(r)

		# print(f"state {state}")
		for n in range(env_eval.max_episode_steps):
			states_his.append(state)
			action =  policy(state)[0]
			actions.append(action)
			xs.append(state[:m])
			refs.append(env_eval.r)	# for water tank or 2-state env
			ys.append(env_eval.y)	# for water tank or 2-state env
			integrators.append(env_eval.integrator)
			state_, reward, done, info = env_eval.step(action)
			total_reward += float(reward)
			total_cost += info.get('cost', 0.)
			totals.append(total_reward)
			# if done:
			#   break
			state = state_
		last_state = env_eval.state
	return np.array(ys), np.array(refs), np.array(integrators), np.array(actions), totals, totals_cost

def test_ph(env_eval_original, agent, log_path, if_uniform=False):
	os.makedirs(log_path, exist_ok=True)

	env_eval = env_eval_original
	env_eval2 = env_eval_original
	if hasattr(env_eval_original, 'integral_punish'):
		env_eval.integral_punish = 0.
		env_eval2.integral_punish = 0.
	# print(f'state{state}')
	deterministic_act = lambda x: agent.select_action(x , if_deterministic=True)
	test_ = test_ph_policy_uniform if if_uniform else test_ph_policy
	xs_agent, ys_agent, refs_agent, actions_agent, totals_agent, totals_cost = test_(env_eval, deterministic_act)

	plt.clf() 
	plt.plot(xs_agent, label='agent')
	pltfn = os.path.join(log_path, f'states.png')
	plt.savefig(pltfn, dpi=300)

	plt.clf()
	plt.plot(ys_agent, label='agent')
	plt.plot(refs_agent, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, f'output_ref.png')
	plt.savefig(pltfn, dpi=300)
	plt.clf()
	plt.plot(actions_agent, label='agent')
	plt.ylim(-1.1, 1.1)
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	
	plt.clf()
	plt.plot(totals_agent, label='agent')
	# plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)

	# robust test
	print(f"==================== Roobust Test ====================")
	m = env_eval.dim
	env_eval.set_reset_all(False)
	i = 0
	for param in params_ph[m]:
		robust_path = os.path.join(log_path, f"robust{i}/")
		os.makedirs(robust_path, exist_ok=True)
		env_eval.set_params(*param)
		xs_agent, ys_agent, refs_agent, actions_agent, totals_agent, totals_cost = test_(env_eval, deterministic_act)
		

		plt.clf()
		plt.plot(ys_agent, label='agent')
		plt.plot(refs_agent, label='ref')
		plt.legend()
		pltfn = os.path.join(robust_path, f'output_ref.png')
		plt.savefig(pltfn, dpi=300)
		plt.clf()

		plt.clf()
		plt.plot(totals_agent, label='agent')
		# plt.plot(totals_linear, label='linear')
		plt.legend()
		pltfn = os.path.join(robust_path, 'total_rewards.png')
		plt.savefig(pltfn,dpi=300)

		np.savetxt(os.path.join(robust_path, 'params.txt'), param)
		i+=1

def test_ph_integrator(env_eval_original, agent, log_path, if_uniform=False):
	os.makedirs(log_path, exist_ok=True)
	env_eval = env_eval_original
	env_eval2 = env_eval_original
	if hasattr(env_eval_original, 'integral_punish'):
		env_eval.integral_punish = 0.
		env_eval2.integral_punish = 0.
	# print(f'state{state}')
	deterministic_act = lambda x: agent.select_action(x , if_deterministic=True)
	test_ = test_ph_policy_uniform_integrator if if_uniform else test_ph_policy_integrator
	xs_agent, refs_agent, agent_integrators, actions_agent, totals_agent, totals_cost = test_(env_eval, deterministic_act)


	plt.clf() 
	plt.plot(xs_agent, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(xs_linear[:,0], label='linear')
	plt.plot(refs_agent, label='ref')
	plt.legend()
	pltfn = os.path.join(log_path, f'state.png')
	plt.savefig(pltfn, dpi=300)

	plt.clf()
	plt.plot(agent_integrators, label='agent')
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'integrators.png')
	plt.savefig(pltfn,dpi=300)
	

	plt.clf()
	plt.plot(actions_agent, label='agent')
	plt.ylim(-1.1, 1.1)
	# plt.plot(refs_agent, label='agent_ref')
	# plt.plot(actions_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'actions.png')
	plt.savefig(pltfn,dpi=300)
	# plt.close('all')
	
	plt.clf()
	plt.plot(totals_agent, label='agent')
	# plt.plot(totals_linear, label='linear')
	plt.legend()
	pltfn = os.path.join(log_path, 'total_rewards.png')
	plt.savefig(pltfn,dpi=300)

	# robust test
	print(f"==================== Roobust Test ====================")
	m = env_eval.dim
	env_eval.set_reset_all(False)
	i = 0
	for param in params_ph[m]:
		robust_path = os.path.join(log_path, f"robust{i}/")
		os.makedirs(robust_path, exist_ok=True)
		env_eval.set_params(*param)
		xs_agent, refs_agent, agent_integrators, actions_agent, totals_agent, totals_cost = test_(env_eval, deterministic_act)
		plt.clf()
		plt.plot(agent_integrators, label='agent')
		# plt.plot(refs_agent, label='agent_ref')
		# plt.plot(actions_linear, label='linear')
		plt.legend()
		pltfn = os.path.join(robust_path, 'integrators.png')
		plt.savefig(pltfn,dpi=300)

		plt.clf() 
		plt.plot(xs_agent, label='agent')
		# plt.plot(refs_agent, label='agent_ref')
		# plt.plot(xs_linear[:,0], label='linear')
		plt.plot(refs_agent, label='ref')
		plt.legend()
		pltfn = os.path.join(robust_path, f'state.png')
		plt.savefig(pltfn, dpi=300)

		plt.clf()
		plt.plot(totals_agent, label='agent')
		# plt.plot(totals_linear, label='linear')
		plt.legend()
		pltfn = os.path.join(robust_path, 'total_rewards.png')
		plt.savefig(pltfn,dpi=300)

		plt.clf()
		plt.plot(actions_agent, label='agent')
		plt.ylim(-1.1, 1.1)
		# plt.plot(refs_agent, label='agent_ref')
		# plt.plot(actions_linear, label='linear')
		plt.legend()
		pltfn = os.path.join(robust_path, 'actions.png')
		plt.savefig(pltfn,dpi=300)

		np.savetxt(os.path.join(robust_path, 'params.txt'), param)
		i+=1

