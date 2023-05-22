import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.utils import MODELS, IF_ONPOLICY
from utils.test import *
from utils.robust_test import robust_test_nonlinear_watertank
import gym
import gym_control
# import mujoco_py
import os
from elegantrl.run import Arguments, train_and_evaluate
from elegantrl.env import PreprocessEnv
from elegantrl.utils import configure_logger
import time

import torch
from elegantrl import logger
from copy import deepcopy
from gym_control.envs.nonlinear_watertank import StackingHistoryPreprocessing
gym.logger.set_level(40)  # Block warning

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--algo', default='PPO', type=str)
	parser.add_argument('--env', default='NonLinearWaterTankChangingParamUniformGoalIntegrator-SquareDistance-v2', type=str)
	# parser.add_argument('--wrapper', default='None', type=str)
	parser.add_argument('--reward_type', default='distance', type=str, choices=['distance', 'sparse'])
	parser.add_argument('--fix_K', dest='fix_K', action='store_true')

	parser.add_argument('--seed', default=0, type=int, help='random seed for the code')
	
	parser.add_argument('--target_return', default=1e6, type=float)
	parser.add_argument('--target_step', default=2**3, type=int) # collect target_step, then update network
	parser.add_argument('--reward_scale', default=1., type=float)
	parser.add_argument('--break_step', default=2 ** 20, type=int)
	parser.add_argument('--learning_start', default=0, type=int)

	parser.add_argument('--gamma', default=0.995, type=float)
	parser.add_argument('--batch_size', default=2 ** 6, type=int)
	parser.add_argument('--learning_rate', default=3e-4, type=float)

	parser.add_argument('--buffer_size', default=2 ** 20, type=int)
	parser.add_argument('--net_dim', default=2**4, type=int)	# net dim	
	parser.add_argument('--verbose', default=0, type=int)# verbose: the verbosity level: 0 no output, 1 info, 2 debug
	parser.add_argument('--tensorboard_log', default='tensorboard', type=str)


	parser.add_argument('--env_zero_noise', dest='env_zero_noise', action='store_true')
	parser.set_defaults(env_zero_noise=False)


	# evalutation
	parser.add_argument('--eval_times1', default=2 ** 3, type=int)
	parser.add_argument('--eval_times2', default=2 ** 4, type=int)
	parser.add_argument('--eval_gap', default=5, type=int)
	parser.add_argument('--robust_test', dest='robust_test', action='store_true')
	# training r for fixed goals
	parser.add_argument('--goal', default=4.0, type=float)

	# on policy
	parser.add_argument('--repeat_times', default=2 ** 4, type=int)	# always 1 for off-policy
	# PPO
	parser.add_argument('--lambda_gae_adv', default=0.97, type=float)	# always 1 for off-policy
	parser.add_argument('--lambda_entropy', default=0.02, type=float)	# always 1 for off-policy
	parser.add_argument('--ratio_clip', default=0.2, type=float)	# always 1 for off-policy
	# self-design evaluation
	parser.add_argument('--test_render_times', default=10000, type=int)	# evale self design test per training step

	# load
	parser.add_argument('--load', default='None', type=str)
	# frozen integrator modular part
	parser.add_argument('--frozen_modular_integrator', dest='frozen_modular_integrator', action='store_true')
	parser.set_defaults(frozen_modular_integrator=False)
	parser.add_argument('--frozen_transfer', dest='frozen_transfer', action='store_true')
	parser.set_defaults(frozen_transfer=False)

	args = parser.parse_args()
	assert args.test_render_times % args.target_step == 0, "Must be an integer multiple"
	print("==============================================================================================")
	print(f"Agent: {args.algo}, Env: {args.env}, Seed: {args.seed}")
	if 'FixedGoal' in args.env:
		print(f"Fixed goal reference: {args.goal}")
	print("==============================================================================================")

	# set env
	if args.env_zero_noise:
		# assme the env is 2d!
		if 'NonLinearWaterTank' in args.env:
			env = gym.make(args.env,
								 noise_scale=0., 
								 reward_type=args.reward_type, 
								 r=args.goal)
		else:
			env = gym.make(args.env, 
								 noise_scale=0.)
	else:
		if 'NonLinearWaterTank' in args.env:	
			env = gym.make(args.env, 
								 reward_type=args.reward_type, 
								 r=args.goal)
		else:
			env = gym.make(args.env)
	# print(f"action_change_punishment: {env.action_change_punishment}")
	env.seed(args.seed)
	np.random.seed(args.seed)
	action_dim = env.action_space.high.shape[0]
	env.target_return = args.target_return
	kargs, tensorboard_log = prepare_train(args, env)

	SCN_kwargs = {}
	residual_kwargs = {}
	Modular_kwargs = {}
	if 'residualscn' in args.algo.lower() and hasattr(env, 'K'):
		if action_dim == 1:
			SCN_kwargs['init_K'] = env.K.reshape(-1, 1)
		else:
			SCN_kwargs['init_K'] = env.K.T
	elif 'residual' in args.algo.lower() and hasattr(env, 'K'):
		if action_dim == 1:
			residual_kwargs['init_K'] = env.K.reshape(-1, 1)
		else:
			residual_kwargs['init_K'] = env.K.T
	if 'modular' in args.algo.lower() and hasattr(env, 'n_integrator'):
		Modular_kwargs['integrator_dim'] = env.n_integrator
	kargs.SCN_kwargs = SCN_kwargs
	kargs.residual_kwargs = residual_kwargs
	kargs.Modular_kwargs = Modular_kwargs

	kargs.Q_kwargs = {}
	kargs.if_residual = True


	# add self design evaluation
	test_render = None
	if_uniform = ('Uniform' in args.env)
	print(f'if_uniform {if_uniform}')
	if 'RealWaterTankObserver' in args.env:
		test_render = lambda agent, save_dir: test_realwatertankobserver(env, agent, save_dir, if_uniform)
	elif 'RealWaterTankUniformGoalIntegrator' in args.env:
		test_render = lambda agent, save_dir: test_realwatertank_integrator(env, agent, save_dir, if_uniform)
	elif 'RealWaterTank' in args.env:
		test_render = lambda agent, save_dir: test_realwatertank(env, agent, save_dir, if_uniform)
	elif 'NonLinearWaterTankChangingParam' in args.env:
		test_render = lambda agent, save_dir: [test_watertank(env, agent, save_dir, if_uniform), robust_test_nonlinear_watertank(env, agent, save_dir, if_uniform)]
	elif 'NonLinearWaterTankObserver' in args.env:
		test_render = lambda agent, save_dir: test_watertankobserver(env, agent, save_dir, if_uniform)
	elif 'NonLinearWaterTank' in args.env:
		test_render = lambda agent, save_dir: test_watertank(env, agent, save_dir, if_uniform)
	elif 'Quadcopter' in args.env:	
		test_render = lambda agent, save_dir: test_quadcopter(env, agent, save_dir)
	elif 'Reacher' in args.env and 'Uniform' in args.env:
		test_render = lambda agent, save_dir: test_reacher(env, agent, save_dir)
	elif 'PH' in args.env and 'Uniform' in args.env and 'Integrator' in args.env:
		test_render = lambda agent, save_dir: test_ph_integrator(env, agent, save_dir, if_uniform)
	elif 'PH' in args.env and 'Uniform' in args.env:
		test_render = lambda agent, save_dir: test_ph(env, agent, save_dir, if_uniform)
	kargs.test_render = test_render

	agent, _ = train_and_evaluate(kargs)
	# save final act and q
	save_dir = os.path.join(kargs.cwd,'final_model')
	os.makedirs(save_dir, exist_ok=True)
	agent.save_load_model(save_dir, if_save=True)
	
	# test final
	if 'RealWaterTankObserver' in args.env:
		# test_realwatertankobserver(env, agent, save_dir, if_uniform=if_uniform)
		pass
	elif 'RealWaterTankUniformGoalIntegrator' in args.env:
		# test_realwatertank_integrator(env, agent, save_dir, if_uniform=if_uniform)
		pass
	elif 'RealWaterTank' in args.env:
		# test_realwatertank(env, agent, save_dir, if_uniform=if_uniform)
		pass
	elif 'WaterTank' in args.env:
		test_watertank(env, agent, save_dir, if_uniform=if_uniform)
	elif 'PH' in args.env and 'Integrator' in args.env:
		test_ph_integrator(env, agent, save_dir, if_uniform=if_uniform)
	elif 'PH' in args.env:
		test_ph(env, agent, save_dir, if_uniform=if_uniform)

	# load best agent
	agent.save_load_model(kargs.cwd, if_save=False)
	if 'RealWaterTankObserver' in args.env:
		pass
	elif 'RealWaterTankUniformGoalIntegrator' in args.env:
		pass
	elif 'RealWaterTank' in args.env:
		pass
	elif 'WaterTank' in args.env:
		test_watertank(env, agent, kargs.cwd, if_uniform=if_uniform)
	elif 'PH' in args.env and 'Uniform' in args.env and 'Integrator' in args.env:
		test_ph_integrator(env, agent, kargs.cwd,if_uniform=if_uniform)
	elif 'PH' in args.env and 'Uniform' in args.env:
		test_ph(env, agent, kargs.cwd,if_uniform=if_uniform)
	
	# robust test
	if args.robust_test and 'NonLinearWaterTank' in args.env:
		robust_test_nonlinear_watertank(env, agent, kargs.cwd, if_uniform)

	# log the args
	with open(os.path.join(kargs.cwd,'args.txt'),'w') as data:
		data.write(str(args))
	if 'modular' in args.algo.lower():
		with open(os.path.join(kargs.cwd,'trainable_parameters.txt'),'w') as data:
			data.write(str({'number of actor parameter': sum(p.numel() for p in agent.act.parameters() if p.requires_grad)}))
	print("==============================================================================================")
	print(f"Finish Training and Saved in {kargs.cwd}")
	print("==============================================================================================\n\n")
	env.close()

def prepare_train(args, env):
	log_prex = f'log_{args.break_step}'
	# training setting
	algo = args.algo.lower()
	kargs = Arguments(if_on_policy=IF_ONPOLICY[algo])
	if IF_ONPOLICY[algo]:
		kargs.repeat_times = args.repeat_times
	else:
		kargs.repeat_times = 1

	kargs.gpu_id = 0
	kargs.if_remove = False

	kargs.random_seed = args.seed
	kargs.env = PreprocessEnv(env=env)
	kargs.env_eval = PreprocessEnv(env=env)

	kargs.reward_scale = args.reward_scale
	# RewardRange: -1800 < -200 < -50 < 0
	kargs.net_dim = args.net_dim
	kargs.batch_size = args.batch_size
	kargs.break_step = args.break_step
	kargs.learning_start = args.learning_start

	kargs.eval_times1 = args.eval_times1
	kargs.eval_times2 = args.eval_times2
	kargs.eval_gap = args.eval_gap

	kargs.fix_K = args.fix_K
	kargs.frozen_modular_integrator = args.frozen_modular_integrator
	kargs.frozen_transfer = args.frozen_transfer
	kargs.test_render_times = args.test_render_times
	

	kargs.agent = MODELS[algo]()
	if 'ppo' in args.algo.lower():
		kargs.agent.lambda_entropy = args.lambda_entropy
		kargs.agent.ratio_clip = args.ratio_clip
		kargs.agent.lambda_gae_adv = args.lambda_gae_adv

	str_reward_sparse = "-sparse" if args.reward_type == 'sparse' else ""
	str_no_noise = "-zero" if args.env_zero_noise else ""
	str_fixed = f"/goal_{args.goal}" if 'FixedGoal' in args.env else ""
	str_fixK = "-fixK" if args.fix_K else ""

	tensorboard_log = os.path.join(log_prex,\
							 f"{args.tensorboard_log}_{args.env}{str_no_noise}{str_fixed}/")
	print(f"| tensorboard log: {tensorboard_log}")
	configure_logger(args.verbose, tensorboard_log, f"{args.algo}{str_reward_sparse}-{args.net_dim}{str_fixK}", True)


	now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
	now2 = time.strftime("%H_%M_%S",time.localtime(time.time()))
	kargs.cwd = log_path = os.path.join(log_prex, f'{args.env}{str_no_noise}{str_fixed}/{args.algo}{str_reward_sparse}-{args.net_dim}{str_fixK}/seed{args.seed}/{now}-{now2}')
	
	kargs.target_step = args.target_step
	kargs.load = args.load
	return kargs, tensorboard_log

if __name__ == '__main__':
	main()