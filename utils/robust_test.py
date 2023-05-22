import os
from copy import deepcopy
from .test import test_watertank
def robust_test_nonlinear_watertank(env, agent, save_path, if_uniform):
	test_env = deepcopy(env)
	# "a1": 0.0019,
	# "a2": 0.0019,
	# "A1": 1,
	# "A2": 1,
	# "Kp": 0.12,
	# "G": 980,
	test_env.a1=0.0024
	test_env.a2 = 0.0019
	test_env.Kp = 0.12

	test_env.if_reset_all = False
	test_env.max_step=500
	save_dir = os.path.join(save_path,'robust_test/test1')
	os.makedirs(save_dir, exist_ok=True)
	test_watertank(test_env, agent, save_dir, if_uniform=if_uniform)

	with open(os.path.join(save_dir,'params.txt'),'w') as data:
		data.write(str({'a1': test_env.a1,
						'a2': test_env.a2,
						'Kp': test_env.Kp}))


	test_env.a2 = 0.0015
	save_dir = os.path.join(save_path,'robust_test/test2')
	os.makedirs(save_dir, exist_ok=True)
	test_watertank(test_env, agent, save_dir, if_uniform=if_uniform)
	with open(os.path.join(save_dir,'params.txt'),'w') as data:
		data.write(str({'a1': test_env.a1,
						'a2': test_env.a2,
						'Kp': test_env.Kp}))


	test_env.Kp = 0.07
	save_dir = os.path.join(save_path,'robust_test/test3')
	os.makedirs(save_dir, exist_ok=True)
	test_watertank(test_env, agent, save_dir, if_uniform=if_uniform)
	with open(os.path.join(save_dir,'params.txt'),'w') as data:
		data.write(str({'a1': test_env.a1,
						'a2': test_env.a2,
						'Kp': test_env.Kp}))

	test_env.if_reset_all = True