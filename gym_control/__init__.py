from gym.envs.registration import register
import numpy as np
register(
	id="PH1DChangingParamUniformGoalIntegrator-SqaureDistance-v35",
	entry_point="gym_control.envs:PH1DChangingParamUniformGoalIntegrator",
	max_episode_steps=50,
	kwargs={
		'P_control_K': np.array([-0.02, 0.02, 0.035]),
		'P_control_L': None,
		'reward_type': 'square_distance',
		'max_episode_steps': 50,
		'MHCl':np.arange(0.,1, step=0.00001),    # amount of hydrocloric acid used            
	}
)

register(
	id="PH1DChangingParamUniformGoalIntegrator-SqaureDistance-NoIB-v35",
	entry_point="gym_control.envs:PH1DChangingParamUniformGoalIntegrator_NoBound",
	max_episode_steps=50,
	kwargs={
		'P_control_K': np.array([-0.02, 0.02, 0.035]),
		'P_control_L': None,
		'reward_type': 'square_distance',
		'max_episode_steps': 50,
		'MHCl':np.arange(0.,1, step=0.00001),    # amount of hydrocloric acid used            
	}
)

register(
	id='NonLinearWaterTankChangingParamUniformGoal-SquareDistance-v2',
	entry_point='gym_control.envs:NonLinearWaterTankChangingParamUniformGoal',
	kwargs= {
			 "reset_from_last_state": False,
			 "max_step": 200,
			 "a1": [0.0015, 0.0024],
			 "a2": [0.0015, 0.0024],
			 "A1": 1,
			 "A2": 1,
			 "Kp": [0.07, 0.17],
			 "G": 980,
			 "sample_t": 2,
			 "n_discrete": 20,
			 "controller_type": 'P',
			 "reward_type": "square_distance",
			 "gamma": 0.99,
			 "P_control_K": np.array([0., 0.4, -0.4, 0.]),  # 4 state
			 "reset_from_last_state": False}
)

register(
	id='NonLinearWaterTankChangingParamUniformGoalIntegrator-SquareDistance-v2',
	entry_point='gym_control.envs:NonLinearWaterTankChangingParamUniformGoalIntegrator',
	kwargs= {
			 "reset_from_last_state": False,
			 "max_step": 200,
			 "a1": [0.0015, 0.0024],
			 "a2": [0.0015, 0.0024],
			 "A1": 1,
			 "A2": 1,
			 "Kp": [0.07, 0.17],
			 "G": 980,
			 "sample_t": 2,
			 "n_discrete": 20,
			 "controller_type": 'P',
			 "reward_type": "square_distance",
			 "gamma": 0.99,
			 "P_control_K": np.array([0., 0.4, -0.4, 0.]),  # 4 state
			 "reset_from_last_state": False}
)

num_stack=4
K=np.zeros(3*num_stack)
K[-3:]=np.array([0., 0.4, -0.4,])
register(
	id='NonLinearWaterTankChangingParamUniformGoalStacking4-SquareDistance-v2',
	entry_point='gym_control.envs:NonLinearWaterTankChangingParamUniformGoalStacking',
	kwargs= {
			 "reset_from_last_state": False,
			 "max_step": 200,
			 "a1": [0.0015, 0.0024],
			 "a2": [0.0015, 0.0024],
			 "A1": 1,
			 "A2": 1,
			 "Kp": [0.07, 0.17],
			 "G": 980,
			 "sample_t": 2,
			 "n_discrete": 20,
			 "controller_type": 'P',
			 "reward_type": "square_distance",
			 "gamma": 0.99,
			 "P_control_K": K,  # 4 state
			 "reset_from_last_state": False,
			 "num_stack": num_stack}
)
num_stack=10
K=np.zeros(3*num_stack)
K[-3:]=np.array([0., 0.4, -0.4,])
register(
	id='NonLinearWaterTankChangingParamUniformGoalStacking10-SquareDistance-v2',
	entry_point='gym_control.envs:NonLinearWaterTankChangingParamUniformGoalStacking',
	kwargs= {
			 "reset_from_last_state": False,
			 "max_step": 200,
			 "a1": [0.0015, 0.0024],
			 "a2": [0.0015, 0.0024],
			 "A1": 1,
			 "A2": 1,
			 "Kp": [0.07, 0.17],
			 "G": 980,
			 "sample_t": 2,
			 "n_discrete": 20,
			 "controller_type": 'P',
			 "reward_type": "square_distance",
			 "gamma": 0.99,
			 "P_control_K": K,  # 4 state
			 "reset_from_last_state": False,
			 "num_stack": num_stack}
)
num_stack=1
K=np.zeros(3*num_stack)
K[-3:]=np.array([0., 0.4, -0.4,])
register(
	id='NonLinearWaterTankChangingParamUniformGoalStacking1-SquareDistance-v2',
	entry_point='gym_control.envs:NonLinearWaterTankChangingParamUniformGoalStacking',
	kwargs= {
			 "reset_from_last_state": False,
			 "max_step": 200,
			 "a1": [0.0015, 0.0024],
			 "a2": [0.0015, 0.0024],
			 "A1": 1,
			 "A2": 1,
			 "Kp": [0.07, 0.17],
			 "G": 980,
			 "sample_t": 2,
			 "n_discrete": 20,
			 "controller_type": 'P',
			 "reward_type": "square_distance",
			 "gamma": 0.99,
			 "P_control_K": K,  # 4 state
			 "reset_from_last_state": False,
			 "num_stack": num_stack}
)