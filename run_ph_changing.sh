break_step=200000;
test_render_times=50000;
target_step=1000;
eval_gap=1;
batch_size=128;
target_return=0;
eval_times1=50;
eval_times2=100;
repeat_times=8;
lambda_gae_adv=0.99;
ratio_clip=0.2;
gamma=0.98;
lr=0.0003;

for s in 0 1 2 3 4; do
	for env in 'PH1DChangingParamUniformGoalIntegrator-SqaureDistance-v35'; do
		for net_dim in 128; do
			python3 train.py --fix_K --algo ResidualIntegratorModularPPO \
							--gamma $gamma \
							--learning_rate $lr \
							--net_dim $net_dim \
							--repeat_times $repeat_times \
							--env $env \
							--break_step $break_step \
							--target_return $target_return \
							--seed $s --target_step $target_step \
							--reward_type distance \
							--batch_size $batch_size \
							--lambda_gae_adv $lambda_gae_adv \
							--ratio_clip $ratio_clip\
							--eval_times1 $eval_times1 --eval_times2 $eval_times2 --eval_gap $eval_gap --test_render_times $test_render_times
			python3 train.py --algo PPO \
							--gamma $gamma \
							--learning_rate $lr \
							--net_dim $net_dim \
							--repeat_times $repeat_times \
							--env $env \
							--break_step $break_step \
							--target_return $target_return \
							--seed $s --target_step $target_step \
							--reward_type distance \
							--batch_size $batch_size \
							--lambda_gae_adv $lambda_gae_adv \
							--ratio_clip $ratio_clip\
							--eval_times1 $eval_times1 --eval_times2 $eval_times2 --eval_gap $eval_gap --test_render_times $test_render_times
		
		done
	done
done