break_step=400000;
target_return=0;
eval_times1=50;
eval_times2=100;
batch_size=256;
repeat_times=10;
target_step=2000;
test_render_times=200000;
eval_gap=1;

# for env in 'NonLinearWaterTankChangingParamUniformGoalIntegrator-SquareDistance-v2'; do
# 	for s in 0 1 2 3 4; do
# 		for n in 256; do
# 			python3 train.py --fix_K --robust_test --algo ResidualIntegratorModularPPO --repeat_times $repeat_times --net_dim $n --env $env --break_step $break_step --target_return $target_return  --seed $s --target_step $target_step --reward_type distance --batch_size $batch_size --eval_times1 $eval_times1 --eval_times2 $eval_times2  --eval_gap $eval_gap --test_render_times $test_render_times
# 			python3 train.py --robust_test --algo PPO --repeat_times $repeat_times --net_dim $n --env $env --break_step $break_step --target_return $target_return  --seed $s --target_step $target_step --reward_type distance --batch_size $batch_size --eval_times1 $eval_times1 --eval_times2 $eval_times2  --eval_gap $eval_gap --test_render_times $test_render_times
# 		done
# 	done
# done

for env in 'NonLinearWaterTankChangingParamUniformGoalStacking10-SquareDistance-v2' 'NonLinearWaterTankChangingParamUniformGoalStacking1-SquareDistance-v2'; do
	for s in 0 1 2 3 4; do
		for n in 256; do
			python3 train.py --fix_K --robust_test --algo ResidualPPO --repeat_times $repeat_times --net_dim $n --env $env --break_step $break_step --target_return $target_return  --seed $s --target_step $target_step --reward_type distance --batch_size $batch_size --eval_times1 $eval_times1 --eval_times2 $eval_times2  --eval_gap $eval_gap --test_render_times $test_render_times
			# python3 train.py --robust_test --algo PPO --repeat_times $repeat_times --net_dim $n --env $env --break_step $break_step --target_return $target_return  --seed $s --target_step $target_step --reward_type distance --batch_size $batch_size --eval_times1 $eval_times1 --eval_times2 $eval_times2  --eval_gap $eval_gap --test_render_times $test_render_times
		done
	done
done
