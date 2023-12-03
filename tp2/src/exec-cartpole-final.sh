python run-cartpole.py \
  --multirun \
  +n_observations="range(0,10)" \
  interactive.plot="false" \
  model_params.batch_size="256" \
  model_params.memory_size="5000" \
  model_params.gamma="0.95" \
  model_params.tau="0.005" \
  n_episodes="1001"
