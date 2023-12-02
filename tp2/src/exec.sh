# https://hydra.cc/docs/advanced/override_grammar/extended/

python src/run-inverted-double-pendulum.py \
  model_params.batch_size="128" \
  model_params.gamma="0.99" \
  model_params.eps_start="0.9" \
  model_params.eps_end="0.05" \
  model_params.eps_decay="10000" \
  model_params.tau="0.005" \
  model_params.lr="1e-4" \
  model_params.memory_size="10000" \
  n_episodes="2000"
