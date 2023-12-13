# https://hydra.cc/docs/advanced/override_grammar/extended/

python run-inverted-double-pendulum.py \
  --multirun \
  hydra/launcher="joblib" \
  ++hydra.launcher.n_jobs="2" \
  interactive.plot="false" \
  model_params.batch_size="128,256" \
  model_params.gamma="0.95,0.99" \
  model_params.eps_start="0.9,0.99" \
  model_params.eps_end="0.01,0.05" \
  model_params.eps_decay="10000,20000" \
  model_params.tau="0.005,0.01" \
  model_params.lr="1e-1,1e-4" \
  model_params.memory_size="5000,10000" \
  n_episodes="500"
