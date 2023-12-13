# https://hydra.cc/docs/advanced/override_grammar/extended/

python run-cartpole.py \
  --multirun \
  +n_observations="range(0,10)" \
  hydra/launcher="joblib" \
  ++hydra.launcher.n_jobs="1" \
  interactive.plot="false" \
  model_params.batch_size="128,256"

python run-cartpole.py \
  --multirun \
  +n_observations="range(0,10)" \
  hydra/launcher="joblib" \
  ++hydra.launcher.n_jobs="1" \
  interactive.plot="false" \
  model_params.gamma="0.95,0.99"

python run-cartpole.py \
  --multirun \
  +n_observations="range(0,10)" \
  hydra/launcher="joblib" \
  ++hydra.launcher.n_jobs="1" \
  interactive.plot="false" \
  model_params.tau="0.005,0.01"

python run-cartpole.py \
  --multirun \
  +n_observations="range(0,10)" \
  hydra/launcher="joblib" \
  ++hydra.launcher.n_jobs="1" \
  interactive.plot="false" \
  model_params.memory_size="5000,10000"
