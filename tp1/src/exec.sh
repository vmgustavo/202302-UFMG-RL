# https://hydra.cc/docs/advanced/override_grammar/extended/

python run.py \
  --multirun \
  hydra/launcher="joblib" \
  n_episodes="20000" \
  n_observations="range(0, 30)" \
  qtable.alpha="0.001,0.005,0.01,0.05,0.1,0.2,0.4" \
  qtable.gamma="0.0,0.3,0.7,0.9,1.0"
