# https://hydra.cc/docs/advanced/override_grammar/extended/

python run.py \
  --multirun \
  hydra/launcher="joblib" \
  n_episodes=5 \
  qtable.alpha="range(0, 1, step=0.1)" \
  qtable.gamma="range(0, 1, step=0.2)"
