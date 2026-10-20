# Research: Gymnasium + SB3 PPO stack on Python 3.13

Issue: shadowgate15/shadow-trader#130 (parent map #128). Researched 2026-10-20.
Findings only; no stack decision is made here and no project dependencies were changed.

Legend: **[verified]** = checked against a primary source or reproduced locally in a throwaway uv project outside the repo; **[unverified]** = from memory or not checked against a primary source.

## 1. Versions and Python 3.13 / macOS arm64 install

| Package | Latest stable | requires-python | Source |
|---|---|---|---|
| gymnasium | 1.4.0 | >=3.10 | PyPI JSON API, https://pypi.org/pypi/gymnasium/json [verified] |
| stable-baselines3 | 2.9.0 (released 2026-06-15) | >=3.10 | https://pypi.org/pypi/stable-baselines3/json ; changelog https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html [verified] |
| torch | 2.14.1 | >=3.10 | https://pypi.org/pypi/torch/json [verified] |
| numpy (resolved) | 2.5.4 | | local resolve [verified] |

- SB3 2.9.0 core deps: `gymnasium>=0.29.1,<2.0`, `numpy>=1.20,<3`, `torch>=2.8,<3`, `cloudpickle`. Pandas and matplotlib moved from core to the `extra` extra in 2.9.0. [verified, PyPI + changelog]
- SB3 2.8.0 (2026-04-01) added official Python 3.13 support and dropped 3.9. [verified, changelog]
- torch 2.14.1 publishes `cp313-cp313-macosx_14_0_arm64` wheels (also cp310-cp314, incl. 3.14 and 3.14t). So Apple Silicon needs macOS 14+ for the wheel. [verified, PyPI file list]
- Local reproduction: in a scratch uv project (`uv init --bare --python 3.13`, outside the repo) `uv add stable-baselines3 tqdm rich` resolved and installed cleanly on macOS arm64: gymnasium 1.4.0, stable-baselines3 2.9.0, torch 2.14.1, numpy 2.5.4, tqdm 4.70.1, rich 15.0.0. [verified]
- Compatibility with existing repo deps: this repo already pins `numpy>=2.5.4` (oracle group), `rich>=15.0.0`, `polars>=2`, `pyarrow>=26`. The scratch resolve picked numpy 2.5.4 and rich 15.0.0, so no visible conflict, but the combined resolve with the repo's pyproject was not run. [partially verified]
- Gymnasium extras of note: `gymnasium[torch]` only adds array-api-compat; not needed for SB3. Box2D etc. extras are irrelevant for a custom env. [verified, PyPI metadata]
- Torch size: it is a large dependency (not measured here). [unverified]
- SB3 install docs state Python 3.10+ and PyTorch >= 2.8: https://stable-baselines3.readthedocs.io/en/master/guide/install.html . The page says nothing about Apple Silicon/MPS. [verified]

## 2. Environment API contract and how SB3 checks it

Sources: Gymnasium docs via Context7 (/websites/gymnasium_farama, /farama-foundation/gymnasium); SB3 custom env guide https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html (Context7 /dlr-rm/stable-baselines3).

Contract (Gymnasium >= 0.26 API):
- Subclass `gymnasium.Env`; define `self.action_space` and `self.observation_space` as `gymnasium.spaces` objects.
- `reset(self, *, seed=None, options=None) -> (observation, info)`. Call `super().reset(seed=seed)` to seed `self.np_random`. Must be called before the first `step`.
- `step(self, action) -> (observation, reward, terminated, truncated, info)`. `terminated` = natural end of the MDP (e.g. out of data / account blown); `truncated` = cut off by a limit (time/data horizon). The old 4-tuple `done` API is not what an SB3 custom env should return.
- Optional: `metadata`, `render`, `close`.
- Observations must lie in `observation_space` and match its dtype (Box float32 is conventional). Gymnasium's own checker is a superset of what SB3 supports.

SB3 checking:
- `from stable_baselines3.common.env_checker import check_env; check_env(env, warn=True)`. SB3 docs: it "checks a subset of what Gymnasium's checker checks" and emits additional warnings. Gymnasium has its own `gymnasium.utils.env_checker.check_env`, which checks a superset of what SB3 supports. [verified]
- Locally, `check_env` passed on a minimal Box(4,)/Discrete(3) env. [verified]
- SB3 auto-wraps a bare `gym.Env` in a `DummyVecEnv` + `Monitor`. VecEnv semantics: finished sub-envs are auto-reset. [SB3 behavior from memory; the Monitor/DummyVecEnv wrapping was not re-checked against source] [unverified]
- Other documented tip: smoke-test an env with random actions before training (SB3 rl_tips). [verified via Context7]

## 3. PPO with discrete actions and a flat observation

Source: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html [verified]
- PPO supports Discrete, Box, MultiDiscrete, MultiBinary actions; Dict actions are not supported. Policies: `MlpPolicy` (flat vectors), `CnnPolicy`, `MultiInputPolicy` (Dict observations).
- Defaults: learning_rate 3e-4, n_steps 2048 per env, batch_size 64, n_epochs 10, gamma 0.99, gae_lambda 0.95, clip_range 0.2, max_grad_norm 0.5.
- Docs say PPO is meant primarily for CPU when not using a CNN; use `device="cpu"`. `SubprocVecEnv` is recommended over `DummyVecEnv` for speed with many envs.
- A flat Box observation with `MlpPolicy` is the standard case. Observation normalization is not automatic; `VecNormalize` exists for that [unverified in this research].

Minimal calls, reproduced locally on SB3 2.9.0 / Python 3.13 [verified]:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

check_env(env)
model = PPO("MlpPolicy", env, verbose=0, device="cpu", seed=0)
model.learn(total_timesteps=4096)
model.save("model")                      # writes model.zip
model = PPO.load("model", device="cpu")  # optionally PPO.load(path, env=env)
action, _state = model.predict(obs, deterministic=True)
```

Observed: `predict` returned `(array(1), None)` for a single obs. `torch.backends.mps.is_available()` was True on this machine, but SB3 docs recommend CPU for MLP PPO; not benchmarked here. [verified/unbenchmarked]

Caveat: `PPO.load` unpickles via cloudpickle/torch inside the zip; only load model files you trust. [from general knowledge, unverified against SB3 docs]

## 4. Progress output during training (CLI fit)

Source: SB3 callbacks guide https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (Context7) [verified]
- `model.learn(total_timesteps, progress_bar=True)` is equivalent to passing `ProgressBarCallback()`. It requires `tqdm` and `rich` (installed via `stable-baselines3[extra]`). Locally, with tqdm and rich installed, it rendered a rich bar `100% ... 4,096/4,096 [0:00:01 < 0:00:00, 2,816 it/s]`. [verified] Note: the repo already depends on `rich`; `tqdm` would be new. Without them `learn(progress_bar=True)` raises an ImportError telling you to install them (message not reproduced here) [unverified].
- Custom callbacks: subclass `BaseCallback`; hooks `_on_training_start`, `_on_rollout_start`, `_on_step` (return False to abort training early), `_on_rollout_end`, `_on_training_end`. Attributes: `self.model`, `self.training_env`, `self.n_calls`, `self.num_timesteps`, `self.locals`, `self.logger`. Locally a `_on_step` callback printing every 1024 calls worked. [verified]
- Built-ins: `CheckpointCallback(save_freq, save_path, name_prefix)` (save_freq counts per-env calls; divide by n_envs), `EvalCallback(eval_env, best_model_save_path, log_path, eval_freq, deterministic=True)` (writes `evaluations.npz`), `CallbackList`/list passed to `learn`. [verified]
- Logger: `verbose=1` prints SB3's periodic table to stdout; `train/*` metrics (e.g. `train/loss`) only appear in `logger.name_to_value` after the first policy update (observed `None` before 3072 steps). TensorBoard output via `tensorboard_log=` needs the `tensorboard` extra. [first part verified; TensorBoard details unverified]

## 5. Alternatives if there is a blocker

No blocker was found for the SB3 stack on Python 3.13 / macOS arm64. Alternatives, for reference only (PyPI metadata checked, API suitability not evaluated):

| Option | Latest | requires-python | Note |
|---|---|---|---|
| sb3-contrib | 2.9.0 | >=3.10 | Same family; depends on `stable_baselines3>=2.9.0,<3`. Docs list MaskablePPO (invalid-action masking), RecurrentPPO, TRPO, QR-DQN, etc. (Context7 /stable-baselines-team/stable-baselines3-contrib) [verified] |
| TorchRL | 0.14.1 | >=3.10 | PyTorch-native; heavier API [PyPI verified, rest unverified] |
| Tianshou | 2.0.1 | >=3.11,<4 | Gymnasium-compatible; 3.13 inside range [PyPI verified, rest unverified] |
| Ray RLlib (`ray`) | 2.59.0 | >=3.10 | Heavy distributed framework [PyPI verified; 3.13 wheels not checked] |
| CleanRL | 1.2.0 | >=3.7.1,<3.11 | PyPI metadata excludes Python 3.13 and pins old gym/torch; typically copied as scripts rather than installed [PyPI verified] |

## Open questions not answered here
- Combined `uv` resolve against this repo's real `pyproject.toml` (including the `oracle` group) was not run.
- Training throughput on this machine (CPU vs MPS) was not benchmarked.
- Reproducibility/determinism guarantees of `seed=` across torch versions not examined.
