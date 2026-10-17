# Gymnasium API and vectorization (issue #96)

Question: what does the current Gymnasium API require of an environment (`Env`, spaces, `reset`/`step` signature, `terminated` vs `truncated`, seeding, the env checker) and what does it offer for vectorization (`VectorEnv`, autoreset modes)? What do the major RL libraries expect from an env (Gymnasium-compat, vector env use)? Map: #95. Charter: #8. Researched 2026-10-17. Throwaway research branch, never merged.

**Status:** facts only. No RL library is chosen here.

Evidence labels: **[D]** official documentation/source read this session (Context7 snippets of the project's own docs, or the project's docs/raw source fetched directly); **[P]** PyPI JSON metadata (pypi.org/pypi/<pkg>/json) read this session; **[T]** observed in a small local run this session (Python 3.13, `uv run --no-project --with gymnasium`, macOS arm64; single probe, not a proof); **[U]** not verified.

## 1. Gymnasium: version

- Latest on PyPI: `gymnasium 1.4.0`, requires Python >=3.10 (uploaded 2026-10-05) [P]. Local runs below used 1.4.0 [T].
- Context7 lists Gymnasium docs up to `v1.2.3`; items below stated from docs come from the `main` docs, items marked [T] were checked on 1.4.0.

## 2. What Gymnasium requires of an `Env`

Sources: https://gymnasium.farama.org/api/env/ , https://gymnasium.farama.org/introduction/create_custom_env/ , https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/core.py [D]

- Subclass `gymnasium.Env`; the guide says environments "inherit from `gymnasium.Env` that defines the structure all environments must follow." Methods shown: `__init__`, `reset`, `step` (plus `render`, `close`).
- Required attributes: `action_space` ("all valid actions should be contained within the space") and `observation_space` ("all valid observations should be contained within the space"), both `gymnasium.spaces.Space` instances.
- Optional/other attributes: `metadata` (dict, default `{'render_modes': []}`; holds `render_modes`, `render_fps`), `render_mode` (str | None, set at init), `spec` (EnvSpec | None, set by `make`), `unwrapped`, `np_random` (a `np.random.Generator`, lazily seeded with a random seed if unset), `np_random_seed` (-1 if the seed is unknown) [D].
- `reset(*, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]`. Returns `(observation, info)`. `seed`: "If None, the RNG is not reset." [D]
- `step(action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]`. Returns `(observation, reward, terminated, truncated, info)`. "When the end of an episode is reached (terminated or truncated), it is necessary to call reset() to reset the environment's state for the next episode." [D]
- The old single `done` flag and `env.seed()` are the pre-v0.26 API; the migration guide shows `obs, info = env.reset(seed=42)` as the replacement and says seeding moved into `reset` because "some environments can only set their random state at the beginning of an episode." [D] (https://gymnasium.farama.org/introduction/migration_guide/)

### terminated vs truncated

- `terminated`: "Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative." [D]
- `truncated`: "Whether the truncation condition outside the scope of the MDP is satisfied." Typical example: a time limit. [D]
- The guide's custom-env example: `terminated` = goal reached, `truncated` = cutoff such as a step limit; both must be returned from `step()`. [D]
- Why they are split (migration guide): the single `done` was ambiguous for value bootstrapping. Its sketch: on `terminated`, next value = 0; on `truncated`, next value = value function of the next observation. [D]
- `gym.register(..., max_episode_steps=N)` makes `make()` add a `TimeLimit` wrapper that sets `truncated` [D] (create_custom_env guide; migration guide on the v0.26+ TimeLimit using `truncated`).

### Seeding

- Custom `reset` must call `super().reset(seed=seed)` first: "the first line of reset should be `super().reset(seed=seed)` which implements the seeding correctly" (core.py docstring); the custom-env guide: "On the first line, you must call `super().reset(seed=seed)` to properly initialize the random number generator." [D]
- Randomness inside the env is then drawn from `self.np_random` (e.g. `self.np_random.integers(...)`). [D]
- Spaces have their own RNG: `Space.seed()`, `Space.np_random` (lazily created), `Space.sample()` uses it. [D] (https://gymnasium.farama.org/api/spaces/)

### Spaces

Source: https://gymnasium.farama.org/api/spaces/ [D]

- Fundamental: `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Text`. Composite: `Dict`, `Tuple`, `Sequence`, `Graph`, `OneOf`.
- Base `Space` methods: `sample()`, `contains()`, `seed()`, `to_jsonable()`/`from_jsonable()`; attributes `shape`, `dtype`, `np_random`, `is_np_flattenable` (can flatten to a `Box`).
- `Discrete.sample(mask=..., probability=...)`: mask must be `np.int8` array of shape `(n,)` with 1 = valid; all-zero mask returns `space.start`. [D] (action masking exists at sample level only; this note does not say the API enforces masks in `step`).

### Env checker

- `gymnasium.utils.env_checker.check_env(env, warn=None, skip_render_check=False, skip_close_check=False) -> None`. Checks `observation_space`/`action_space` and calls `reset`, `step`, `render` with various values; "highly recommend" calling it after construction and in CI. `warn` is ignored. It raises/prints on problems rather than returning a result. [D] (https://gymnasium.farama.org/api/utils/)
- `PassiveEnvChecker` wrapper: applied automatically by `gym.make` (stack shown as `TimeLimit<OrderEnforcing<PassiveEnvChecker<Env>>>`), disable with `gym.make(..., disable_env_checker=True)`. "No vector version exists." [D]
- `OrderEnforcing` also appears in the default stack (from the same printed repr) [D].

## 3. Gymnasium vectorization

Sources: https://gymnasium.farama.org/api/vector/ , https://farama.org/Vector-Autoreset-Mode , https://gymnasium.farama.org/api/registry/ [D]

- `gymnasium.vector.VectorEnv`: base class running "multiple independent copies of the same environment". Rewards, terminations and truncations are NumPy arrays of shape `(num_envs,)`; observations/actions are batched in a way that depends on the space and is "generally optimised for neural network input/outputs"; infos remain a dict where "a key will give the data for all sub-environment." [D]
- Attributes: `num_envs`, `single_observation_space`, `single_action_space`, batched `observation_space`, batched `action_space` ("The input actions of step must be valid elements of action_space"), `metadata`. `reset(*, seed=None, options=None)` returns batched `(obs, infos)`; `step(actions)` returns `(obs, rewards, terminations, truncations, infos)`. [D]
- Implementations: `SyncVectorEnv` (serial) and `AsyncVectorEnv` (multiprocessing with pipes; shared memory option). The docs claim vector envs "can provide a linear speed-up" in steps/second. [D]
  - `SyncVectorEnv(env_fns, copy=True, observation_mode='same', autoreset_mode=AutoresetMode.NEXT_STEP)`
  - `AsyncVectorEnv(env_fns, shared_memory=True, copy=True, context=None, daemon=True, worker=None, observation_mode='same', autoreset_mode=AutoresetMode.NEXT_STEP, max_concurrency=None)`
  - `env_fns` are callables that each create one `Env` (so the env must be constructible in a worker process for Async).
- `gymnasium.make_vec(id, num_envs=1, vectorization_mode=None, vector_kwargs=None, wrappers=None, **kwargs)`: `vectorization_mode` of `"sync"`, `"async"` or `"vector_entry_point"`; default uses the registered spec's `vector_entry_point` if present, else sync. `wrappers` is only usable in sync/async mode. [D]
- Vector-specific wrappers live in `gymnasium.wrappers.vector` (e.g. `ClipReward`, `NormalizeObservation`, `RecordEpisodeStatistics`). Not every wrapper supports every autoreset mode (below). [D]
- Environments can supply their own batched implementation through a registered `vector_entry_point` [D] (`make_vec` description).

### Autoreset modes

`AutoresetMode` enum; the active mode is exposed as `metadata["autoreset_mode"]` ("The mode used by vector environment should be available in metadata["autoreset_mode"]"). "Warning, some vector implementations or training algorithms will only support particular autoreset modes." [D]

| Mode | Behaviour (from farama.org/Vector-Autoreset-Mode) |
|---|---|
| `NEXT_STEP` (default for Sync and Async) | A sub-env that terminates/truncates is reset on the *following* `step` call. Training code must track episode boundaries to avoid storing invalid transitions in replay buffers. Widely supported by vector wrappers. |
| `SAME_STEP` | Sub-env is reset in the same `step` call; the returned observation is the reset observation, and "the terminated observation" is stored in `info["final_obs"]`. Some vector wrappers do not support this mode. |
| `DISABLED` | No automatic reset; the user resets terminated sub-envs with `env.reset(options={"reset_mask": np.array([True, False, ...], dtype=bool)})`. Training code mirrors single-env patterns but may have overhead from extra calls. Limited wrapper support. |

Local probe on 1.4.0 with CartPole-v1, 2 envs, sync [T]:

- Default `metadata` includes `'autoreset_mode': <AutoresetMode.NEXT_STEP: 'NextStep'>`.
- NEXT_STEP: when env 0 terminated, `infos` had no keys (no `final_obs`/`final_info`); on the next `step`, env 0's reward was `0.0` (the reset step) and flags were False, consistent with the doc's description.
- SAME_STEP: at termination `infos` keys were `final_obs`, `_final_obs`, `final_info`, `_final_info`.
- Wrapper note: the single-env `Autoreset` wrapper (v1.0.0 rename of `AutoResetWrapper`) "reset[s] on the step after" termination/truncation and removed `final_observation`/`final_info`. [D] (https://gymnasium.farama.org/api/wrappers/misc_wrappers/)

## 4. What major RL libraries expect

Versions from PyPI [P]: stable-baselines3 2.9.0 (2026-06-15, Python >=3.10, depends `gymnasium<2.0,>=0.29.1`); cleanrl 1.2.0 (2023-05-22, Python >=3.7.1,<3.11, depends `gym==0.23.1` and `gymnasium>=0.28.1`); ray 2.59.0 (2026-10-02, Python >=3.10; the `rllib`/`all` extras pin `gymnasium==1.2.2`).

### Stable-Baselines3 (SB3)

Sources: SB3 docs `guide/custom_env.md`, `guide/vec_envs.md` (raw docs from github.com/DLR-RM/stable-baselines3) [D]

- Custom env must follow the Gymnasium interface: subclass `gym.Env`, set `action_space`/`observation_space` as `gym.spaces` objects, implement `step` (5-tuple), `reset(seed, options)` (returns `(obs, info)`), `render`, `close`. Template sets `metadata = {"render_modes": [...], "render_fps": ...}`.
- Space support (custom_env doc): Discrete actions with `start=0`, `MultiDiscrete` with a single-dimensional array; non-standard action spaces via wrappers. Observations: Box standard; images must be `np.uint8` in [0, 255]; CNN policies divide by 255 by default (`normalize_images=False` in `policy_kwargs` to disable); channel-first recommended, channel-last transposed by `VecTransposeImage`. (The vec_envs doc states both DummyVecEnv and SubprocVecEnv support Box, Discrete, Dict and Tuple observation spaces.)
- `stable_baselines3.common.env_checker.check_env(env)`: its own checker; Context7's SB3 docs state it "checks a subset of what Gymnasium's checker checks" and may output additional warnings.
- Env registration with `gym.register` is optional; allows `gym.make`.
- Training uses its own `VecEnv` API, **not** Gymnasium's `VectorEnv`: `DummyVecEnv` (sequential) and `SubprocVecEnv` (multiprocessing), built from callables (`make_vec_env` helper). Documented differences from the Gym API:
  - `reset()` returns only obs (infos in `vec_env.reset_infos`); no `seed`/`options` args, use `vec_env.seed(seed)` before `reset()`.
  - `step` returns a 4-tuple `obs, rewards, dones, infos`.
  - Auto-resets; only the first `reset()` call is needed. On episode end the returned obs is the first obs of the next episode; the last obs is in `infos[i]["terminal_observation"]`.
  - `infos[i]["TimeLimit.truncated"] = truncated and not terminated`.
  - The doc states the VecEnv API "aligns with Gym 0.21 API but differs to Gym 0.26+ API".

### CleanRL

Sources: raw `cleanrl/ppo.py` on master (fetched and read this session), CleanRL docs via Context7 [D]; PyPI [P]

- Single-file scripts. `ppo.py` builds `gym.vector.SyncVectorEnv([make_env(...) for i in range(args.num_envs)])`, wrapping each sub-env with `gym.wrappers.RecordEpisodeStatistics` (and optional `RecordVideo`).
- Uses the 5-tuple API: `envs.reset(seed=args.seed)` and `next_obs, reward, terminations, truncations, infos = envs.step(...)`; merges flags with `next_done = np.logical_or(terminations, truncations)` (no separate bootstrapping on truncation in that script) and uses `1.0 - next_done` as the non-terminal mask.
- Reads episode stats from `infos["final_info"]` (`info["episode"]["r"]`, `["l"]`). Per the Gymnasium autoreset docs above, `final_info` is provided in SAME_STEP mode and absent in NEXT_STEP (confirmed absent in default mode on 1.4.0 [T]); whether `ppo.py` on master matches the default mode of a given Gymnasium release was not tested [U]. The PyPI release 1.2.0 (2023) predates Gymnasium 1.0 [P].
- Space assumptions are per-script: `ppo.py` asserts `isinstance(envs.single_action_space, gym.spaces.Discrete)` ("only discrete action space is supported") and reads `envs.single_observation_space.shape` and `envs.single_action_space.n`. Continuous-action variants are separate scripts [U, not read].
- CleanRL docs note a EnvPool (<=0.6.4) vs gym vector difference in the observation returned after termination (last state vs new-episode first state), i.e. libraries differ on this and scripts adapt to one.

### RLlib (Ray)

Sources: RLlib docs `rllib/env.md` and `rllib/scaling-guide.md` via Context7 [D]; PyPI [P]

- Custom env: a `gymnasium.Env` subclass whose constructor takes a single `config` argument (default None) and which provides `observation_space` and `action_space`; `reset(seed=None, options=None)` returns `(obs, info)`; `step` returns `(obs, reward, terminated, truncated, info)`.
- Registration: pass the class directly to `PPOConfig().environment(MyEnv, env_config={})`, or `ray.tune.registry.register_env(name, creator)` where the creator takes one `config` arg and returns a non-vectorized `gymnasium.Env`. The docs say to use Ray's registry rather than Gymnasium's because Ray is distributed.
- Vectorization: RLlib vectorizes envs on `EnvRunner` actors "through gymnasium's VectorEnv API"; `config.env_runners(num_envs_per_env_runner=N)` sets the number of sub-envs; `gym_env_vectorize_mode=gym.envs.registration.VectorizeMode.ASYNC` runs each sub-env in its own process (default `SYNC`).
- Pinned Gymnasium: `gymnasium==1.2.2` in Ray 2.59.0's `rllib` extra [P]. Which autoreset mode RLlib's EnvRunner uses was not read [U].

### Others

Not researched in this pass (the question said "e.g."): Tianshou, TorchRL, skrl, Sample Factory, EnvPool, etc. [U]

## 5. Cross-cutting facts that matter for any choice (observations, not recommendations)

- All three libraries read here accept a `Env` using the 5-tuple Gymnasium API with `observation_space`/`action_space`; each lists its own limits on space types (SB3 actions: Discrete(start=0)/MultiDiscrete plus others via wrappers; CleanRL script-specific; RLlib: Gymnasium spaces).
- Env must be constructible by a zero-arg callable (Gymnasium Sync/Async, SB3 Dummy/Subproc) or by a `config`-taking constructor (RLlib).
- The `terminated`/`truncated` split is preserved by Gymnasium vector envs, collapsed to `dones` in SB3's VecEnv (with `TimeLimit.truncated` in infos) and merged to `next_done` in CleanRL PPO.
- Autoreset semantics differ between Gymnasium NEXT_STEP (default in 1.x), Gymnasium SAME_STEP, and SB3's own VecEnv (reset in the same step with `terminal_observation`); consumers must match the producer.

## Sources

- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- Custom env guide: https://gymnasium.farama.org/introduction/create_custom_env/
- Migration guide: https://gymnasium.farama.org/introduction/migration_guide/
- Core source: https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/core.py
- Spaces: https://gymnasium.farama.org/api/spaces/
- Utils / `check_env`: https://gymnasium.farama.org/api/utils/
- Misc wrappers (`PassiveEnvChecker`, `Autoreset`): https://gymnasium.farama.org/api/wrappers/misc_wrappers/
- Vector API: https://gymnasium.farama.org/api/vector/ , https://gymnasium.farama.org/api/vector/sync_vector_env/ , https://gymnasium.farama.org/api/vector/async_vector_env/
- Registry / `make_vec`: https://gymnasium.farama.org/api/registry/
- Autoreset modes: https://farama.org/Vector-Autoreset-Mode
- SB3: https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/custom_env.md , https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/vec_envs.md
- CleanRL: https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo.py , https://github.com/vwxyzjn/cleanrl/blob/master/docs/rl-algorithms/ppo.md
- RLlib: https://github.com/ray-project/ray/blob/master/doc/source/rllib/env.md , https://github.com/ray-project/ray/blob/master/doc/source/rllib/scaling-guide.md
- PyPI JSON: https://pypi.org/pypi/{gymnasium,stable-baselines3,cleanrl,ray}/json

## Local test method

`uv run --no-project --python 3.13 --with gymnasium python -I t.py` (scratch script, not committed): `gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")` with default and `autoreset_mode=AutoresetMode.SAME_STEP`; printed `metadata`, termination flags, infos keys, and the reward on the step after termination.
