# How PPO is tuned for discrete trading under transaction costs

Resolves #168 (child of map #166). Question: which levers matter when a PPO Agent with a discrete short/flat/long action space collapses to high turnover under a per-change cost, and which suit this environment.

Each claim is marked **[S]** (stated by the cited source) or **[I]** (our inference from sources plus this repo; treat as a hypothesis for the experiment tickets).

## This environment (from the code)

- Actions: Discrete(3) = short/flat/long; Reward per Bar = `position * move_pips - cost`, `cost = SPREAD_PIPS/2 * |target - prev|` (`src/shadow_trader/env.py`). With `SPREAD_PIPS = 1.0` a flat-to-long change costs 0.5 pip and a long-to-short reversal costs 1.0 pip.
- Reward is in raw pips, unnormalised. Episodes are 288 Bars. The Position is already part of the Observation (last element).
- PPO (`src/shadow_trader/config.py`): gamma 0.99, gae_lambda 0.95, ent_coef 0.01, n_steps 2048, batch 64, 10 epochs, separate pi/vf networks [64, 64], `normalize_advantage` True, no VecNormalize, single seed 42.

## Findings by lever, with sources

### Action persistence / frame-skip
- Frame skip (repeating an action for several steps) was a strong determinant of success in all six Atari games tested, and a large skip was critical in two. **[S]** Braylan et al., "Frame Skip Is a Powerful Parameter for Learning to Play Atari", AAAI workshop 2015 (https://cdn.aaai.org/ocs/ws/ws0098/10156-45978-1-PB.pdf; summary read via search, abstract-level only).
- Skipping cuts the effective horizon from about 1/(1-gamma) to about 1/(1-gamma^d), and the authors argue this can offset the cost of repeating actions; they also note one unwarranted repeated action can be costly. **[S]** Kalyanakrishnan et al., "An Analysis of Frame-skipping in Reinforcement Learning" (https://arxiv.org/abs/2102.03718).
- FiGAR lets the agent choose an action and how long to repeat it, attachable to any policy-based method. **[S]** Sharma, Srinivas, Ravindran (https://arxiv.org/abs/1702.06054).
- None of these papers study trading costs. **[I]** Acting only every k Bars (or a minimum-hold rule) bounds turnover mechanically to at most 1/k of Bars, whatever the policy does, and also shortens the credit-assignment horizon. Cost: it removes the Agent's ability to react within k Bars, so k is a parameter to sweep (e.g. 3, 6, 12 on M5).

### Discount factor (gamma) and GAE lambda
- "Discount factor gamma is one of the most important hyperparameters and should be tuned per environment"; 0.99 was a reasonable start. Lambda 0.9 with GAE was recommended. **[S]** Andrychowicz et al., "What Matters In On-Policy RL" (https://arxiv.org/abs/2006.05990; findings read via ar5iv HTML). Caveat: five MuJoCo continuous-control tasks only.
- **[I]** Here the Reward arrives every Bar (position times move), so little is lost by a short horizon. At gamma 0.99 the effective horizon is about 100 Bars, so each action's advantage absorbs about 100 Bars of unrelated, high-variance price noise against a 0.5 pip one-step cost. Gamma around 0.9 to 0.95 (horizon 10 to 20) is a cheap test. Risk: it also lowers the weight on holding through multi-Bar moves.

### Entropy coefficient
- SB3 PPO default `ent_coef` is 0.0; the loss is `policy_loss + ent_coef * entropy_loss + vf_coef * value_loss`, with entropy a minibatch mean. **[S]** SB3 PPO docs (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and source (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py).
- Advantages are normalised per minibatch to unit std in SB3 PPO. **[S]** same source. **[I]** So the entropy bonus competes against a unit-scale advantage regardless of pip scale; 0.01 is then not negligible when the true signal is near zero, and keeps the policy close to uniform over 3 actions (a uniform policy flips Position on about 2/3 of Bars). Sweep ent_coef in {0, 0.001, 0.01}.
- Andrychowicz et al. found no evidence that regularisers help significantly (continuous control, with a careful initial policy). **[S]**, same link as above.

### Reward scaling / VecNormalize
- SB3 recommends always normalising inputs, e.g. with `VecNormalize`. **[S]** (https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html; that page says nothing about reward scale specifically.)
- `VecNormalize` divides rewards by the running std of the discounted return (gamma 0.99 default), then clips to +/-10 (`clip_reward`). **[S]** source: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py.
- SB3 notes `clip_range_vf` "depends on the reward scaling". **[S]** PPO docs.
- Andrychowicz et al.: value-function normalisation was crucial on HalfCheetah and Humanoid but significantly hurt Walker2d; observation normalisation was crucial on most tasks. **[S]**
- **[I]** Uniform scaling leaves the cost-to-P&L ratio unchanged and, with advantage normalisation and separate pi/vf networks, mostly affects value-function fit rather than the policy gradient. Expect little effect on turnover; low priority. Note VecNormalize's clip and running statistics change what the evaluation Reward means, so evaluation must use raw pips (unnormalised env), not the normalised one.

### Cost-aware observation
- Moody & Saffell: nonzero trading costs discourage excessive trading, and the system must be recurrent because the return depends on the previous position F(t-1). **[S]** NeurIPS 1998 "Reinforcement Learning for Trading" (https://papers.nips.cc/paper/1998/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html; text read from the PDF); journal version IEEE TNN 12(4), 2001, not read directly.
- Zhang, Zohren, Roberts: cost is a reward term on position change; the same {-1, 0, 1} action space is used with DQN, which had the best Sharpe of their methods; they call the cost a regulariser on turnover; the state is price-derived features and does not explicitly list the position. **[S]** https://arxiv.org/abs/1911.10107 (read via ar5iv). Their cost rates (20 bp of value) are not comparable to this setup.
- **[I]** This env already exposes the Position, which is the key cost-relevant input those papers rely on. Extra features (time since last change, a unit-cost channel) are plausible but low-evidence; lower priority.

### Value-function noise and measurement
- SB3: "always do several runs" because results vary with seed alone; evaluate over several episodes. Automatic hyperparameter optimisation is the recommended practice on a new problem. **[S]** RL tips page.
- **[I]** Rollout is 2048 steps of a single env (about 7 Episodes) with a per-step Reward whose noise is a few pips against a 0.5 pip cost; more parallel envs or larger `n_steps` reduce gradient and value noise at a CPU-time cost. Pair with the multi-seed yardstick from map #166.

## Ranked candidate levers for this environment

1. **Action persistence** (decide every k Bars, or a minimum hold). Strongest and most direct against 35% turnover; supported by the frame-skip literature as a general trick, though not tested under trading costs. Sweep k.
2. **Lower gamma (and GAE lambda)**, about 0.9 to 0.95. Cheap config-only change; shortens the noise horizon that swamps the cost. Supported by Andrychowicz (tune gamma) and Kalyanakrishnan (shorter effective horizon).
3. **ent_coef down** to 0 to 0.001 (possibly annealed). Cheap, config-only; the 0.01 setting is the repo's one deliberate deviation from default and plausibly pins the policy near uniform.
4. **Lower rollout noise** (more envs / larger n_steps, larger batch). Config-only; modest expected effect, but makes every other comparison cleaner.
5. **Cost-aware observation extras** (time in position, last-change flag). Position is already observed; little literature support for more.
6. **Reward scaling / VecNormalize.** Standard hygiene but unlikely to change turnover on its own (see inference above); if used, evaluate in raw pips.

All six need the multi-seed (3 to 5) validation net/gross/turnover yardstick from map #166 to judge.

## Not found / not verified

- No source studied PPO specifically on discrete trading with a flat per-change cost at M5 frequency; transfer from Atari and MuJoCo tuning papers is by analogy.
- The Braylan paper was confirmed at abstract level only. The IEEE TNN 2001 Moody & Saffell paper was not read.
- Rank order is a judgment, not a measured result.
