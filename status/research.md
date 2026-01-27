# Maximizing Performance in Gymnasium's CarRacing Environment

## Overview of the CarRacing Environment and Goals

**CarRacing** is a **continuous-control racing task** from pixels, where an agent drives a car on a procedurally generated top-down track. The observation is a 96×96 RGB image (optionally with on-screen indicators like speed and sensors), and the action space can be **continuous** (steering in [-1,1], throttle in [0,1], brake in [0,1]) or discretized to 5 actions (do nothing, turn left, turn right, gas, brake).

The reward structure strongly incentivizes covering new track tiles quickly:
- The agent gains `+1000/N` for each track tile (N = total tiles)
- Incurs `-0.1` per frame, effectively rewarding fast completion
- An episode ends when the car finishes all tiles or goes off-track for too long (resulting in a -100 termination penalty)

**Performance metric:** A full track completion typically yields a score in the high hundreds (e.g. finishing in 732 frames gives ~926.8 points). "Solving" CarRacing is defined as averaging **900+ points over 100 episodes**, meaning the agent nearly perfectly completes each track with minimal time off-track.

Human-level performance is challenging due to the speed and precision required. The **state-of-the-art scores** hover around this 900+ mark. For instance, using advanced PPO-based methods one can average ~913 points, effectively reaching the environment's completion threshold.

---

## Algorithm Comparison: PPO, SAC, TD3, DDPG, and More

Research and benchmarks indicate that **policy-gradient methods** (like PPO) and modern **actor-critic methods** (SAC, TD3) tend to outperform older approaches on CarRacing's high-dimensional pixel input. Below is a comparison of top algorithms and their reported performance:

| Algorithm / Approach | Average Score (100 eps) | Notes |
|---------------------|------------------------|-------|
| **PPO (Beta-distribution policy)** | **~913 ± 26** | *On-policy; uses Beta policy for bounded actions.* Achieved state-of-art performance in literature. |
| **World Models (VAE + RNN + ES)** | ~906 ± 21 | *Model-based latent world with evolution strategy.* First to "solve" CarRacing by imagining future states. |
| **Discrete DQN (with action discretization)** | ~905 ± 24 | *Off-policy Q-learning on discretized actions* – surprisingly competitive when actions are limited to a few discrete controls. |
| **Neuroevolution (Genetic Algorithm)** | ~903 ± 72 | *Evolutionary strategy (CMA-ES or GA) on policy weights.* Achieved near-solving performance (albeit with high variance). |
| **Standard PPO (Gaussian policy)** | ~897 ± 41 | *Baseline PPO with Gaussian continuous actions.* Stable learning but slightly under top scores. |
| **Weight-Agnostic Network (Gaier et al.)** | ~893 ± 74 | *Topology search with fixed-magnitude weights.* Demonstrated a novel approach, nearing the 900 threshold. |
| **A3C (Continuous actions)** | ~591 ± 45 | *Earlier policy-gradient (asynchronous).* Struggles with pixel input, failing to reach high scores. |
| **DDPG (Continuous actions)** | – | *Off-policy DDPG is reported as unstable on CarRacing*, often requiring heavy tuning. TD3, which addresses DDPG's pitfalls, can improve stability, but few published score reports exist. |
| **Random Policy (for reference)** | **~ -32** | Essentially drives off track almost immediately, illustrating the task difficulty. |

### Key Observations

For *continuous control from vision*, **PPO and SAC** are generally top choices due to their stability and exploration ability in continuous action spaces. PPO's on-policy updates yield stable improvements, while SAC's off-policy entropy-regularized learning is highly sample-efficient and can attain very high rewards if tuned.

By contrast, **DQN** without modifications performs poorly (it must discretize the actions; naive DQN got ~343 points). Indeed, early attempts with DQN or standard A3C topped out around 600 (not completing tracks).

**DDPG/TD3** have the capacity for continuous control, but their training on pixels is finicky – PPO's robustness often leads practitioners to favor it over raw DDPG.

The success of a *discrete-action DQN approach* at ~905 points is noteworthy – by constraining the action space (steer hard left/right, full throttle/brake, or coast), the problem becomes more tractable for Q-learning. This indicates that *action-space design* can be as important as the algorithm itself in CarRacing.

### Takeaway

**PPO and SAC emerge as top contenders** for highest scores, with PPO (especially variants like Beta-PPO) proving extremely effective. SAC and TD3 are likewise capable of reaching high performance but may require more careful hyperparameter tuning (e.g. learning rates, exploration noise/entropy) to match PPO's stability.

For example, one comparative study found SAC achieved slightly higher initial rewards than PPO or TD3 in CarRacing, albeit over only short training runs. Overall, **well-tuned policy gradient methods** consistently drive the car to complete tracks and achieve the ~900+ scores required to "solve" the environment.

---

## Advanced Strategies for Improving Scores

Reaching top performance in CarRacing doesn't rely on algorithm choice alone – **training strategies and environment setups** are crucial. Here we outline best practices and enhancements that experienced practitioners use to maximize scores:

### Frame Stacking for Temporal Context

CarRacing is partially observable from a single image frame; the agent benefits from knowing recent motion (speed, turn rate). A common practice is to **stack 4 consecutive frames** as the observation input, effectively providing velocity and steering momentum cues.

For example, using a `VecFrameStack` wrapper to stack 4 × 96×96 images yields a 96×96×12 input tensor. This helps the policy anticipate turns and avoid over-steering by incorporating short-term memory in the state.

### Observation Preprocessing

Simplifying input can improve learning. Many solutions **crop or mask irrelevant parts** of the image (e.g. the HUD indicators or blank sky) and **convert to grayscale** to reduce dimensionality.

Not only does this speed up training (fewer input channels), it also forces the agent to focus on the track geometry rather than color details. (Gymnasium's `CarRacing-v3` already includes a domain-randomized color scheme option, so color is not an essential feature for the agent to rely on.)

### Discrete Action Simplification

Although the environment is continuous, **using a discrete action set** can stabilize training. For instance, one can limit the agent to 5 actions – *steer left, steer right, accelerate, brake, or no-op* – with full throttle/steering in those directions.

This was effectively used in DQN and even PPO implementations. It reduces the action search space and eliminates subtle analog adjustments the network must learn. The drawback is a loss of fine-grained control (e.g. you can't gently steer while lightly accelerating in a purely discrete setup), but agents often overcome this by rapidly alternating actions (the "do-nothing" action allows coasting straight).

**Hybrid approaches** (soft discretization) pick a grid of continuous actions (e.g. half-steering with half-throttle, etc.) to enrich the discrete set – this can strike a balance between simplicity and control fidelity.

### Reward Shaping (Use Caution)

The sparse reward can make exploration difficult initially. Some practitioners add shaping rewards to guide the agent early on – for example, giving a small reward for forward progress or penalizing leaving the track even before the environment's termination triggers.

One recorded approach added a bonus when the agent "dies" off-track during training to encourage it to cover more tiles; however, this artificially inflated evaluation scores and had to be removed for true testing (their unshaped agent actually scored ~820).

**Safer shaping** includes clipping the per-step reward to a max value to prevent reckless speed: e.g. cap the reward at +1 each step so that going extremely fast doesn't disproportionately reward the agent. This can encourage moderate speeds that keep the car on the road, improving long-term tile coverage.

Another shaping trick is implementing a **"timeout"**: if the car is off-track for more than T steps, end the episode early. This spares training time by not letting the agent flail in the dirt.

**Important:** any shaping should ideally be **removed or accounted for during final evaluation** to ensure the agent's policy isn't dependent on artificial rewards.

### Curriculum Learning

Curriculum strategies can significantly expedite learning. Early in training, you can **simplify the task** – for example, fix one easy track layout or allow shorter episode lengths – so the agent at least learns to drive straight and take gentle turns.

As performance improves, gradually ramp up the difficulty: introduce fully random tracks, tighter turns, and require completing the entire lap. One approach is to start episodes with the car partway through a track or to reset quickly after the car goes off-track, giving the agent more frequent "practice" on challenging segments.

Another idea is adjusting the **action repeat/frame skip**: using a frame-skip (repeat the same action for k frames) can make the dynamics more stable and was suggested as a way to ensure the agent's actions have lasting effect and avoid jitter.

In practice, a curriculum might also tune environment parameters (like friction or car acceleration) to be forgiving initially, then approach real settings. While Gymnasium doesn't natively provide multiple difficulty levels, these can be simulated via wrappers or by manipulating the `lap_complete_percent` argument (e.g., require only 50% of tiles for completion early on, then increase to 100%) during training.

### Imitation & Demonstrations

Given the complexity of learning to drive from scratch, **leveraging expert demonstrations** can drastically improve sample efficiency. Imitation learning methods can achieve near-expert driving with surprisingly few demos.

For example, Stanford researchers applied an inverse-Q learning algorithm (IQ-Learn) and attained **expert-level performance with only ~20 human driving demonstrations**.

Approaches like **Behavioral Cloning (BC)** (training the policy network to imitate logged expert actions) can provide a good initialization for RL. The agent learns basic skills – staying on track, following curves – before reinforcement learning even begins.

One can then perform **fine-tuning with RL (e.g. PPO or SAC)** to further improve the policy beyond the demonstrator's capabilities (this is often called *imitation-augmented RL* or **learning from human experience**).

Generative Adversarial Imitation Learning (GAIL) is another popular technique: it learns a reward function that encourages the agent to mimic the expert, which can then be optimized by standard RL. In CarRacing, GAIL and its variants have been shown to train policies that drive smoothly around the track using far fewer environment interactions than pure RL.

**Recommendation:** If you have access to even a small number of human play episodes, use them – either to pretrain the policy or to shape a reward via inverse reinforcement learning. This can cut down training time dramatically on a single GPU and often yields a higher final score. (Many top CarRacing results, including some ~900+ scores on the OpenAI leaderboard, were achieved with help from imitation or supervised pre-training of the network.)

### Recurrent Policies

As an alternative to frame stacking, consider using a **recurrent neural network (RNN/LSTM) policy** to handle partial observability. Stable Baselines3 now offers Recurrent PPO (PPO-LSTM) which can maintain an internal state across time.

An LSTM can, in theory, infer the car's velocity and orientation from the sequence of observations. In practice, frame stacking plus a feedforward CNN has been sufficient for many, but a recurrent policy might excel if the agent needs a longer memory (e.g. remembering a sharp turn is coming after a long straight segment, or recovering from skids).

The pioneering **World Models** approach is essentially an extreme version of this idea: it trained a recurrent world model (MDN-RNN) to encode the dynamics, and a controller that operated in this latent space. The result was a very compact policy that "remembers" upcoming turns and consistently scored ~906.

Thus, using memory (either via frame stacks or an RNN) is a **proven strategy** for mastering CarRacing's long horizons.

---

## Recommended Tools, Implementations, and Hyperparameters

Implementing these algorithms and strategies is made easier by existing RL libraries and baseline repositories:

### Stable Baselines3 (SB3)

This is a reliable choice for quick experimentation. It provides ready-to-use implementations of PPO, SAC, TD3, DDPG, etc., and supports Gymnasium environments. For CarRacing, you'd likely use SB3's CNN policies.

**Example: Initializing and training a PPO agent on CarRacing:**

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CarRacing-v3')            # create environment (v3 is latest Gymnasium version)
model = PPO('CnnPolicy', env, verbose=1)  # use a CNN policy suitable for image input
model.learn(total_timesteps=1_000_000)    # train for 1e6 timesteps (adjust as needed)
model.save("ppo_car_racing")
```

You should wrap the environment to stack frames and normalize observations. SB3's **RL Baselines3 Zoo** (a companion repository) includes hyperparameter configs – for instance, it enables `frame_stack=4` and uses reward normalization for CarRacing.

A typical configuration is to use **8 parallel environment workers** (to stabilize PPO's on-policy updates with more diverse samples) and a relatively small CNN (e.g. 3 convolutional layers) with an LSTM or 2-layer MLP on top.

**Hyperparameter guidelines:**
- **Learning rate**: `3e-4` to `1e-4` is common for PPO
- **Batch sizes**: 256 or 512 (since images are high-dimensional)
- **Clip range**: ~0.1–0.2

For SAC/TD3, a replay buffer on the order of 100k–500k steps and similar CNN architecture can be used. SAC's entropy coefficient can be left to automatic tuning (it tends to work well on this environment to ensure sufficient exploration).

**Note:** The action space in continuous mode is already bounded to [-1,1] for steering and [0,1] for pedals, but SB3's default Gaussian policy doesn't respect those bounds (it will be squashed by tanh). If using SB3, this is fine; if implementing from scratch, consider using a squashed Gaussian or Beta distribution policy to avoid out-of-bound actions. The Beta-PPO paper demonstrated improved learning by using Beta distributions for actions – if using SB3, an equivalent can be achieved by applying a squashing function to Gaussian outputs or using libraries like Tianshou or Spinning Up which allow Beta policies.

### CleanRL

Another excellent resource – it offers single-file implementations of PPO, SAC, etc., with sane default hyperparameters. You can take a CleanRL Atari PPO example and adapt the environment creation to CarRacing.

Ensure to use CNN features (CleanRL often defaults to MLP for classic control, so switch to an image encoder). CleanRL emphasizes clarity, so modifying it for CarRacing (adding FrameStack wrapper, etc.) is straightforward. It's a good way to run experiments if you want to tweak the code easily.

### Sample Code Repositories

Many community examples exist for CarRacing:
- Mike Wang's *"Solving CarRacing with PPO"* blog post provides code snippets and even pre-trained weights for a PPO agent
- Oliver Heilmann's GitHub repo demonstrates DQN, DDQN, and DDPG on CarRacing – useful if you want to see how to discretize actions or modify the environment

Additionally, the Farama Gymnasium docs show how to instantiate CarRacing with different parameters:
```python
gym.make("CarRacing-v3", continuous=False, domain_randomize=True)
```
This uses discrete actions and random colors.

Using these resources as a starting point can save you a lot of tuning time.

### Hyperparameter Hints & Best Practices

- Use a **FrameStack of 4** and consider a **GrayScaleObservation** wrapper (if using Stable Baselines, you might manually convert the RGB to grayscale in a custom wrapper to halve the channels – SB3 doesn't have a built-in grayscale wrapper as of writing, but you can easily add one). Keep the image at 96×96; some have tried downsampling further for speed, but resolution too low makes curves hard to distinguish.

- **Reward normalization** (running mean/std) can help PPO not be thrown off by the scale of returns. CarRacing returns can be ~ +900 at best, which is fine, but normalizing helps if using a general RL pipeline.

- If using **vectorized environments**, make sure to **synchronize resets** so that at least one environment is always providing experience (CarRacing episodes have variable lengths). Gymnasium's `AsyncVectorEnv` can be handy to not stall when one car finishes earlier than others.

- Monitor **evaluation scores** regularly (e.g. run a deterministic evaluation every 10k steps). CarRacing has a lot of variance in tracks, so an agent might score 800 on one random seed and 950 on another – you want to ensure the *average* is trending up. The leaderboard uses 100-episode averages to smooth this out.

- For **single-GPU training**, the environment is usually the bottleneck (rendering and physics in Box2D). Using 8–16 parallel envs can keep the GPU fed with data. Ensure your GPU memory is sufficient for the batch size and model – CarRacing CNN policies aren't huge, but stacking frames multiplies memory usage. In practice, 1M timesteps of PPO training (with, say, 8 envs × 125k steps each) might take on the order of a few hours on a modern GPU. SAC might take longer per step due to off-policy updates, but its sample-efficiency could reach high scores in fewer steps overall.

---

## Expected Performance and Reproducibility

### Score Ranges

With the above algorithms and strategies, you should target an **average score of 900+** on CarRacing, which corresponds to nearly complete lap coverage on most tracks.

In concrete terms, top-performing models consistently hit 900–910 average reward over 100 trials. Scores in the 800s indicate the agent is very good but still occasionally misses a significant chunk of track or spends too long off-road.

For instance, a DQN agent was reported to hover around 800–850 average (with some runs above 900) when carefully tuned. Breaking into the 900+ elite requires minimizing serious crashes or spin-outs.

The table above shows that **913** is a published state-of-art (Beta PPO), and a few independent projects have claimed even slightly higher (~917 in one submission, likely via massive training or additional tricks). However, anything above ~930 is exceedingly rare, as that would mean the car almost never slows down or deviates (recall 1000 would be the theoretical max for instant perfect completion).

### Replicating and Exceeding Top Scores

To reproduce these results, **combine the best practices**: use a robust algorithm (PPO or SAC), extensive frame-stacking or memory, discrete action simplification if using Q-learning, and possibly kickstart training with an imitation phase.

**A recommended recipe for a high-scoring agent:**

> *PPO with 8 environments, 4-frame stack, clipped rewards, early termination on off-track, and a carefully tuned CNN policy (e.g. 3 conv layers + LSTM). Train for on the order of 5–10 million frames.*

This setup, based on community reports and benchmarks, can achieve 900+ average reward. If that alone doesn't reach the mark, consider generating a few human-driving episodes (even imperfect human data can help the agent learn basic control) and use **behavioral cloning** for a few epochs at the start.

One project found that mixing human data with RL (sometimes called **RL with Human Experience, RLHE**) significantly improved stability and final scores.

Finally, remember to **evaluate your best model on many random tracks** to get a statistically sound average. The CarRacing leaderboard uses 100 episodes; you should do similar to claim a "solved" agent.

With careful adherence to these strategies, a single GPU is sufficient to train a champion CarRacing agent that either matches or **exceeds previous bests**.

### Summary

Focus on:
- *A suitable algorithm (PPO/SAC)*
- *Enriched observations (frames or memory)*
- *Thoughtful reward design*
- *Possibly expert guidance*

This combination is proven to produce CarRacing agents that hug the track and consistently post **top scores**.

---

## References

Performance figures and strategies were drawn from published papers and community benchmarks, including the OpenAI Gym leaderboard, the **world models** research, a PPO-Beta distribution study, and expert blogs – these provide further details for reproducing high scores in CarRacing.

- OpenAI Gym Leaderboard (CarRacing entries)
- "World Models" for CarRacing (Ha & Schmidhuber, 2018)
- Petrazzini et al. 2021, *PPO with Beta Distribution* (CarRacing results)
- Michael K.'s blog on solving CarRacing with PPO/SAC (FindingTheta, 2024)
- Mike Wang's *Solving CarRacing with PPO* (tips on actions & observations)
- Stanford SAIL blog on IQ-Learn imitation (CarRacing with demos)
