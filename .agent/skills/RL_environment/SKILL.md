Here is your **clean `skills.md` file** (ready to copy/paste into your repo):

---

````markdown
# RL Environment & Reward Modeling Skills

## 0. Mental Model (Non-negotiable)

An RL environment for LLM agents is NOT just Gym.

It is:
> A stateful, tool-augmented, partially observable system that produces verifiable or learned rewards under constraints.

Core loop:

obs_t → model → action_t → env.step(action_t)  
      → (obs_{t+1}, reward_t, done, info)

Standard interface:
- reset()
- step(action)
- state()

---

## 1. Environment Design Skills

### 1.1 Define the MDP (Extended)

Define:
- State (S)
- Action (A)
- Transition (T)
- Reward (R)

Also handle:
- Partial observability (POMDP)
- Tool usage
- Multi-step reasoning

---

### 1.2 State Representation

Support:
- Text states
- Structured states (JSON/dict)
- Hybrid states

Example:
```python
state = {
  "instruction": ...,
  "history": [...],
  "tools": {...},
  "hidden_state": ...
}
````

---

### 1.3 Action Space Design

Support:

* Discrete actions
* Structured actions (tool calls, APIs, code)

Example:

```python
action = {
  "type": "tool_call",
  "tool": "search",
  "args": {...}
}
```

---

### 1.4 Execution Layer

Environment should run in:

* Local process
* Docker container
* Remote server

Goals:

* Reproducibility
* Sandboxing
* Scalability

---

### 1.5 Determinism vs Stochasticity

Include:

* Seed control
* Randomization hooks
* Domain randomization

---

## 2. Reward Modeling Skills

### 2.1 Reward Types

#### (A) Verifiable Rewards (RLVR)

* Exact correctness (math, code, logic)
* Binary or scalar

#### (B) Learned Rewards

* Trained from preferences
* Human or AI feedback

#### (C) Programmatic Rewards

* Rules
* Heuristics
* Simulations

---

### 2.2 Reward Design Principles

* Dense vs Sparse rewards
* Credit assignment (step vs trajectory)
* Reward decomposition

Example:

```python
reward =
  w1 * correctness +
  w2 * efficiency +
  w3 * safety
```

---

### 2.3 RLVR Data Generation

Generate tasks from raw data:

```python
def generate_task(text):
    mask_reasoning()
    create_distractors()
    compute_answer()
    return task
```

---

### 2.4 Self-Rewarding Systems

Pipeline:

output → critique model → score → RL update

---

## 3. Environment Patterns

### 3.1 Tool-Use Environments

* Search
* Code execution
* APIs

---

### 3.2 Simulation Environments

* Games
* Coding tasks
* Deterministic evaluation

---

### 3.3 Curriculum Environments

* Increasing difficulty
* Auto-generated tasks

---

### 3.4 Multi-Agent Environments

* Cooperation
* Competition
* Negotiation

---

## 4. Training Integration Skills

### 4.1 Rollouts

* Custom rollout loop
* External reward computation

---

### 4.2 Parallelization

* Batched environments
* Async rollouts
* Distributed simulation

---

### 4.3 Logging

Track:

```python
episode_return
success_rate
trajectory_length
failure_modes
```

---

## 5. Safety & Constraints

* Sandbox execution
* API limits
* Permission systems

---

## 6. Debugging Skills

### 6.1 Trajectory Inspection

(state, action, reward)

---

### 6.2 Reward Hacking Detection

* Loophole exploitation
* Shortcut behavior

---

### 6.3 Counterfactual Testing

* Modify rewards
* Observe policy changes

---

## 7. Advanced Capabilities

### 7.1 Synthetic Environments

* Learn environment models

---

### 7.2 Domain Randomization

* Improve generalization

---

### 7.3 Meta-RL

* Adaptive environments

---

## 8. Implementation Template

```python
class Env:
    def __init__(self, config):
        self.state = None

    def reset(self):
        self.state = self.init_state()
        return self.observe()

    def step(self, action):
        next_state = self.transition(self.state, action)
        reward = self.compute_reward(self.state, action, next_state)
        done = self.is_done(next_state)

        self.state = next_state
        return self.observe(), reward, done, {}

    def observe(self):
        return serialize(self.state)
```

---

## 9. Agent Requirements

### Core

* Define environment
* Implement step/reset/state
* Design rewards
* Integrate with RL training

### Advanced

* Generate synthetic tasks
* Build tool environments
* Scale rollouts
* Detect reward hacking

---

## 10. Heuristics

* Bad reward > bad model
* Environment quality > algorithm choice
* Verifiable rewards scale best
* If reward is unclear → environment is broken

---

## 11. Key Takeaways

1. Environment = interface + constraints + reward
2. Reward design is the main challenge
3. Scale comes from synthetic data + standardization

```

---

If you want next step, I can:
- convert this into a **code-generating agent prompt**
- or build a **real OpenEnv-compatible environment template repo** tailored to your ML goals
```
