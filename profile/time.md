# Profiling Results of verl on GSM8K multi-turn

The profile docs are based on this [script](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/debug-tp-2-multi-turn.md) and this [run](https://wandb.ai/zhaochenyang20/verl-profile-sglang-qwen2.5/runs/2xv2fcad?nw=nwuserzhaochenyang20).

Note that this is running in TP 1.

In open-source version, we do not log the "turns in each request", we only log the total time the rollout and following training.

I am not sure what's the response length means in multi-turn? I think it should mean the total number of tokens of multi-turn and the response of tools.

## Sampled Time Distribution Across Four Steps

1. step 1

total: 107
update actor: 30
reward: 1
reshard: 2
ref: 11
recompute old log probs: 11
generate sequences: 50
total gen time: 55
adv: 0.08

response length mean: 360

2. step 10

total: 107
update actor: 31
reward: 1
reshard: 4
ref: 9
recompute old log probs: 9
generate sequences: 51
total gen time: 57
adv: 0.05

response length mean: 407

3. step 30

total: 112
update actor: 31
reward: 1
reshard: 4
ref: 9
recompute old log probs: 9
generate sequences: 55
total gen time: 61
adv: 0.07

response length mean: 431

4. step 70

total: 144
update actor: 33
reward: 1
reshard: 4
ref: 9
recompute old log probs: 9
generate sequences: 82
total gen time: 88
adv: 0.06

response length mean: 415

Note that step 20, 40, 60 is a spike on the total time since verl does validation every 20 steps. But step 70 is indeed a spike.

I am not sure why step 70 takes a long time but just with a normal response length. I shall dive into the log.

## Indetailed Log for Step 70

ref to [step-70.log](./step-70.txt)

Here is something interesting: