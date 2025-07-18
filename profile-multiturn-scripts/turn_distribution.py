import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# 日志根目录（包含多个 step_i 文件夹）
ROOT_LOG_DIR = "."

# 控制 x 轴标签稀疏显示（每隔几个 step 显示一个标签）
def reduce_ticks(labels, step=5):
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]

# 结构：step_i -> turn -> list of durations
step_stats = defaultdict(lambda: defaultdict(list))

# 遍历所有 step_i 目录
for step_dir in os.listdir(ROOT_LOG_DIR):
    full_step_path = os.path.join(ROOT_LOG_DIR, step_dir)
    if not os.path.isdir(full_step_path) or not step_dir.startswith("step_"):
        continue

    for fname in os.listdir(full_step_path):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(full_step_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") != "async_rollout_request_complete":
                        continue
                    turns = entry["extra"].get("turns")
                    duration = entry.get("duration_sec", 0)
                    if isinstance(turns, int):
                        step_stats[step_dir][turns].append(duration)
                except Exception:
                    continue

# 所有 step（按数字排序）
steps_sorted = sorted(step_stats.keys(), key=lambda s: int(s.split('_')[-1]))

# 所有 turn（例如 turn=1, 2, 3...）
all_turn_ids = sorted({turn for stats in step_stats.values() for turn in stats.keys()})

# 每个 turn 的 count（step × turn）
turn_counts_by_step = {
    turn: [len(step_stats[step].get(turn, [])) for step in steps_sorted]
    for turn in all_turn_ids
}

# 每个 turn 的平均耗时（step × turn）
turn_avg_duration_by_step = {
    turn: [
        sum(step_stats[step][turn]) / len(step_stats[step][turn])
        if step_stats[step].get(turn) else None
        for step in steps_sorted
    ]
    for turn in all_turn_ids
}

# ---------- 图一：每个 step 的 turn count 堆叠图 ----------
# plt.figure(figsize=(16, 6))
# bottom = [0] * len(steps_sorted)
# for turn in all_turn_ids:
#     counts = turn_counts_by_step[turn]
#     plt.bar(steps_sorted, counts, bottom=bottom, label=f"turn={turn}")
#     bottom = [b + c for b, c in zip(bottom, counts)]
# plt.ylabel("Count")
# plt.title("Turn Count per Step (All Turns)")
# plt.xticks(ticks=range(len(steps_sorted)), labels=reduce_ticks(steps_sorted, 5), rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---------- 图二：每个 step 的 turn 平均耗时折线图 ----------
# ---------- 图二：每个 step 的 turn 平均耗时折线图 ----------
plt.figure(figsize=(16, 6))
for turn in all_turn_ids:
    durations = turn_avg_duration_by_step[turn]
    plt.plot(steps_sorted, durations, marker="o", label=f"turn={turn} avg duration")
plt.ylabel("Avg Duration (sec)")
plt.title("Avg Duration per Turn per Step")
plt.xticks(ticks=range(len(steps_sorted)), labels=reduce_ticks(steps_sorted, 5), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("turn_avg_duration_per_step.png", dpi=200)

