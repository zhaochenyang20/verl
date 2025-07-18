import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# 根目录按你的结构修改
ROOT_LOG_DIR = "."
MAX_STEP = 80

# 统计结构: turn_id -> stage -> [耗时]
stage_stats = defaultdict(lambda: defaultdict(list))

for i in range(1, MAX_STEP+1):
    step_dir = f"step_{i}"
    full_step_path = os.path.join(ROOT_LOG_DIR, step_dir)
    if not os.path.isdir(full_step_path):
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
                    turn_timings = entry["extra"].get("turn_timings", [])
                    for turn_obj in turn_timings:
                        turn_id = turn_obj.get("turn")
                        for stage in [
                            "pre_process_duration",
                            "engine_call_duration",
                            "post_process_duration",
                            "tool_parsing_duration",
                            # 你如果还想加别的字段可自行补充
                        ]:
                            if stage in turn_obj:
                                stage_stats[turn_id][stage].append(turn_obj[stage])
                except Exception as e:
                    continue

# 计算平均值
import numpy as np
avg_stage_durations = defaultdict(dict)
for turn_id, stages in stage_stats.items():
    for stage, values in stages.items():
        if values:
            avg_stage_durations[turn_id][stage] = np.mean(values)

# 输出表格
print("Turn\t" + "\t".join(["pre_process", "engine_call", "post_process", "tool_parse"]))
for turn_id in sorted(avg_stage_durations.keys()):
    row = [f"{avg_stage_durations[turn_id].get(stage, 0):.6f}" for stage in [
        "pre_process_duration", "engine_call_duration", "post_process_duration", "tool_parsing_duration"
    ]]
    print(f"{turn_id}\t" + "\t".join(row))

print("每个 turn 的样本数：")
for turn_id in sorted(stage_stats.keys()):
    # 用其中一个 stage 的样本量做代表（比如 pre_process_duration）
    # 你也可以取 max(len(v) for v in stage_stats[turn_id].values())
    sample_count = len(next(iter(stage_stats[turn_id].values())))
    print(f"Turn {turn_id}: {sample_count}")
print()

# 画图：每个 stage 横向展示不同 turn 的平均耗时
# stages = ["pre_process_duration", "engine_call_duration", "post_process_duration", "tool_parsing_duration"]
import numpy as np
import matplotlib.pyplot as plt

stages = ["engine_call_duration"]
turn_ids = sorted(avg_stage_durations.keys())

plt.figure(figsize=(10, 5))
bar_width = 0.5
x = np.arange(len(turn_ids))

y = [avg_stage_durations[turn_id].get("engine_call_duration", 0) for turn_id in turn_ids]
plt.bar(x, y, width=bar_width, label="engine_call_duration")

plt.xticks(x, [str(t) for t in turn_ids])
plt.xlabel("Turn")
plt.ylabel("Average Engine Call Duration (sec)")
plt.title("Avg Engine Call Duration per Turn (first 80 steps)")
plt.legend()
plt.tight_layout()
plt.savefig("avg_engine_call_duration_per_turn.png", dpi=200)
plt.show()
