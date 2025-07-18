import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from pathlib import Path

# 设置日志根目录
BASE_LOG_DIR = "."
STEP_GLOB = os.path.join(BASE_LOG_DIR, "step_*", "worker_*.jsonl")

# 只保留这 3 个事件和总时长
FOCUS_EVENTS = {
    "async_generate_duration",
    "preprocessing_duration",
    "barrier_wait_duration",
    "total_step_duration"
}

# 读取所有日志
records = []
for file in tqdm(glob(STEP_GLOB), desc="Loading logs"):
    log_path = Path(file)
    step_num = int(log_path.parent.name.split("_")[1])
    with open(file, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                # 确保字段存在
                if (
                    isinstance(j, dict)
                    and "event" in j
                    and "duration_sec" in j
                    and j["event"] in FOCUS_EVENTS
                ):
                    records.append({
                        "step": step_num,
                        "event": j["event"],
                        "duration_sec": float(j["duration_sec"]),
                    })
            except Exception as e:
                continue

# 转 DataFrame
if not records:
    raise ValueError("❌ 没有有效的日志数据被读取，请检查路径或 event 名称。")

df = pd.DataFrame(records)

# pivot 成 step × event 表
pivot_df = df.pivot_table(index="step", columns="event", values="duration_sec", aggfunc="mean")

# 确保 total_step_duration 存在
if "total_step_duration" not in pivot_df.columns:
    raise ValueError("❌ total_step_duration 缺失，无法计算占比。")

# 计算百分比
for event in ["async_generate_duration", "preprocessing_duration", "barrier_wait_duration"]:
    if event in pivot_df.columns:
        pivot_df[f"{event}_pct"] = pivot_df[event] / pivot_df["total_step_duration"] * 100

# 画图
plt.figure(figsize=(10, 5))
for event in ["async_generate_duration", "preprocessing_duration", "barrier_wait_duration"]:
    col = f"{event}_pct"
    if col in pivot_df.columns:
        plt.plot(
            pivot_df.index,
            pivot_df[col],
            label=event.replace("_duration", ""),
            marker='o'
        )

plt.xlabel("Step")
plt.ylabel("Percent of Total Duration (%)")
plt.title("Step-wise Time Proportion for Key Events")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("stepwise_key_event_proportion.png")
plt.show()
