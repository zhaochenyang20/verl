# Over Sampling Drop Methods 实现说明

## 背景

为了解决 long tail 问题，我们实现了 over sampling 策略。相比于 partial rollout，此策略更加粗暴：没有完成的 requests 将会直接被丢弃。

## 核心设计

通过三个核心函数实现：

1. **`monitor_and_cancel`**: 监控完成数量，一旦达到目标立即取消剩余任务，并向 engine 发送 abort 信号
2. **`process_request_with_monitoring`**: 处理单个请求，根据完成情况返回真实结果或 padding 数据  
3. **`run_with_cancellation`**: 同时启动监控和请求处理任务

## Drop 策略

通过环境变量 `drop_method` 控制处理被 drop 请求的方式：

### Strategy 1: `padding` (全 Padding)
```bash
export drop_method=padding
```

**行为:**
- response_ids: 全部设为 `pad_token_id`
- response_attention_mask: 全部设为 `0`
- response_loss_mask: 全部设为 `0` (确保被训练时忽略)
- reward_scores: **计算真实 reward** (修复后)

**适用场景:** 完全忽略被 drop 的请求，但仍需要 reward 信息用于统计

### Strategy 2: `partial_zero_mask` (部分结果 + 零损失掩码)
```bash
export drop_method=partial_zero_mask
```

**行为:**
- response_ids: 保留已生成部分，不足部分用 padding 补齐
- response_attention_mask: 只对已生成部分设为 `1`
- response_loss_mask: 全部设为 `0` (不参与训练)
- reward_scores: 计算真实 reward

**适用场景:** 保留部分生成结果用于分析，但不参与训练

### Strategy 3: `partial_normal_mask` (部分结果 + 正常损失掩码)
```bash
export drop_method=partial_normal_mask
```

**行为:**
- response_ids: 保留已生成部分，不足部分用 padding 补齐
- response_attention_mask: 只对已生成部分设为 `1`  
- response_loss_mask: 只对已生成部分设为 `1` (参与训练)
- reward_scores: 计算真实 reward

**适用场景:** 让部分生成的结果也参与训练，可能有助于学习中间状态

## 关键修复

### 修复前 (Strategy 1):
```python
reward_scores={},  # 空的reward_scores - 问题!
```

### 修复后 (Strategy 1):
```python
# 计算真实的reward_scores，即使response是padding的
reward_scores = await self._calculate_reward_scores_for_partial_request_async(original_req)
```

**修复原因:** 
- GSM8K 等任务的 reward 计算基于工具状态，不依赖于 response 内容
- 即使 response 是 padding 的，仍应该调用 `tool.calc_reward()` 获取真实分数
- 确保训练时 reward 信息的一致性和完整性

## 测试

```bash
# 测试单个策略
export drop_method=padding && python test_drop_methods.py
export drop_method=partial_zero_mask && python test_drop_methods.py
export drop_method=partial_normal_mask && python test_drop_methods.py

# 测试所有策略
export drop_method=all && python test_drop_methods.py
```

## Over Sampling 控制

通过环境变量 `OVER_SAMPLE_RATE` 控制目标完成率：

```bash
export OVER_SAMPLE_RATE=0.8  # 80% 完成时开始 abort
```

## 实现细节

### Reward 计算逻辑
```python
async def _calculate_reward_scores_for_partial_request_async(self, original_req):
    reward_scores = {}
    
    # 计算工具reward  
    if original_req.tools_kwargs and self._tool_map:
        for tool_name in original_req.tools_kwargs.keys():
            tool = self._tool_map[tool_name]
            try:
                # 调用工具的calc_reward方法
                reward = await tool.calc_reward(original_req.request_id, ...)
                await tool.release(original_req.request_id, ...)
                reward_scores[tool_name] = reward
            except Exception as e:
                logger.warning(f"Failed to calculate reward for tool {tool_name}: {e}")
                reward_scores[tool_name] = 0.0
    
    # 添加user_turn_rewards
    reward_scores["user_turn_rewards"] = []
    return reward_scores
```

### 监控和取消逻辑
```python
async def monitor_and_cancel():
    while completed_count < target_completion:
        await asyncio.sleep(0.1)
    
    # 达到目标后的操作:
    # 1. 取消剩余任务
    # 2. 向 engine 发送 abort 信号
    # 3. 等待所有任务完成或被取消
```

## 性能考虑

1. **并发处理**: 所有请求并发处理，不等待慢请求
2. **提前终止**: 达到目标完成率后立即终止剩余请求
3. **资源释放**: 及时调用 `tool.release()` 释放资源
4. **异常处理**: 将失败的请求也转换为合理的 padding 数据

## 选择建议

- **训练阶段**: 推荐 `padding` 或 `partial_zero_mask`，避免不完整数据影响训练
- **分析阶段**: 可使用 `partial_normal_mask` 了解模型的中间行为
- **生产环境**: 根据具体任务特性选择，GSM8K 等推荐 `padding`