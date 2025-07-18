from inference import QwenInference
import json

# 配置模型路径
base_model_path = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
lora_model_path = "/remote-home1/yzyang/day4-exercise/day4-exercise/qwen_lora_model"  # 如无LoRA可设为None

# 初始化推理模型
inference_model = QwenInference(base_model_path, lora_model_path)

# 读取数据
with open('synthetic_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

thinking_total = 0
thinking_hit = 0
search_total = 0
search_hit = 0

for item in data:
    msgs = item.get('messages', [])
    if len(msgs) < 2:
        continue
    input_msgs = [msgs[0], msgs[1]]
    result = inference_model.generate_response(input_msgs, max_new_tokens=4096)
    output = result.get("raw_response", "")
    print(output)

    if item.get('type') == 'thinking':
        thinking_total += 1
        if '<thinking>' in output:
            thinking_hit += 1
    elif item.get('type') == 'search':
        search_total += 1
        if '<tool_call>' in output:
            search_hit += 1

print(f"思考类样本总数: {thinking_total}")
print(f"推理输出含<thinking>标签: {thinking_hit}")
if thinking_total:
    print(f"思考类标签命中率: {thinking_hit / thinking_total * 100:.2f}%")

print(f"搜索类样本总数: {search_total}")
print(f"推理输出含<tool_call>标签: {search_hit}")
if search_total:
    print(f"搜索类标签命中率: {search_hit / search_total * 100:.2f}%") 