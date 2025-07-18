# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
lora_model_path = "/remote-home1/yzyang/day4-exercise/day4-exercise/qwen_lora_model"
output_path = "/remote-home1/yzyang/day4-exercise/qwen_full_model"

# 1. 加载base模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
# 2. 加载LoRA权重
model = PeftModel.from_pretrained(model, lora_model_path)
# 3. 合并LoRA到base
model = model.merge_and_unload()
# 4. 保存合并后的完整模型
model.save_pretrained(output_path)

# 5. 可选：保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)