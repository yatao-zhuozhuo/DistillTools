import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
import requests
from typing import List, Dict, Any
from transformers import modeling_utils


if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

class QwenInference:
    def __init__(self, base_model_path: str, lora_model_path: str = None):
        """
        初始化推理类
        Args:
            base_model_path: 基础模型路径
            lora_model_path: LoRA模型路径，如果为None则使用原始模型
        """
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.setup_model()
    
    def setup_model(self):
        """设置模型和tokenizer"""
        print("正在加载模型...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            revision='refs/pr/4'
        )
        
        # 如果有LoRA模型，加载LoRA权重
        if self.lora_model_path:
            print(f"加载LoRA权重: {self.lora_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_model_path)
        
        print("模型加载完成！")
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话"""
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    def extract_thinking(self, text: str) -> tuple:
        """提取thinking内容和最终回复"""
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_match = re.search(thinking_pattern, text, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # 移除thinking部分，获取最终回复
            final_response = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        else:
            thinking = ""
            final_response = text
        
        return thinking, final_response
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """提取工具调用"""
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_matches = re.findall(tool_pattern, text, re.DOTALL)
        
        tools = []
        for match in tool_matches:
            try:
                tool_data = json.loads(match)
                tools.append(tool_data)
            except:
                continue
        
        return tools
    
    def mock_search(self, query: str) -> str:
        """模拟搜索功能"""
        # 这里可以集成真实的搜索API
        return f"关于'{query}'的搜索结果：这是一个模拟的搜索结果，包含相关信息..."
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """生成回复"""
        # 格式化输入
        formatted_input = self.format_conversation(messages)
        formatted_input += "<|im_start|>assistant\n"
        
        # tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 处理响应
        thinking, final_response = self.extract_thinking(response)
        tool_calls = self.extract_tool_calls(response)
        
        result = {
            "thinking": thinking,
            "response": final_response,
            "tool_calls": tool_calls,
            "raw_response": response
        }
        
        # 如果有工具调用，执行搜索
        if tool_calls:
            search_results = []
            for tool in tool_calls:
                if tool.get("name") == "search":
                    query = tool.get("arguments", {}).get("query", "")
                    if query:
                        search_result = self.mock_search(query)
                        search_results.append({"query": query, "result": search_result})
            
            if search_results:
                result["search_results"] = search_results
                
                # 基于搜索结果生成最终回复
                search_context = "\n".join([f"搜索: {sr['query']}\n结果: {sr['result']}" for sr in search_results])
                follow_up_messages = messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"搜索结果:\n{search_context}\n\n请基于搜索结果回答原问题。"}
                ]
                
                final_result = self.generate_response(
                    follow_up_messages, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )
                result["final_response"] = final_result["response"]
        
        return result

class InteractiveChat:
    def __init__(self, inference_model: QwenInference):
        self.inference_model = inference_model
        self.conversation_history = []
        self.system_prompt = """你是一个智能助手，具有以下能力：
1. 对复杂问题进行深入思考，使用<thinking>标签包围你的思考过程
2. 判断是否需要调用网络搜索工具来获取最新信息
3. 处理网络搜索结果并进行分析

当遇到需要最新信息、实时数据或你不确定的事实时，你应该调用search工具。
格式如下：
<tool_call>
{"name": "search", "arguments": {"query": "搜索关键词"}}
</tool_call>

在回答之前，请在<thinking>标签中进行思考。"""
    
    def start_chat(self):
        """开始交互式对话"""
        print("=== Qwen 智能助手 ===")
        print("输入 'quit' 退出对话")
        print("输入 'clear' 清空对话历史")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("对话历史已清空")
                    continue
                elif not user_input:
                    continue
                
                # 构建消息
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.extend(self.conversation_history)
                messages.append({"role": "user", "content": user_input})
                
                # 生成回复
                print("\n助手: 正在思考...")
                result = self.inference_model.generate_response(messages)
                
                # 显示思考过程
                if result["thinking"]:
                    print(f"\n[思考过程]: {result['thinking']}")
                
                # 显示工具调用
                if result["tool_calls"]:
                    print(f"\n[工具调用]: {result['tool_calls']}")
                
                # 显示搜索结果
                if "search_results" in result:
                    print(f"\n[搜索结果]: {result['search_results']}")
                
                # 显示最终回复
                final_response = result.get("final_response", result["response"])
                print(f"\n助手: {final_response}")
                
                # 更新对话历史
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": result["raw_response"]})
                
                # 保持对话历史在合理长度
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                    
            except KeyboardInterrupt:
                print("\n\n对话已中断")
                break
            except Exception as e:
                print(f"\n错误: {e}")

def main():
    # 配置路径
    base_model_path = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
    lora_model_path = "day4-exercise/qwen_lora_model"  # 如果没有训练完成，设为None
    
    # 创建推理模型
    print("正在初始化模型...")
    inference_model = QwenInference(base_model_path, lora_model_path)
    
    # 开始交互式对话
    chat = InteractiveChat(inference_model)
    chat.start_chat()

if __name__ == "__main__":
    main() 