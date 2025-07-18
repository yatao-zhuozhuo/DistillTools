import requests
import time
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
import torch
import numpy as np
from vllm import LLM, SamplingParams

# Load environment variables from .env file
load_dotenv()

# 全局模型缓存
_model_cache = {}

API_KEYS = {
    'api': os.getenv('API_KEY','sk-ygqhtmfalsbdswixgicjhjwxxmbpldhnpzyiiyvsmbsmebsr'),                    
}

API_URLS = {
    'api': 'https://api.siliconflow.cn/v1/chat/completions',
}


def get_or_create_model(model_path, tensor_parallel_size=None):
    """获取或创建模型实例，实现单例模式"""
    global _model_cache
    
    # 自动检测GPU数量
    if tensor_parallel_size is None:
        try:
            import torch
            tensor_parallel_size = torch.cuda.device_count()
            print(f"自动检测到 {tensor_parallel_size} 个GPU，将使用张量并行")
        except:
            tensor_parallel_size = 1
            print("无法检测GPU数量，使用单卡模式")
    
    # 创建缓存键
    cache_key = f"{model_path}_{tensor_parallel_size}"
    
    # 如果模型已经加载，直接返回
    if cache_key in _model_cache:
        print(f"使用已缓存的模型: {model_path}")
        return _model_cache[cache_key]
    
    # 否则创建新模型
    print(f"正在加载模型: {model_path}")
    local_model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,  # 多卡并行
        max_model_len=8192,  # 可根据显存调整
        gpu_memory_utilization=0.9,  # GPU显存使用率
        enforce_eager=False,  # 使用CUDA图加速
    )
    
    # 缓存模型
    _model_cache[cache_key] = local_model
    print(f"模型已加载并缓存: {model_path}")
    return local_model

def query_llm_local(messages, model_path, temperature=1.0, max_new_tokens=4096, disable_think=True, tensor_parallel_size=None):
    """本地推理函数 - 使用vllm，支持多卡，模型单例"""
    
    # 获取或创建模型实例
    local_model = get_or_create_model(model_path, tensor_parallel_size)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens
    )
    
    # 将消息列表转换为模型可接受的格式
    if isinstance(messages, list):
        # 构建符合模型期望的提示格式
        prompt_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_text += f"{content}\n\n"
            elif role == "user":
                prompt_text += f"User: {content}\n\n"
            elif role == "assistant":
                prompt_text += f"Assistant: {content}\n\n"
    else:
        # 如果已经是字符串，直接使用
        prompt_text = messages
    
    try:
        # 使用vllm生成回复
        outputs = local_model.generate([prompt_text], sampling_params) 
        # 获取生成的文本
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
        
    except Exception as e:
        print(f"Error in local inference: {e}")
        # 返回一个默认回复而不是None，避免后续处理出错
        return "I cannot provide a response at this time."

def query_llm(messages, model, temperature=1, max_new_tokens=4096, use_local_model=False, disable_think=True, tensor_parallel_size=None):
    if use_local_model:
        # 使用本地推理
        return query_llm_local(messages, model, temperature, max_new_tokens, disable_think, tensor_parallel_size)
    else:
        # 使用API推理
        api_key, api_url = API_KEYS['api'], API_URLS['api']
        
        tries = 0
        while tries < 5:
            tries += 1
            try:
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': "2023-06-01",
                }
                    
                resp = requests.post(api_url, json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                }, headers=headers, timeout=600)
                
                if resp.status_code != 200:
                    raise Exception(resp.text)
                resp = resp.json()
                break
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                if "maximum context length" in str(e):
                    raise e
                elif "triggering" in str(e):
                    return 'Trigger OpenAI\'s content management policy.'
                print("Error Occurs: \"%s\"        Retry ..." % (str(e)))
                time.sleep(1)
        else:
            print("Max tries. Failed.")
            return None
        
        try:
            if 'content' not in resp["choices"][0]["message"] and 'content_filter_results' in resp["choices"][0]:
                resp["choices"][0]["message"]["content"] = 'Trigger OpenAI\'s content management policy.'
            else:
                return resp["choices"][0]["message"]["content"]
        except: 
            return None

def clear_model_cache():
    """清理模型缓存，释放GPU内存"""
    global _model_cache
    for cache_key in list(_model_cache.keys()):
        del _model_cache[cache_key]
    _model_cache.clear()
    print("模型缓存已清理")

if __name__ == "__main__":
    messages = "Hello, how are you?"
    model = "/remote-home1/yzyang/Game-for-LLM/checkpoints/maze_rl/maze_multiturn_grpo5/global_step_80/actor/huggingface"
    print(query_llm_local(messages, model))