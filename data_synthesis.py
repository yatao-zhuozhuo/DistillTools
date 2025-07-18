import json
import random
import os
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from search_tool import FakeSearch

class DataSynthesizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.search_tool = FakeSearch()
        self.setup_model()
        
    def setup_model(self):
        """初始化vllm模型"""
        print(f"正在加载模型: {self.model_path}")
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,  # 根据GPU数量调整
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
        )
        print("模型加载完成！")
    
    def query_llm_local(self, messages: List[Dict[str, str]], temperature: float = 0.8, max_new_tokens: int = 4096) -> str:
        """本地推理函数"""
        # 格式化消息为文本
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
        
        # 添加Assistant标识
        prompt_text += "Assistant: "
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        
        try:
            outputs = self.model.generate([prompt_text], sampling_params)
            generated_text = outputs[0].outputs[0].text
            return generated_text
        except Exception as e:
            print(f"推理错误: {e}")
            return "抱歉，我无法在此时提供回复。"

    def load_questions(self, filename: str) -> List[str]:
        """从文件加载问题"""
        questions = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 跳过空行和注释
                        questions.append(line)
            print(f"从 {filename} 加载了 {len(questions)} 个问题")
            return questions
        except Exception as e:
            print(f"加载问题文件 {filename} 失败: {e}")
            return []

    def generate_thinking_content(self, question: str) -> str:
        """生成思考内容"""
        thinking_prompt = f"""请对以下问题进行深入思考，详细分析问题的各个方面，展示你的推理过程：

问题：{question}

请只输出你的思考过程，不要包含最终答案。思考过程应该包括：
1. 问题分析
2. 相关概念或知识点
3. 推理步骤
4. 可能的解决方法或角度"""

        messages = [{"role": "user", "content": thinking_prompt}]
        return self.query_llm_local(messages, temperature=0.7, max_new_tokens=1024)

    def generate_final_answer(self, question: str, context: str = "") -> str:
        """生成最终答案"""
        if context:
            answer_prompt = f"""基于以下信息，请回答问题：

问题：{question}

相关信息：
{context}

请给出清晰、准确的答案："""
        else:
            answer_prompt = f"""请回答以下问题：

问题：{question}

请给出清晰、准确的答案："""

        messages = [{"role": "user", "content": answer_prompt}]
        return self.query_llm_local(messages, temperature=0.6, max_new_tokens=2048)

    def extract_search_keywords(self, question: str) -> str:
        """从问题中提取搜索关键词"""
        keyword_prompt = f"""请从以下问题中提取最重要的搜索关键词，用于网络搜索。只输出关键词，不要其他内容：

问题：{question}

搜索关键词："""

        messages = [{"role": "user", "content": keyword_prompt}]
        keywords = self.query_llm_local(messages, temperature=0.3, max_new_tokens=128)
        return keywords.strip()

    def generate_system_prompt(self) -> str:
        """生成系统提示词"""
        return """你是一个智能助手，具有以下能力：
1. 对复杂问题进行深入思考，使用<thinking>标签包围你的思考过程
2. 判断是否需要调用网络搜索工具来获取最新信息
3. 处理网络搜索结果并进行分析

当遇到需要最新信息、实时数据或你不确定的事实时，你应该调用search工具。
格式如下：
<tool_call>
{"name": "search", "arguments": {"query": "搜索关键词"}}
</tool_call>

在回答之前，请在<thinking>标签中进行思考。"""

    def generate_thinking_sample(self, question: str) -> Dict[str, Any]:
        """生成需要思考的样本"""
        # 生成思考内容
        thinking_content = self.generate_thinking_content(question)
        
        # 生成最终答案
        final_answer = self.generate_final_answer(question)
        
        # 组合完整回复：思考标签 + 思考内容 + 最终答案
        complete_response = f"<thinking>\n{thinking_content}\n</thinking>\n\n{final_answer}"
        
        return {
            "messages": [
                {"role": "system", "content": self.generate_system_prompt()},
                {"role": "user", "content": question},
                {"role": "assistant", "content": complete_response}
            ],
            "type": "thinking",
            "category": "reasoning"
        }

    def generate_search_sample(self, question: str) -> Dict[str, Any]:
        """生成需要搜索的样本"""
        # 提取搜索关键词
        search_keywords = self.extract_search_keywords(question)
        
        # 执行搜索
        try:
            search_results = self.search_tool.search(search_keywords, top_k=3)
            search_results_text = "\n".join([f"{i+1}. {result}" for i, result in enumerate(search_results)])
        except Exception as e:
            print(f"搜索执行失败: {e}")
            search_results_text = f"搜索关于'{search_keywords}'的信息时出现错误"
            search_results = [f"搜索'{search_keywords}'时发生错误"]
        
        # 生成基于搜索结果的答案
        final_answer = self.generate_final_answer(question, search_results_text)
        
        # 组合完整回复：工具调用标签 + 搜索结果 + 最终答案
        tool_call_content = f'{{"name": "search", "arguments": {{"query": "{search_keywords}"}}}}'
        complete_response = f"<tool_call>\n{tool_call_content}\n</tool_call>\n\n根据搜索结果：\n{search_results_text}\n\n{final_answer}"
        
        return {
            "messages": [
                {"role": "system", "content": self.generate_system_prompt()},
                {"role": "user", "content": question},
                {"role": "assistant", "content": complete_response}
            ],
            "type": "search",
            "category": "factual_query",
            "search_query": search_keywords,
            "search_results": search_results
        }

    def synthesize_data(self, num_thinking_samples: int = 50, num_search_samples: int = 50) -> List[Dict[str, Any]]:
        """合成训练数据"""
        data = []
        
        # 加载问题
        thinking_questions = self.load_questions("question_without_search.txt")
        search_questions = self.load_questions("question_with_search.txt")
        
        # 生成思考样本
        if thinking_questions:
            print(f"正在生成 {num_thinking_samples} 个思考样本...")
            for i in range(num_thinking_samples):
                question = random.choice(thinking_questions)
                try:
                    sample = self.generate_thinking_sample(question)
                    data.append(sample)
                    print(f"已生成思考样本 {i+1}/{num_thinking_samples}")
                except Exception as e:
                    print(f"生成思考样本失败 {i+1}: {e}")
        else:
            print("警告：无法加载思考类问题，跳过思考样本生成")
        
        # 生成搜索样本
        if search_questions:
            print(f"正在生成 {num_search_samples} 个搜索样本...")
            for i in range(num_search_samples):
                question = random.choice(search_questions)
                try:
                    sample = self.generate_search_sample(question)
                    data.append(sample)
                    print(f"已生成搜索样本 {i+1}/{num_search_samples}")
                except Exception as e:
                    print(f"生成搜索样本失败 {i+1}: {e}")
        else:
            print("警告：无法加载搜索类问题，跳过搜索样本生成")
        
        print(f"数据合成完成！共生成 {len(data)} 个有效样本")
        return data

def main():
    # 模型路径
    model_path = "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    # 创建数据合成器
    synthesizer = DataSynthesizer(model_path)
    
    # 合成数据
    data = synthesizer.synthesize_data(num_thinking_samples=200, num_search_samples=200)
    
    # 保存数据
    output_file = "synthetic_data_trained.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据合成完成！共生成 {len(data)} 个样本，保存到 {output_file}")

if __name__ == "__main__":
    main() 