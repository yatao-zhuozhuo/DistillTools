from llm_api import query_llm
import json

class FakeSearch:
    def __init__(self):
        # 使用硅基流动的模型
        self.model = "Pro/THUDM/glm-4-9b-chat"
        
    def chat(self, messages: list):
        """使用llm_api中的查询方法"""
        result = query_llm(
            messages=messages,
            model=self.model,
            temperature=0.5,
            max_new_tokens=4096,
            use_local_model=False  # 使用API模式
        )
        return result

    def search(self, keyword, top_k=3):
        """搜索方法，返回模拟的搜索结果"""
        messages = [{
            "role": "user",
            "content": f"请你扮演一个搜索引擎，对于任何的输入信息，给出 2 个合理的搜索结果，以列表的方式呈现。每个搜索结果独占一行，每行的内容是不超过500字的搜索结果。请直接给出搜索结果，不要有其他说明文字。\n\n输入: {keyword}"
        }]
        
        try:
            result = self.chat(messages)
            if result:
                # 分割结果并清理
                res_list = result.strip().split("\n")
                # 过滤空行和无效内容
                filtered_results = []
                for res in res_list:
                    res = res.strip()
                    if len(res) > 10:  # 确保内容有意义
                        # 去除可能的编号前缀
                        if res.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                            res = res[2:].strip()
                        elif res.startswith(('1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、', '10、')):
                            res = res[2:].strip()
                        filtered_results.append(res)
                
                return filtered_results[:top_k]
            else:
                return [f"未找到关于'{keyword}'的相关信息"]
        except Exception as e:
            print(f"搜索错误: {e}")
            return [f"搜索'{keyword}'时发生错误"]

if __name__ == "__main__":
    import sys
    search = FakeSearch()
    if len(sys.argv) > 1:
        results = search.search(sys.argv[1], 5)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
    else:
        print("请提供搜索关键词")
