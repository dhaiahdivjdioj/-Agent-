import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- 1. 配置与环境 ---
# 请确保在环境变量中设置了 OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = "your-api-key"

class CodeDocumentationAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        """
        初始化 Agent，配置 LLM 和提示词模板
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # 低温度以保证技术准确性
        )
        
        # 定义文档生成的 Prompt 模板
        # 这是一个“思维链”设计，要求 AI 先分析再输出
        template = """
        你是一位资深技术文档工程师。请根据以下代码内容，生成一份详细的技术文档。
        
        ### 任务要求：
        1. 简要描述该模块/函数的功能。
        2. 分析输入参数和返回值类型。
        3. 提取核心逻辑步骤（如果有复杂算法）。
        4. 给出一个简单调用示例。
        5. 格式要求：使用标准 Markdown 格式。
        
        ### 待分析文件路径：
        {file_path}
        
        ### 代码内容：
        ```{language}
        {code_content}
        ```
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)

    def read_code_file(self, file_path: str) -> str:
        """读取文件内容"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_language_from_extension(self, file_path: str) -> str:
        """简单根据后缀判断语言"""
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java'}
        return mapping.get(ext, 'text')

    def generate_documentation(self, file_path: str) -> str:
        """
        核心逻辑流：读取文件 -> 组装 Prompt -> 调用 LLM -> 输出文档
        """
        print(f"🤖 Agent 正在扫描文件: {file_path}...")
        
        # 1. 读取代码
        code_content = self.read_code_file(file_path)
        language = self.get_language_from_extension(file_path)
        
        # 2. 构建链式调用 (Chain)
        # 使用 RunnablePassthrough 确保上下文传递
        chain = (
            {
                "code_content": RunnablePassthrough(), 
                "file_path": lambda x: file_path,
                "language": lambda x: language
            }
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        print(f"🧠 Agent 正在分析逻辑并生成文档...")
        
        # 3. 执行生成
        documentation = chain.invoke(code_content)
        
        return documentation

    def save_documentation(self, doc_content: str, output_path: str):
        """保存生成的文档"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        print(f"✅ 文档已保存至: {output_path}")

# --- 2. 运行示例 ---

if __name__ == "__main__":
    # 创建一个临时的测试 Python 文件用于演示
    sample_code_path = "sample_module.py"
    with open(sample_code_path, "w", encoding="utf-8") as f:
        f.write("""
import math

def calculate_compound_interest(principal, rate, time, frequency=1):
    """
    计算复利。
    参数:
        principal: 本金
        rate: 年利率 (小数，例如 0.05)
        time: 时间（年）
        frequency: 每年复利次数
    """
    if principal <= 0 or time < 0:
        raise ValueError("本金必须大于0，时间不能为负")
        
    amount = principal * (1 + rate / frequency) ** (frequency * time)
    return round(amount, 2)
""")

    # 初始化 Agent
    agent = CodeDocumentationAgent(model_name="gpt-4o") # 如果没有 gpt-4o 权限，可改为 "gpt-3.5-turbo"
    
    try:
        # 生成文档
        result_md = agent.generate_documentation(sample_code_path)
        
        # 打印预览
        print("\n--- 📄 生成结果预览 ---\n")
        print(result_md)
        
        # 保存文件
        agent.save_documentation(result_md, "sample_module_DOC.md")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("提示：请检查是否设置了 OPENAI_API_KEY 环境变量。")