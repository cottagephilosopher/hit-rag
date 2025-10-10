"""
DSPy Signatures 定义
定义各个 RAG 处理步骤的输入输出接口
"""

import dspy


class IntentClassification(dspy.Signature):
    """识别用户查询意图"""

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="用户当前查询")

    intent = dspy.OutputField(desc="意图类型：question（问答）| clarification（澄清）| chitchat（闲聊）")
    confidence = dspy.OutputField(desc="置信度分数 0.0-0.8。要求保守评估：明确清晰的意图才给>0.75，有模糊性的给0.5-0.7，不确定的给<0.5")
    reasoning = dspy.OutputField(desc="判断理由")


class QueryRewrite(dspy.Signature):
    """优化用户查询以提高检索效果"""

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="原始用户查询")

    rewritten_query = dspy.OutputField(desc="优化后的查询语句，更适合向量检索")
    key_entities = dspy.OutputField(desc="提取的关键实体和概念，JSON格式列表")
    search_strategy = dspy.OutputField(desc="建议的搜索策略：exact_match | semantic | hybrid")


class ConfidenceEvaluation(dspy.Signature):
    """评估检索结果是否足以回答问题"""

    user_query = dspy.InputField(desc="用户查询")
    retrieved_chunks = dspy.InputField(desc="检索到的文档片段（JSON格式）")

    is_sufficient = dspy.OutputField(desc="信息是否充分：yes | no")
    confidence = dspy.OutputField(desc="置信度 0.0-1.0")
    missing_info = dspy.OutputField(desc="缺失的关键信息（如果不充分），JSON格式列表")
    reasoning = dspy.OutputField(desc="评估理由")


class ClarificationGeneration(dspy.Signature):
    """生成澄清问题"""

    user_query = dspy.InputField(desc="用户的原始查询")
    missing_info = dspy.InputField(desc="识别出的缺失信息")
    conversation_history = dspy.InputField(desc="对话历史")

    clarification_question = dspy.OutputField(desc="生成的澄清问题，要求明确、友好")
    suggested_options = dspy.OutputField(desc="可选的澄清选项，JSON格式列表")


class ResponseGeneration(dspy.Signature):
    """基于检索结果生成回复"""

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="用户查询")
    retrieved_chunks = dspy.InputField(desc="检索到的文档片段（JSON格式，包含内容和来源）")

    response = dspy.OutputField(desc="生成的回复，必须使用 Markdown 格式。对于文档中的图片链接，使用 ![图片描述](图片URL) 语法直接嵌入图片（不要用普通链接）。使用标准Markdown：列表、标题、加粗、代码块等。要求准确、简洁、有帮助")
    source_ids = dspy.OutputField(desc="引用的文档片段ID列表，JSON格式")
    confidence = dspy.OutputField(desc="回复置信度 0.0-1.0")
