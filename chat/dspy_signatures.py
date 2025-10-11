"""
DSPy Signatures 定义
定义各个 RAG 处理步骤的输入输出接口
"""

import dspy


class IntentClassification(dspy.Signature):
    """
    识别用户查询意图

    业务上下文：
    - 这是一个企业产品知识库助手，服务企业内部成员
    - 知识库包含：产品手册、安装文档、说明书、销售资料等
    - 核心业务：帮助用户快速找到产品相关信息
    """

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="用户当前查询")

    intent = dspy.OutputField(desc="意图类型：question（产品相关问答）| chitchat（闲聊/无关话题）")
    confidence = dspy.OutputField(desc="置信度分数 0.0-0.8。要求保守评估：明确清晰的意图才给>0.7，有模糊性的给0.45-0.65，不确定的给<0.45")
    business_relevance = dspy.OutputField(desc="业务相关性：high（明确的产品问题）| medium（可能相关）| low（偏离主题）")
    reasoning = dspy.OutputField(desc="判断理由")


class QueryRewrite(dspy.Signature):
    """优化用户查询以提高检索效果"""

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="原始用户查询")

    rewritten_query = dspy.OutputField(desc="优化后的查询语句，更适合向量检索")
    key_entities = dspy.OutputField(desc="提取的关键实体和概念，JSON格式列表")
    search_strategy = dspy.OutputField(desc="建议的搜索策略：exact_match | semantic | hybrid")


class ConfidenceEvaluation(dspy.Signature):
    """
    评估检索结果的充分性和二义性

    业务上下文：
    - 产品知识库可能包含多个产品、多个版本的文档
    - 需要识别：是否涉及多个产品？是否需要明确版本/场景？
    - 目标：检测二义性，为"先答后问"策略提供依据
    """

    user_query = dspy.InputField(desc="用户查询")
    retrieved_chunks = dspy.InputField(desc="检索到的文档片段（JSON格式），注意来源文档名称可能包含产品/版本信息")

    is_sufficient = dspy.OutputField(desc="信息是否充分回答基础问题：yes | no")
    has_ambiguity = dspy.OutputField(desc="是否存在二义性：yes（涉及多个产品/版本/场景）| no（明确单一场景）")
    confidence = dspy.OutputField(desc="置信度 0.0-1.0")
    ambiguity_type = dspy.OutputField(desc="如果有二义性，类型是：multi_product（多产品）| multi_version（多版本）| multi_scenario（多场景）| none")
    clarification_hint = dspy.OutputField(desc="如果有二义性，建议追问的方向（简短提示，非完整问题）")
    reasoning = dspy.OutputField(desc="评估理由")


class ClarificationGeneration(dspy.Signature):
    """生成澄清问题"""

    user_query = dspy.InputField(desc="用户的原始查询")
    missing_info = dspy.InputField(desc="识别出的缺失信息")
    conversation_history = dspy.InputField(desc="对话历史")

    clarification_question = dspy.OutputField(desc="生成的澄清问题，要求明确、友好")
    suggested_options = dspy.OutputField(desc="可选的澄清选项，JSON格式列表")


class ResponseGeneration(dspy.Signature):
    """
    基于检索结果生成回复

    业务上下文：
    - 企业产品知识库助手，面向内部成员
    - 回答要专业、准确，引用具体文档位置
    - 如果检索到多个产品/版本的内容，需在回答中说明差异
    """

    conversation_history = dspy.InputField(desc="对话历史上下文")
    user_query = dspy.InputField(desc="用户查询")
    retrieved_chunks = dspy.InputField(desc="检索到的文档片段（JSON格式，包含内容和来源）")
    clarification_hint = dspy.InputField(desc="可选的澄清提示（如果需要在回答后引导用户）", default="")
    intent_note = dspy.InputField(desc="意图识别备注（如：闲聊置信度不足，基于知识库尝试回答）", default="")

    response = dspy.OutputField(desc="生成的回复，必须使用 Markdown 格式。对于文档中的图片链接，使用 ![图片描述](图片URL) 语法直接嵌入图片（不要用普通链接）。使用标准Markdown：列表、标题、加粗、代码块等。要求准确、简洁、有帮助。如果有 clarification_hint，在回答末尾自然地引导用户提供更多信息（非生硬的反问）")
    source_ids = dspy.OutputField(desc="引用的文档片段ID列表，JSON格式")
    confidence = dspy.OutputField(desc="回复置信度 0.0-1.0")
