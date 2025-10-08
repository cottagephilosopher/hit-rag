"""
Prompt 模板：包含杂质标记、标签推理、切分指令的 LLM 提示词模板
"""

from typing import List, Dict

# 使用绝对导入
try:
    from config import JunkPatterns, TagConfig, ChunkConfig
except ImportError:
    from ..config import JunkPatterns, TagConfig, ChunkConfig


class PromptTemplates:
    """LLM 提示词模板集合"""

    @staticmethod
    def get_junk_marking_system_prompt() -> str:
        """
        获取杂质标记的系统提示词

        Returns:
            系统提示词
        """
        junk_features_str = "\n".join(
            f"- {jtype}: {desc}"
            for jtype, desc in JunkPatterns.JUNK_FEATURES.items()
        )

        return f"""你是一个专业的文档清洗助手。你的任务是识别并标记文档中的版式杂质。

## 任务说明
请仔细阅读提供的 Markdown 文档片段，识别其中的版式杂质（非核心内容），并使用 XML 标签进行标记。

## 杂质特征参考
{junk_features_str}

## 标记格式
使用以下格式标记杂质：
<JUNK type="杂质类型">杂质内容</JUNK>

## 重要约束
1. **格式保持**: 必须保留原文的所有 Markdown 格式、换行符、空格等。不得修改任何非杂质内容。
2. **完整标记**: 每个杂质必须完整地包含在一对 <JUNK> 标签中。
3. **类型准确**: type 属性必须准确描述杂质类型（页眉、页脚、警告等）。
4. **保守处理**: 如果不确定是否为杂质，请保留原文，不要标记。
5. **核心内容**: 绝不标记有实际信息价值的核心内容。

## 示例

输入:
```
公司机密 | 2024年1月
# 产品介绍
本产品是...
⚠️ 注意：此文档仅供内部使用
```

输出:
```
<JUNK type="页眉">公司机密 | 2024年1月</JUNK>
# 产品介绍
本产品是...
<JUNK type="警告信息">⚠️ 注意：此文档仅供内部使用</JUNK>
```

请严格遵守以上规则，处理用户提供的文档片段。
"""

    @staticmethod
    def get_junk_marking_user_prompt(chunk_text: str) -> str:
        """
        获取杂质标记的用户提示词

        Args:
            chunk_text: 要处理的文档片段

        Returns:
            用户提示词
        """
        return f"""请识别并标记以下文档片段中的版式杂质：

```markdown
{chunk_text}
```

请直接返回标记后��完整文档，不要添加任何说明或注释。
"""

    @staticmethod
    def get_tagging_system_prompt(existing_tags: List[str] = None) -> str:
        """
        获取标签推理的系统提示词

        Args:
            existing_tags: 系统现有的标签列表（从 /api/tags/all 获取）

        Returns:
            系统提示词
        """
        # 如果没有提供现有标签，使用配置中的默认标签
        if existing_tags is None or len(existing_tags) == 0:
            existing_tags = TagConfig.USER_DEFINED_TAGS

        tags_str = ", ".join(existing_tags)

        return f"""你是一个专业的文档分类和关键词提取助手。

## 任务说明
请分析提供的文档片段，完成两个任务：
1. 从系统现有标签列表中选择最匹配的标签作为用户标签
2. 从系统现有标签列表中选择 {TagConfig.CONTENT_TAG_COUNT} 个最相关的标签作为内容标签

## 系统现有标签
{tags_str}

## 输出格式
请以 JSON 格式返回结果：
{{
    "user_tag": "用户标签",
    "content_tags": ["标签1", "标签2", "标签3", "标签4", "标签5"]
}}

## 要求
1. **必须使用系统现有标签**：user_tag 和 content_tags 必须从系统现有标签列表中选择，不允许创建新标签
2. **如果标签数量不足**：如果系统现有标签少于 {TagConfig.CONTENT_TAG_COUNT} 个，则尽量选择，不足的部分可以留空或重复使用
3. content_tags 应包含 {TagConfig.CONTENT_TAG_COUNT} 个标签（从现有标签中选择）
4. 标签不要重复
5. 必须返回有效的 JSON 格式

请始终返回有效的 JSON 格式响应。
"""

    @staticmethod
    def get_chunking_system_prompt() -> str:
        """
        获取精细切分的系统提示词

        Returns:
            系统提示词
        """
        return f"""你是一个专业的文档切分助手，专门为 RAG（检索增强生成）系统准备文档块。

## 任务说明
请将提供的文档片段切分成多个语义连贯的小块，每块用于 RAG 检索。

## 切分原则（语义完整性优先）

**核心原则**: 语义完整性 > Token 数量限制

1. **Token 参考值**:
   - **最小值**: {ChunkConfig.FINAL_MIN_TOKENS} tokens（硬性要求，避免切片过小）
   - **目标值**: {ChunkConfig.FINAL_TARGET_TOKENS} tokens（理想大小，尽量接近）
   - **建议最大值**: {ChunkConfig.FINAL_MAX_TOKENS} tokens（可超出，优先保证语义完整）
   - **硬性上限**: {ChunkConfig.FINAL_HARD_LIMIT} tokens（安全阀，超出此值必须切分）

2. **语义完整性要求**（优先级从高到低）:
   - **句子完整性**: 绝不允许在句子中间切断
   - **段落完整性**: 尽量保持段落完整，不要将段落拆分
   - **小节完整性**: 标题与其内容必须在同一 chunk
   - **语义单元完整性**: 完整的概念、步骤、示例应保持在一起

3. **Markdown 结构完整性**:
   - 标题（#）必须与其紧随的内容在同一 chunk 中
   - 列表不能与其上方的标题或说明文字分离
   - 对比结构（如 "VS"、"对比" 等）的前后内容应保持在同一 chunk
   - 避免产生孤立的标题或过短的内容块

4. **特殊结构完整性**:
   - 表格、代码块、公式必须作为整体保留（使用 ATOMIC 标记）
   - 步骤序列（如 (1)(2)(3) 或 1.2.3.）必须保持完整
   - 连续的示例或案例应保持在同一 chunk

5. **切分决策逻辑**:
   - 如果语义单元完整且 token 数在最小值到建议最大值之间 → 保持完整，不切分
   - 如果语义单元完整但超过建议最大值 → 检查是否超过硬性上限：
     - 未超过硬性上限 → 标记为 ATOMIC-CONTENT，保持完整
     - 超过硬性上限 → 在语义边界处切分（标题、段落等）
   - 如果无法在语义边界切分（如超长表格、代码块）→ 标记为对应的 ATOMIC 类型

## 特殊标记（ATOMIC 块）

对于以下内容，**必须**使用 <ATOMIC-TYPE> 标签标记，作为一个完整的块输出（即使超过建议最大值）：

- **表格**: <ATOMIC-TABLE>表格内容</ATOMIC-TABLE>
- **代码块**: <ATOMIC-CODE>代码内容</ATOMIC-CODE>
- **超长链接列表**: <ATOMIC-LINK>链接列表</ATOMIC-LINK>
- **公式**: <ATOMIC-FORMULA>公式内容</ATOMIC-FORMULA>
- **超长步骤序列**: <ATOMIC-STEP>步骤序列内容</ATOMIC-STEP>
- **语义完整的大段落**: <ATOMIC-CONTENT>完整内容</ATOMIC-CONTENT>（超过建议最大值但未达硬性上限，且语义不可分割）

**ATOMIC 块使用规则**:
- **表格识别优先级最高**：表格必须独立成 chunk，不能与其他内容混合
- 即使文档片段整体 token 数合理，如果包含表格，也必须将表格分离出来
- **表格前的标题必须与表格合并**：
  - 如果表格前有标题（如"# 产品参数"、"1.2产品规格"），必须将标题包含在表格的 ATOMIC 块中
  - 标题与表格之间即使有空行，也应视为一个整体
  - 示例：`<ATOMIC-TABLE>1.2产品规格\n<table>...</table></ATOMIC-TABLE>`
- **编号步骤序列必须保持完整**：
  - 如果检测到编号步骤序列（1. 2. 3. ... 或 (1)(2)(3) ... 格式），必须保持完整，不能切开
  - 即使步骤序列较长，也应作为一个 ATOMIC-STEP chunk
  - **绝对禁止**：将连续的步骤序列（如 1-4 和 5-9）切分成多个 chunk
  - 示例：`<ATOMIC-STEP>1. 步骤一\n2. 步骤二\n3. 步骤三\n...</ATOMIC-STEP>`
  - 只有当步骤序列超过硬性上限（{ChunkConfig.FINAL_HARD_LIMIT} tokens）时才允许切分
- **禁止合并多个顶级章节**：不要将 "3.3 章节" 和 "3.4 章节" 合并到一个 chunk
  - 如果多个章节的总 token 数超过建议最大值，必须分开
  - 每个顶级章节（如 3.1, 3.2, 3.3）应该独立或与其子章节一起
- **非 ATOMIC chunk 的严格限制**：
  - 如果切分后的普通 chunk 超过建议最大值（{ChunkConfig.FINAL_MAX_TOKENS} tokens），必须继续细分
  - 只有无法再细分的语义完整内容才能标记为 ATOMIC-CONTENT
  - **绝对禁止**：创建超过建议最大值的非 ATOMIC chunk
- ATOMIC-CONTENT 用于语义完整但超过建议最大值的内容（如完整的产品介绍、完整的安装步骤等）
- 所有 ATOMIC 块的 token 数不得超过硬性上限（{ChunkConfig.FINAL_HARD_LIMIT} tokens）

## 输出格式
请以 JSON 格式返回切分结果：
{{
    "chunks": [
        {{
            "content": "第一块的完整内容",
            "is_atomic": false,
            "atomic_type": null
        }},
        {{
            "content": "<ATOMIC-TABLE>表格内容</ATOMIC-TABLE>",
            "is_atomic": true,
            "atomic_type": "table"
        }}
    ]
}}

## 重要提醒
1. 不要修改任何文本内容、格式、空格、换行符
2. **切分策略优先级**：
   - **优先在章节标题（# 开头）之前切分**，确保标题位于新块的开头
   - 标题应该与其下方内容保持在同一块中（最小语义单元）
   - 避免将标题留在块的末尾
   - 如果没有标题，再考虑在段落边界切分
3. ATOMIC 块可以超过建议最大值
4. 必须返回有效的 JSON 格式
5. 每个 chunk 的 content 必须是原文的连续片段

**良好切分示例**：
```
Chunk 1: [内容A]
Chunk 2: [# 3 产品安装 + 相关内容B]  ✅ 标题在开头
```

**不良切分示例**：
```
Chunk 1: [内容A + # 3 产品安装]  ❌ 标题在末尾
Chunk 2: [相关内容B]
```

请始终返回有效的 JSON 格式响应。
"""

    @staticmethod
    def get_chunking_user_prompt(chunk_text: str, estimated_tokens: int) -> str:
        """
        获取精细切分的用户提示词

        Args:
            chunk_text: 要切分的文档片段
            estimated_tokens: 估算的 Token 数量

        Returns:
            用户提示词
        """
        return f"""请将以下文档片段切分成适合 RAG 的小块：

文档片段（估算 {estimated_tokens} tokens）:
```markdown
{chunk_text}
```

目标: 每块 {ChunkConfig.FINAL_MIN_TOKENS}-{ChunkConfig.FINAL_MAX_TOKENS} tokens，保持句子和语义完整。

请返回 JSON 格式的切分结果。
"""

    @staticmethod
    def get_combined_clean_and_tag_system_prompt(existing_tags: List[str] = None) -> str:
        """
        获取组合的清洗和标签提示词（阶段2优化版本）

        Args:
            existing_tags: 系统现有的标签列表（从 /api/tags/all 获取）

        Returns:
            系统提示词
        """
        junk_features_str = "\n".join(
            f"- {jtype}: {desc}"
            for jtype, desc in JunkPatterns.JUNK_FEATURES.items()
        )

        # 如果没有提供现有标签，使用配置中的默认标签
        if existing_tags is None or len(existing_tags) == 0:
            existing_tags = TagConfig.USER_DEFINED_TAGS

        tags_str = ", ".join(existing_tags)

        return f"""你是一个专业的文档处理助手。你需要完成两个任务：

## 任务1: 识别并标记版式杂质
识别文档中的版式杂质（非核心内容），使用 <JUNK type="类型">内容</JUNK> 标记。

### 杂质特征参考
{junk_features_str}

## 任务2: 提取文档标签
分析文档内容，提取：
1. 用户标签: 从系统现有标签中选择最匹配的
2. 内容标签: 从系统现有标签中选择 {TagConfig.CONTENT_TAG_COUNT} 个最相关的

### 系统现有标签
{tags_str}

## 输出格式
请以 JSON 格式返回：
{{
    "marked_text": "标记了杂质的完整文档...",
    "user_tag": "用户标签",
    "content_tags": ["标签1", "标签2", "标签3", "标签4", "标签5"]
}}

## 重要约束
1. marked_text 必须保留原文的所有格式（换行���空格、Markdown 语法）
2. 只标记明确的杂质，不确定的保留
3. **必须使用系统现有标签**：user_tag 和 content_tags 必须从系统现有标签列表中选择，不允许创建新标签
4. **如果标签数量不足**：如果系统现有标签少于 {TagConfig.CONTENT_TAG_COUNT} 个，则尽量选择，不足的部分可以留空或重复使用
5. 标签不要重复
6. 必须返回有效的 JSON 格式

请始终返回有效的 JSON 格式响应。
"""

    @staticmethod
    def get_combined_clean_and_tag_user_prompt(chunk_text: str) -> str:
        """
        获取组合的清洗和标签用户提示词

        Args:
            chunk_text: 要处理的文档片段

        Returns:
            用户提示词
        """
        return f"""请处理以下文档片段：

```markdown
{chunk_text}
```

请返回包含标记后文本和标签的 JSON 结果。
"""


# 便捷函数
def get_junk_marking_prompts(chunk_text: str) -> tuple:
    """
    获取杂质标记的完整提示词

    Args:
        chunk_text: 文档片段

    Returns:
        (system_prompt, user_prompt) 元组
    """
    return (
        PromptTemplates.get_junk_marking_system_prompt(),
        PromptTemplates.get_junk_marking_user_prompt(chunk_text)
    )


def get_tagging_prompts(chunk_text: str, existing_tags: List[str] = None) -> tuple:
    """
    获取标签推理的完整提示词

    Args:
        chunk_text: 文档片段
        existing_tags: 系统现有的标签列表

    Returns:
        (system_prompt, user_prompt) 元组
    """
    return (
        PromptTemplates.get_tagging_system_prompt(existing_tags),
        PromptTemplates.get_tagging_user_prompt(chunk_text)
    )


def get_chunking_prompts(chunk_text: str, estimated_tokens: int) -> tuple:
    """
    获取精细切分的完整提示词

    Args:
        chunk_text: 文档片段
        estimated_tokens: 估算的 Token 数

    Returns:
        (system_prompt, user_prompt) 元组
    """
    return (
        PromptTemplates.get_chunking_system_prompt(),
        PromptTemplates.get_chunking_user_prompt(chunk_text, estimated_tokens)
    )


def get_combined_clean_and_tag_prompts(chunk_text: str, existing_tags: List[str] = None) -> tuple:
    """
    获取组合清洗和标签的完整提示词

    Args:
        chunk_text: 文档片段
        existing_tags: 系统现有的标签列表

    Returns:
        (system_prompt, user_prompt) 元组
    """
    return (
        PromptTemplates.get_combined_clean_and_tag_system_prompt(existing_tags),
        PromptTemplates.get_combined_clean_and_tag_user_prompt(chunk_text)
    )


if __name__ == "__main__":
    # 测试代码
    test_text = """公司机密 | 2024年1月

# 产品介绍
本产品是一个 RAG 系统。

⚠️ 注意：此文档仅供内部使用"""

    # 测试杂质标记提示词
    print("=== 杂质标记提示词 ===")
    system, user = get_junk_marking_prompts(test_text)
    print(f"System: {system[:200]}...")
    print(f"User: {user[:200]}...")

    # 测试标签推理提示词
    print("\n=== 标签推理提示词 ===")
    system, user = get_tagging_prompts(test_text)
    print(f"System: {system[:200]}...")
    print(f"User: {user[:200]}...")

    # 测试切分提示词
    print("\n=== 切分提示词 ===")
    system, user = get_chunking_prompts(test_text, 500)
    print(f"System: {system[:200]}...")
    print(f"User: {user[:200]}...")
