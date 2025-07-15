import logging
from openai import OpenAI
from config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

# --- Chain-of-Thought (思维链) 提示模板 ---

# 英文提示
PROMPT_TEMPLATE_EN = """
You are a professional and meticulous AI medical assistant. Your expertise is in analyzing evidence regarding the use of low-concentration atropine for myopia control in children.

Your task is to answer the user's question based *ONLY* on the provided "Evidence from Medical Literature".

Follow these steps strictly:
1.  **Synthesize Evidence**: First, carefully review the provided literature excerpts. Summarize the key findings, data, and conclusions that are directly relevant to the user's question.
2.  **Formulate Answer**: Based *only* on your synthesis from step 1, construct a clear, evidence-based, and objective answer.
3.  **Cite Sources**: At the end of your answer, list the source documents (e.g., [Source: doc1.pdf, doc2.pdf]) you used to formulate the answer.

If the provided evidence does not contain information to answer the question, you MUST state: "Based on the provided literature, I cannot find sufficient information to answer this question." Do not use any external knowledge.

---
**Evidence from Medical Literature:**
{context}
---
**User's Question:**
{question}
---
**Your Response (following the 3-step process):**
"""

# 中文提示
PROMPT_TEMPLATE_ZH = """
你是一名专业、严谨的AI医疗助理，专门负责分析关于使用低浓度阿托品控制儿童近视的循证医学证据。

你的任务是*仅*根据下面提供的“医学文献证据”来回答用户的问题。

请严格遵循以下步骤：
1.  **综合证据**：首先，仔细阅读提供的文献摘要。总结与用户问题直接相关的关键发现、数据和结论。
2.  **构建答案**：*仅*基于你在第一步中的综合分析，构建一个清晰、客观、有理有据的答案。
3.  **引用来源**：在答案的末尾，列出你用于形成答案的文献来源 (例如: [来源: doc1.pdf, doc2.pdf])。

如果所提供的证据不足以回答问题，你*必须*明确说明：“根据现有文献，我无法找到足够的信息来回答此问题。” 不要使用任何外部知识。

---
**医学文献证据:**
{context}
---
**用户问题:**
{question}
---
**你的回答 (请遵循三步流程):**
"""

PROMPT_TEMPLATES = {
    "en": PROMPT_TEMPLATE_EN,
    "zh": PROMPT_TEMPLATE_ZH
}

class QAEngine:
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def _format_context(self, retrieved_docs: list) -> str:
        """
        将检索到的文档块格式化为可读的字符串，并包含来源信息。
        """
        if not retrieved_docs:
            return "No information retrieved from the literature."
        
        context_parts = []
        sources = set() # 用集合来收集不重复的来源
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
            content = doc.page_content
            context_parts.append(f"Source: {source}\nContent: {content}\n---")
        
        # 将来源信息也加入上下文，方便LLM引用
        source_info = f"Evidence is drawn from the following documents: {', '.join(sorted(list(sources)))}."
        return f"{source_info}\n\n" + "\n".join(context_parts)

    def generate_answer(self, question: str, retrieved_docs: list, language: str = "en") -> dict:
        """
        基于检索到的上下文，使用LLM生成答案。
        """
        formatted_context = self._format_context(retrieved_docs)
        
        prompt_template = PROMPT_TEMPLATES.get(language, PROMPT_TEMPLATE_EN)
        prompt = prompt_template.format(context=formatted_context, question=question)

        answer = "Could not generate an answer."
        error_message = None

        try:
            logger.info("正在使用LLM生成答案...")
            completion = self.llm.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = completion.choices[0].message.content
            logger.info("答案生成成功。")
        except Exception as e:
            logger.error(f"答案生成失败: {str(e)}")
            error_message = str(e)

        return {
            "answer": answer,
            "formatted_context": formatted_context,
            "error": error_message
        }