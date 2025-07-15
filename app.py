import streamlit as st
import logging
from openai import OpenAI

# 项目模块导入
# ===================== 修改点在这里 =====================
# 我们将导入整个 config 模块，而不是只导入里面的变量
import config 
# =======================================================
from vectordb_manager import VectorDBManager
from cache import QueryCache
from workflow import MyopiaControlWorkflow

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="Evidence-Based AI for Myopia Control",
    page_icon="🔬",
    layout="wide"
)

# --- UI 文本多语言支持 ---
UI_TEXTS = {
    "en": {
        "title": "🔬 Evidence-Based Conversational AI for Low-Concentration Atropine in Myopia Control",
        "subtitle": "An AI assistant providing evidence-based answers from medical literature.",
        "input_label": "Ask a question about low-concentration atropine for myopia:",
        "input_placeholder": "e.g., What is the efficacy of 0.01% atropine compared to 0.05%?",
        "spinner_text": "Analyzing medical literature... 💡",
        "answer_header": "💬 Answer",
        "context_header": "View Evidence Source",
        "context_subheader": "The following excerpts from the literature were used to generate the answer:",
        "error_init": "System initialization failed. Please check logs and configuration.",
        "error_runtime": "An unexpected error occurred during processing.",
        "footer": "❤️ **Disclaimer:** This AI assistant provides information from medical literature for research and reference purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns."
    },
    "zh": {
        "title": "🔬 低浓度阿托品近视控制的循证对话式AI",
        "subtitle": "一个基于医学文献，提供循证答案的AI助手。",
        "input_label": "请输入关于低浓度阿托品在近视控制方面的问题：",
        "input_placeholder": "例如：0.01%阿托品与0.05%阿托品相比，疗效如何？",
        "spinner_text": "正在分析医学文献... 💡",
        "answer_header": "💬 回答",
        "context_header": "查看证据来源",
        "context_subheader": "以下是用于生成答案的文献原文片段：",
        "error_init": "系统初始化失败，请检查日志和配置。",
        "error_runtime": "处理过程中发生意外错误。",
        "footer": "❤️ **重要声明:** 本AI助手提供的信息来源于医学文献，仅供科研与参考。它不能替代专业的医疗建议、诊断或治疗。如有任何医疗问题，请务必咨询合格的医疗专业人员。"
    }
}

# --- 组件初始化 (使用Streamlit缓存) ---
@st.cache_resource
def initialize_components():
    """初始化问答系统所需的所有组件。"""
    logger.info("正在初始化组件...")
    try:
        # 1. LLM 客户端
        if not config.DASHSCOPE_API_KEY:
            st.error("错误：请在Streamlit Cloud的Secrets中设置 DASHSCOPE_API_KEY。")
            return None
        llm_client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

        # 2. 向量数据库管理器
        vdb_manager = VectorDBManager()
        vdb_manager.load_db()

        # 3. 查询缓存 (可选)
        cache = QueryCache(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
        if not cache.redis_client:
            st.warning("无法连接到Redis，缓存功能将被禁用。")
            cache = None

        # 4. 初始化工作流
        workflow = MyopiaControlWorkflow(
            llm_client=llm_client,
            retriever=vdb_manager,
            cache=cache
        )
        logger.info("所有组件初始化成功。")
        return workflow

    except Exception as e:
        st.error(f"组件初始化期间发生严重错误: {e}")
        logger.exception("Streamlit应用组件初始化期间发生严重错误。")
        return None

# --- 主应用逻辑 ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def set_language():
    st.session_state['lang'] = st.session_state['lang_selector']

st.sidebar.radio(
    label="Language / 语言",
    options=['en', 'zh'],
    format_func=lambda x: "English" if x == 'en' else "中文",
    key='lang_selector',
    on_change=set_language
)
texts = UI_TEXTS[st.session_state.lang]

st.title(texts["title"])
st.markdown(f"##### {texts['subtitle']}")
st.markdown("---")

workflow_instance = initialize_components()

if workflow_instance:
    user_question = st.text_input(
        texts["input_label"],
        placeholder=texts["input_placeholder"]
    )

    if user_question:
        with st.spinner(texts["spinner_text"]):
            try:
                result = workflow_instance.run(user_question, language=st.session_state.lang)

                st.markdown("---")
                st.subheader(texts["answer_header"])
                st.markdown(result.get("answer", "No answer generated."))

                with st.expander(texts["context_header"]):
                    st.markdown(f"**{texts['context_subheader']}**")
                    st.text(result.get("formatted_context", "No context available."))

                if result.get("error"):
                    st.error(f"An error occurred: {result.get('error')}")

            except Exception as e:
                st.error(f"{texts['error_runtime']}: {str(e)}")
                logger.exception(f"处理问题时出错 '{user_question}':")
else:
    st.error(UI_TEXTS["en"]["error_init"])

st.markdown("---")
st.markdown(texts["footer"])