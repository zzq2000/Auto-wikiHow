import streamlit as st
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import openai
from llama_index_client import SentenceSplitter

from llama_index.core import Settings, PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llm import get_llm
from index import get_index
from evaluate import evaluating
from embedding import get_embedding
from qa_loader import get_qa_dataset
from config import cfg
from retriever import *

# 假设所有必要的模块和函数已经被导入

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Streamlit UI配置
st.title("RAG-benchmark")

# 用户可以修改的配置参数
llm_choice = st.selectbox("choose LLM:", ["chatgpt-3.5", "llama", "chatglm", "qwen", "baichuan", "falcon", "mpt", "yi"])
embeddings_choice = st.selectbox("choose embedding:", ["BAAI/bge-large-en-v1.5", "其他选项"])
split_type_choice = st.selectbox("choose split:", ["sentence", "paragraph"])
chunk_size_choice = st.number_input("Chunk size:", min_value=512, max_value=2048, value=1024)
dataset_choice = st.selectbox("dataset:", ["hotpot_qa", "drop"])
# source_dir = st.text_input("源目录:", "../wiki")
persist_dir = st.text_input("index storage path:", "../storage")

if st.button('初始化系统'):
    cfg = cfg()
    cfg.llm = llm_choice
    cfg.embeddings = embeddings_choice
    cfg.split_type = split_type_choice
    cfg.chunk_size = chunk_size_choice
    cfg.dataset = dataset_choice
    cfg.persist_dir = persist_dir

    qa_dataset = get_qa_dataset(cfg.dataset)
    llm = get_llm(cfg.llm)
    embeddings = get_embedding(cfg.embeddings)

    Settings.chunk_size = cfg.chunk_size
    Settings.llm = llm
    Settings.embed_model = embeddings

    cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + cfg.embeddings + '-' + cfg.split_type + '-' + str(cfg.chunk_size)
    index = get_index(qa_dataset, cfg.persist_dir, split_type=cfg.split_type, chunk_size=cfg.chunk_size)

    query_engine = RetrieverQueryEngine(
        retriever=vector_retriever(index),
        response_synthesizer=response_synthesizer(0),
        node_postprocessors=[LongContextReorder()]
    )

    text_qa_template_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = (
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    )
    refine_template = PromptTemplate(refine_template_str)

    # Setup index query engine using LLM
    # query_engine = index.as_query_engine(response_mode="compact")

    query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template,
                                 "response_synthesizer:refine_template": refine_template})
    query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
    query_engine = RetrieverQueryEngine.from_args(query_engine)

    st.success("系统初始化完成！")

# 添加一个文本输入框让用户输入问题
question = st.text_input("请输入您的问题:", "")

if st.button('查询答案'):
    if question and 'query_engine' in locals():
        # 假设query_engine已经根据上面的配置正确初始化
        response = query_engine.query(question)
        st.write("回答：", response)
    else:
        st.write("请先初始化系统并输入一个问题。")
