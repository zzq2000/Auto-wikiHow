from llama_index.llms.openai import OpenAI

from llms.huggingface_model import get_huggingfacellm
from llms import chatglm4
from embs import chatglmemb
from config import API_KEY, API_BASE


llm_dict = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "THUDM/chatglm3-6b",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "qwen14_int8": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
    "qwen7_int8": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
    "qwen1.8": "Qwen/Qwen1.5-1.8B-Chat",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-chat",
    "yi": "01-ai/Yi-6B-Chat",
}


def get_openai(api_base,api_key):
    return OpenAI(api_key=api_key,api_base=api_base)


def get_llm(name):
    if name in llm_dict.keys():
        return get_huggingfacellm(llm_dict[name])
    elif name == 'chatgpt-3.5':
        return get_openai(API_BASE, API_KEY)
    # elif name == 'wxl':
    #     return OpenAI(api_key="sk-FDMbb0bnifWopAgv4501226dEa4c45E5Ac81841123Eb80B0", api_base="https://uiuiapi.com/v1",
    #                   model='gpt-3.5-turbo-0613')
    else:
        raise ValueError(f"no model name: {name}.")


