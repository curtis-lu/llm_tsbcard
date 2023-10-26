import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key')

## load vector store
@st.cache_resource
def load_vector_store():

    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()

    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

@st.cache_data
def load_templates():

    templates = []
    for filename in ['anatomy_template', 'compression_template', 'chat_template']:
        with open(f'./templates/{filename}.txt', 'r') as file:
            templates.append(file.read())

    return templates



def generate_response(input_text, anatomy_chain, retriever, compression_chain, chat_chain):

    the_question = input_text
    the_anatomy = anatomy_chain.invoke(the_question)
    the_context = retriever.invoke(the_question)
    the_new_context = compression_chain.invoke({'customer_question':the_question, 'anatomy':the_anatomy, 'context':the_context})
    the_answer = chat_chain.invoke({'customer_question':the_question, 'new_context':the_new_context})
    st.info(the_answer['text'])

with st.form('my_form'):
    text = st.text_area('Enter text:', '國外網站消費要用哪張卡回饋最高？')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):

        vectordb_load = load_vector_store()
        retriever=vectordb_load.as_retriever()

        templates = load_templates()

        anatomy_template, compression_template, chat_template = templates

        anatomy_prompt = PromptTemplate.from_template(anatomy_template)
        llm = OpenAI(openai_api_key=openai_api_key)
        anatomy_chain = LLMChain(llm=llm, prompt=anatomy_prompt, output_key="anatomy")

        compression_prompt = PromptTemplate.from_template(compression_template)
        compression_llm = OpenAI(max_tokens=-1, openai_api_key=openai_api_key)
        compression_chain = LLMChain(llm=compression_llm, prompt=compression_prompt, output_key="new_context")

        chat_prompt = ChatPromptTemplate.from_template(chat_template)
        chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0, openai_api_key=openai_api_key)
        chat_chain = LLMChain(llm=chat_llm, prompt=chat_prompt)

        generate_response(text, anatomy_chain, retriever, compression_chain, chat_chain)