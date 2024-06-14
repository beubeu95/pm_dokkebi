import os
import streamlit as st

from PyPDF2 import PdfReader
import os

from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI




folder_dir = './pdf'
pdfs = os.listdir(folder_dir)
raw_text = ""
for i in range(len(pdfs)):
    reader = PdfReader("./pdf/"+pdfs[i])
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text    



model = ChatOpenAI(model="gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4

qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)



st.title('Chat with GPT')

if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_response(prompt):

    response = qa_document_chain.run(
    input_document=raw_text,
    question=prompt)

    return response

def submit():
    user_input = st.session_state.user_input
    initial_content = '다음 내용에 대해 초등학생에게 이야기하듯이 대답해줘  '
    full_prompt = initial_content + user_input
    st.session_state.messages.append(("User", user_input))
    response = get_response(full_prompt)
    st.session_state.messages.append(("GPT", response))
    st.session_state.user_input = ""

st.text_input("You: ", key="user_input", on_change=submit)

for sender, message in st.session_state.messages:
    if sender == "User":
        st.write(f"**{sender}:** {message}")
    else:
        st.write(f"**{sender}:** {message}")