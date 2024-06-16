import random
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#### api 키 설정 

####

#### gpt 모델 설정
model = ChatOpenAI(model="gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4

qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
#### 


def get_response(prompt, raw_text):
    response = qa_document_chain.run(
        input_document=raw_text,
        question=prompt
    )
    return response

app = Flask(__name__)


# Initialize session state
session_state = {
    'messages': []
}


@app.route('/')
def index():
    
    return render_template('page1.html')

@app.route('/two')
def index2():

    return render_template('chatbot.html')


# 데이터 로드 (여기에 데이터를 불러오는 코드 추가)
df = pd.read_csv('C:/beubeu95/cj/chatbot/pm_dokkebi/sample.csv', header=1)
data = df.loc[:,['지문 ', '지문 분야', '지문 주제']]
data['isSelected'] = 0
data.head()

@app.route('/random_question', methods=['GET'])
def get_article():

    # 데이터 필터링 및 샘플 추출
    target = data.loc[(data['isSelected'] != 1), :].sample(n=1)
    targetIndex = target.index
    data.loc[targetIndex, 'isSelected'] = 1

    # 지문 및 지문주제 가져오기
    targetArticle = data.loc[targetIndex[0], '지문 ']
    targetTitle = data.loc[targetIndex[0], '지문 주제']

    # JSON 응답 생성
    response = {
        "question": targetArticle,
        "answer": targetTitle
    }

    print(response)

    return jsonify(response)

@app.route('/chat', methods=['POST'])
def calculate_similarity():
    # 요청에서 데이터 가져오기
    user_input = request.json.get('user_input')
    target_title = request.json.get('target_title')

    print('입력내용: ' + user_input)
    print('지문 주제: ' + target_title)

    # CountVectorizer 객체 생성
    vectorizer = CountVectorizer()

    # 문장들을 벡터화
    vectors = vectorizer.fit_transform([user_input, target_title])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(vectors[0], vectors[1])

    # 유사도에 따라 적절한 응답 생성
    if cosine_sim[0][0] > 0.8:
        # 정답인 경우
        feedback = f"\n고생했어. 핵심주제의 모범답안은 {target_title}이니 참고해줘!"
    else :
        # 오답인 경우
        feedback = f"\n왜 그렇게 생각해?"

    # 결과 JSON 응답 생성
    result = {
        "user_input": user_input,
        "target_title": target_title,
        "cosine_similarity": cosine_sim[0][0],
        "feedback": feedback
    }
    print('피드백[gpt]: ' + result['feedback'])

    return jsonify({'feedback': result['feedback'], 'cosine' : result['cosine_similarity'] })

@app.route('/feedback', methods=['POST'])
def feedback():
    # 요청에서 데이터 가져오기
    evidence = request.json.get('user_input')
    target_article = request.json.get('target_article')
    target_title = request.json.get('target_title')
    user_input = request.json.get('user_title')

    print('입력내용[근거]: ' + evidence)
    print('지문: ' + target_article)
    print('지문 주제: ' + target_title)
    print('학생이 생각한 주제: ' + user_input)

    # CountVectorizer 객체 생성
    vectorizer = CountVectorizer()

    # 문장들을 벡터화
    vectors = vectorizer.fit_transform([evidence, target_title])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(vectors[0], vectors[1])

    if cosine_sim[0][0] > 0.8:
        # 정답인 경우
        response = f"\n맞아. 핵심주제의 모범답안은 {target_title}이니 참고해줘!"
    else : 
        # 오답인 경우, 제시한 근거로 그것의 논리성 해설
        initial_content = (
            f'너는 선생님이야. 학생이 지문을 읽고 핵심주제와 논리적 근거를 다음과 같이 생각했어. '
            f'정답과 비교하여 이것이 오답인 이유를 선생님이 학생에게 알려주듯이 친절하게 그리고 반말로 설명해줘.\n'
            f'지문: {target_article}\n'
            f'정답: {target_title}\n'
            f'학생이 생각한 핵심주제: {user_input}\n'
            f'논리적 근거: {evidence}'
        )
        full_prompt = initial_content
        session_state['messages'].append(("User", user_input))
        response = get_response(full_prompt, target_article)
        session_state['messages'].append(("GPT", response))
        
    print(response)

    # 결과 JSON 응답 생성
    return jsonify({'feedback': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)