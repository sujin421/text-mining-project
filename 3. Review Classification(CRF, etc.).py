#!/usr/bin/env python
# coding: utf-8

# # 2021-2 텍스트마이닝 기말 프로젝트

# # import packages

# In[332]:


import requests
import time
import re
import pandas as pd
import pprint
import pycrfsuite
import eli5
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import site; site.getsitepackages()
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn_crfsuite import CRF 
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite .metrics import flat_classification_report 
from nltk.probability import FreqDist
from pycrfsuite_spacing import TemplateGenerator
from pycrfsuite_spacing import CharacterFeatureTransformer
from pycrfsuite_spacing import sent_to_xy
from konlpy.tag import Kkma as kkm


# # Web Scraping

# In[333]:


# 참고 사이트 https://yeo0.github.io/data/2018/09/24/5.-%EB%A1%9C%EA%B7%B8%EC%9D%B8%EC%9D%B4-%ED%95%84%EC%9A%94%ED%95%9C-%EC%82%AC%EC%9D%B4%ED%8A%B8%EC%97%90%EC%84%9C%EC%9D%98-%ED%81%AC%EB%A1%A4%EB%A7%81/
# 참고 사이트 2 https://hashcode.co.kr/questions/9084/%EC%9E%A1%ED%94%8C%EB%9E%98%EB%8B%9B-%EB%A1%9C%EA%B7%B8%EC%9D%B8-%ED%9B%84-%EC%8A%A4%ED%81%AC%EB%A0%88%EC%9D%B4%ED%95%91-%EB%AC%B8%EC%9D%98%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4-%E3%85%9C%E3%85%9C

# 로그인 할 url
url = "https://www.jobplanet.co.kr/users/sign_in?_nav=gb"
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'User-Agent':user_agent}
login_data = {'user':{'email':'hsmy31@hanyang.ac.kr', 'password':'rhkwpgksmswnd!', 'remember_me':'true'}}
session = requests.session()

# 로그인 실행
login_response = session.post(url, json = login_data, headers = headers)

reviews = []


# In[334]:


#전체 리뷰 추출

def all_reviews(code):
    url = "https://www.jobplanet.co.kr/reviews?&industry_id=" + str(code)
    response =  session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    num = soup.find('span', class_='num') # 리뷰 개수 확인
    num = int(num.get_text().strip())
    import math, random
    pages = math.ceil(num / 10)
    reviews = {}
    for i in range(pages+1):
        time.sleep(1)
        url = "https://www.jobplanet.co.kr/reviews?&industry_id=" + str(code) + "&page="+ str(i)
        response =  session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        label = soup.find_all('h2', class_="us_label") # 한 줄 리뷰 추출
        # 추출한 리뷰에서 태그를 제외한 텍스트만 추출하여 labels 리스트에 저장
        labels = [label.get_text().strip() for label in label[:]]
        star = soup.find_all('div', class_="star_score") # 별점이 포함된 div 추출
        # 추출한 div에서 별점을 나타내는 width 속성의 숫자로 된 부분을 stars 리스트에 저장
        stars = re.findall('[0-9]+[.]+[0-9]', str(star))
        # reviews 딕셔너리에 '리뷰: 별점' 형식으로 추가
        for j in range(len(labels)):
            reviews[labels[j][5:-1]] = stars[j].replace('.0','')
    return reviews


# In[335]:


#전체 리뷰 및 별점 딕셔너리화
codes = ['1001', '900', '704', '1004']
reviews={}

for i in range(0,len(codes)):
    reviews.update(dict(all_reviews(codes[i])))


# In[410]:


len(reviews)


# In[411]:


# 리스트 형식으로 text와 score 각각 저장
text_all=list(reviews.keys())
score_all=list(reviews.values())


# In[412]:


# 점수별 리뷰 수 분포
score_cnt = {} # 
for sc in score_all:
    sc_int = int(sc)
    if sc_int in score_cnt:
        score_cnt[sc_int] += 1
    else:
        score_cnt[sc_int] = 1


# In[413]:


score_sort = sorted(score_cnt.items(), key=lambda item: item[0], reverse=True)


# In[414]:


x=[]
y=[]
for i in range(len(score_sort)):
    x.append(score_sort[i][0])
    y.append(score_sort[i][1])
plt.plot(x, y)
plt.xlabel("score")
plt.ylabel("review freq")
plt.show()


# In[415]:


# 전체 리뷰의 평균값 구하기
sum=0
for i in range(len(score_all)):
    sum+=int(score_all[i])
avr_all=sum/len(score_all)
print("score 평균값: ", round(avr_all,3))


# # 긍정어/부정어사전 만들기

# In[416]:


#공통/ 업종별 불용어(자주 나오지만 문서 전체에 관련되어 있어 성능 체크에 의미가 없는 단어들) 선언
stopwords=['있음','기업', '업무', '회사', '직장','분위기','사람','직원','근무']


# In[417]:


#긍/부정 리뷰별 빈출단어 추출 및 사전화
#검색 성능을 높이기 위해 문장을 제일 잘게 쪼개는 kkma 사용
kkm=kkm()
text_kkm=[]
#형태소 분석 및 문장별 점수 매칭
for i in range(len(text_all)):
    text_kkm.append(kkm.nouns(text_all[i]))
    text_kkm[i].append(score_all[i])


# In[418]:


pos_kkm=[]
morphs_kkm=[]
dic_stopwords=stopwords
#연관이 낮아 보이는 형태소와 1글자 음절, 불용어를 제외한 체언, 용언, 관형사, 부사만 뽑아내기
for i in range(len(text_all)):
    pos_kkm.append(kkm.pos(text_all[i]))
dic_kkm=[]
for i in range(len(pos_kkm)):
    dic_kkm.append([])
    for j in range(len(pos_kkm[i])):
        if pos_kkm[i][j][0] not in dic_stopwords:
            if pos_kkm[i][j][1].startswith('N') or pos_kkm[i][j][1].startswith('V')or pos_kkm[i][j][1].startswith('M'):
                dic_kkm[i].append(pos_kkm[i][j][0])
    dic_kkm[i].append(int(score_all[i]))


# In[419]:


# 긍/부정 빈도수 체크하여 상위 100개를 사전으로 저장
dic_text_pos=[]
dic_text_neg=[]  
for i in range(len(dic_kkm)):
    for j in range(0, len(dic_kkm[i])):
        if type(dic_kkm[i][j]) != int:
            if len(dic_kkm[i][j])>1:
                if float(dic_kkm[i][ -1]) > avr_all:
                    dic_text_pos.append(dic_kkm[i][j])
                else:
                    dic_text_neg.append(dic_kkm[i][j])

fdist1=FreqDist(dic_text_pos)
dic_text_pos=fdist1.most_common(100)

fdist1=FreqDist(dic_text_neg)
dic_text_neg=fdist1.most_common(100)


# In[420]:


dic_text_pos1=[]
dic_text_neg1=[]

for i in range(len(dic_text_pos)):
    dic_text_pos1.append(dic_text_pos[i][0])
    
for i in range(len(dic_text_neg)):
    dic_text_neg1.append(dic_text_neg[i][0])

print(dic_text_pos1)
print(dic_text_neg1)


# # 데이터 전처리

# In[421]:


sent_num=0
text_sent=[]

for i in range(len(text_all)):
    text_pos=kkm.pos(text_all[i])
    sent_num+=1
    for t in text_pos:
        n=list(t)
        n.insert(0, "Sentence: {}".format(sent_num))
        if float(score_all[(sent_num-1)]) > avr_all:
            n.insert(3, '100')
        else: 
            n.insert(3, '0')        
        text_sent.append(n)


# In[422]:


len(text_sent)


# In[423]:


# 불용어 처리 전 token 빈도 수 확인
words=[]
for t in text_sent :
    words.append(t[1])


# In[424]:


# 단어 사전 만들기 (key : value = word : 빈도 수)
words_dict = {} # 
for word in words:
    if word in words_dict:
        words_dict[word] += 1
    else:
        words_dict[word] = 1


# In[425]:


sort_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)    


# In[426]:


rank=0

for k, v in sort_dict:
    rank+=1
    plt.plot(rank, words_dict[k], 'bo')
plt.xlabel('rank')
plt.ylabel('frequency')
plt.show()


# In[427]:


rank=0
for k, v in sort_dict:
    rank+=1
    plt.plot(math.log(rank), math.log(v), 'go')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.show()


# In[428]:


text_del_SW = [] # POS(조사, 특수문자, 접사) 제거

for t in text_sent : 
    if ( (t[2][0] !='E') & (t[2][0] !='J') & (t[2][0] !='S') ):
        text_del_SW.append(t)


# In[429]:


# (수동)불용어 사전 만들기
stopwords2=['하', '수', '있', '이','것', '되', '등', '적', '들', '성', '일', '보', '롭', '주']
text_del_SW2 = []
for t in text_del_SW: 
    if t[1] not in stopwords2:
        text_del_SW2.append(t)


# In[430]:


len(text_sent) # 불용어 처리 전 token 수


# In[431]:


len(text_del_SW2) # 불용어 처리 후 token 수


# In[432]:


words=[]
for t in text_del_SW2 :
    words.append(t[1])
        


# In[433]:


# 단어 사전 만들기 (key : value = word : 빈도 수)
words_dict = {} # 
for word in words:
    if word in words_dict:
        words_dict[word] += 1
    else:
        words_dict[word] = 1


# In[434]:


pprint.pprint(sorted(words_dict.items(), key=lambda item: item[1], reverse=True)[:10])


# # CRF model 적용해보기

# In[435]:


data = pd.DataFrame(text_del_SW2) # data를 pandas로 변환


# In[436]:


data.columns = ['Sentence #', 'Word', 'POS', 'score']


# In[437]:


data.head(10)


# In[438]:


data.tail(10)


# In[439]:


data.isnull().sum() # 비어있는 값 확인 결과: 없음


# In[440]:


words = list(set(data["Word"].values))
n_words = len(words) # 고유한 단어의 수: 4668개
n_words


# In[441]:


# 데이터에서 문장을 원하는 형태로 포매팅하여 추출하는 클래스
# Word, POS -> data를 여러 함수로 씌워서 봄. 
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        # for문 사용법 보기. lambda: 일종의 함수(적용하는), data에 다음과 같은 기능 적용
        agg_func = lambda s: [(w, p, s) for w, p, s in zip(s["Word"].values.tolist(),
                                                            s["POS"].values.tolist(),
                                                            s["score"].values.tolist())]
        # sentence 단위로 grouped
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        # 그룹화된 데이터를 다시 sentences로 저장
        self.sentences = [s for s in self.grouped]
        print(self.grouped)
        self.sentences
    
    # get_next: 다음 문장 처리
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[442]:


# 전체 문장 추출
getter = SentenceGetter(data)


# In[443]:


# 문장 하나
sent = getter.get_next()


# In[444]:


# 추출한 문장 확인(적용 결과)
print(sent)


# In[445]:


# 전체 문장
sentences = getter.sentences


# In[446]:


# 외부의 감정 사전(긍정어 사전과 부정어 사전)을 CRF Features로 활용
dict_pos=[]
dict_neg=[]
for line in open("C:/MY/대학/3학년 2학기/텍스트마이닝/final project/word_dict/pos_pol_word.txt", 'rt', encoding='UTF8'):
    dict_pos.append(line.split('\n')[0])
    
for line in open("C:/MY/대학/3학년 2학기/텍스트마이닝/final project/word_dict/neg_pol_word.txt", 'rt', encoding='UTF8'):
    dict_neg.append(line.split('\n')[0])


# In[447]:


# CRF features
def word2features(sent, i):
    word = sent[i][0]   # index0의 단어를 가져옴
    postag = sent[i][1] # index1의 단어를 가져옴
    score = sent[i][2]

    features = {
        'bias': 1.0,
        'postag': postag,
        'postag[:2]': postag[:2],
        'dic_text_pos' : True if word in dic_text_pos1 else False,
        'dic_text_neg' : True if word in dic_text_neg1 else False,
        'dict_pos' : True if word in dict_pos else False,
        'dict_neg' : True if word in dict_neg else False,
        'word.len' : len(word)
    }
    
    # 직전의 단어에 대해서도 똑같이 적용
    if i > 0:
        word1 = sent[i-1][0] # i-1: 직전의 단어
        postag1 = sent[i-1][1]
        features.update({
            '-1:postag': postag1,
            '-1:postag[:2]' : postag1[:2],
            '-1:dic_text_pos' : True if word1 in dic_text_pos1 else False,
            '-1:dic_text_neg' : True if word1 in dic_text_neg1 else False,
            '-1:dict_pos' : True if word1 in dict_pos else False,
            '-1:dict_neg' : True if word1 in dict_neg else False,
            '-1:word.len' : len(word1)
        })
       
    if i > 1:
        word2 = sent[i-2][0] # i-2: 2개 전의 단어
        postag2 = sent[i-2][1]
        features.update({
            '-2:postag': postag2,
            '-2:postag[:2]' : postag2[:2],
            '-2:dic_text_pos' : True if word2 in dic_text_pos1 else False,
            '-2:dic_text_neg' : True if word2 in dic_text_neg1 else False,
            '-2:dict_pos' : True if word2 in dict_pos else False,
            '-2:dict_neg' : True if word2 in dict_neg else False,
            '-2:word.len' : len(word2)
        })

    # 다음 단어에 대해서도 똑같이 처리
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],            
            '+1:dic_text_pos' : True if word1 in dic_text_pos1 else False,
            '+1:dic_text_neg' : True if word1 in dic_text_neg1 else False,
            '+1:dict_pos' : True if word1 in dict_pos else False,
            '+1:dict_neg' : True if word1 in dict_neg else False,
            '+1:word.len' : len(word1)
        })
    # 다다음 단어에 대해서도 똑같이 처리    
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],            
            '+2:dic_text_pos' : True if word2 in dic_text_pos1 else False,
            '+2:dic_text_neg' : True if word2 in dic_text_neg1 else False,
            '+2:dict_pos' : True if word2 in dict_pos else False,
            '+2:dict_neg' : True if word2 in dict_neg else False,
            '+2:word.len' : len(word2)
        })
    return features

# sentence 전체에 적용하기 위한 함수
def sent2features(sent):
    for i in range(len(sent)):
        print(word2features(sent, i))
    return [word2features(sent, i) for i in range(len(sent))] 

def sent2score(sent): 
    return [score for token, postag, score in sent]

def sent2tokens(sent):
    return [token for token, postag, score in sent]


# In[448]:


sent2score(sent)


# In[449]:


print(sent2features(sent))


# In[450]:


X = [sent2features(s) for s in sentences] # sentence마다 feature로 바뀌고
y = [sent2score(s) for s in sentences]    # sentence마다 score로 바뀜


# In[451]:


print(len(X))
print(len(y))
print(len(sentences))


# In[452]:


#10번째 sentence의 정보 
print(len(X[10])) 
print(len(y[10]))
print(sentences[10])
print(y[10])
print(X[10])


# In[453]:


# 문장 하나 확인
# 각 단어의 features가 dictionary로 표현되고 이를 요소로 하는 리스트
X[0]


# In[454]:


print(len(X), len(X)*0.7) #train-test split (70%, 30%)
print(type(X))


# In[608]:


crf = CRF(algorithm='lbfgs',
          c1=2, #c1의 가중치를 조절하여 feature의 영향력 조절
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)


# In[609]:


# Training
crf.fit(X[:1786], y[:1786])


# In[610]:


# cross validation
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=4) 


# In[598]:


# prediction
y_test_pred = crf.predict(X[1786:])


# In[611]:


# report
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

# data + deep learning 적용 시 늘어날 가능성은 있음. 


# In[612]:


# 태그 간의 transition(전이) probabilities, 태그 별 예측에 중요한 features
# eli5: weight visualization
eli5.show_weights(crf, top=30) # -> 중요한 것만 확인


# 
# 
# 
# 
# 
# 
# # 다른 분류 기법(Naive Bayes, Logistic Regression, SVM, ANN) 적용해보기

# # 데이터 전처리(추가)

# In[461]:


# (수동)불용어 사전 만들기 <- 빈도수 상위 정렬 시 긍/부정 판단이 어려운 중의적 단어, 무의미해 보이는 단어
stopwords3=['회사', '곳', '업무', '사람', '기업', '직원', '근무', '생각', '싶', '같', '기관', '직장', '분', '다', 
            '가지', '직', '중', '및', '그', '공공',' 조직', '내', '점', '지', '가', '거', '팀', '사업', '임', '말',
            '여러', '갈', '때', '업', '한', '사내', '사', '규모', '위', '별', '쓰', '년', '만큼', '라', '오', '함', 
            '모든', '그러', '시', '만', '해보', '관련', '정말', '배', '하', '수', '있', '이','것', '되', '등', '적',
            '들', '성', '일', '보', '롭', '주']
text_del_SW3 = []
for t in text_del_SW: 
    if t[1] not in stopwords3:
        text_del_SW3.append(t)


# In[462]:


len(text_del_SW3)


# In[463]:


text_del_SW3[:10]


# In[464]:


# 불용어 제거 후 token 빈도 수 확인
words_SW=[]
for t in text_del_SW3 :
    words_SW.append(t[1])


# In[465]:


# 단어 사전 만들기 (key : value = word : 빈도 수)
words_dict_SW = {} # 
for word in words_SW:
    if word in words_dict_SW:
        words_dict_SW[word] += 1
    else:
        words_dict_SW[word] = 1


# In[466]:


sort_dict_SW = sorted(words_dict_SW.items(), key=lambda x: x[1], reverse=True)    


# In[467]:


sort_dict_SW


# In[468]:


rank=0
for k, v in sort_dict_SW:
    rank+=1
    plt.plot(rank, words_dict_SW[k], 'bo')
plt.xlabel('rank')
plt.ylabel('frequency')
plt.show()


# In[469]:


rank=0
for k, v in sort_dict_SW:
    rank+=1
    plt.plot(math.log(rank), math.log(v), 'go')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.show()


# In[470]:


len(text_del_SW3)


# In[471]:


text_freq = []
for t in text_del_SW3:
    temp=list(t)
    word_n = words_dict_SW[t[1]]
    temp.insert(4, word_n)
    text_freq.append(temp)


# In[472]:


text_freq[:10]


# In[473]:


# 빈도수 5개 이하 삭제
text_del_words = []
for t in text_freq:
    temp=list(t)
    if t[4]>5:
        text_del_words.append(temp)


# In[474]:


len(text_del_words)


# In[475]:


sent_n = []
for i in range(len(text_del_words)):
    sent_n.append(re.sub(r'[^0-9]', '', text_del_words[i][0]))


# In[476]:


X_text=[]
y_score=[]
sent=[]
for i in range(len(sent_n)-1):
    if (int(sent_n[i]) == int(sent_n[i+1])):
        sent.append(text_del_words[i][1])
    else:
        X_text.append(sent)
        y_score.append(text_del_words[i][3])        
        sent=[]


# In[477]:


X_text[:10]


# In[478]:


y_score[:10]


# In[479]:


tokenizer = Tokenizer()
X_text_token = tokenizer.fit_on_texts(X_text)


# In[480]:


X_text_token


# In[481]:


print(len(tokenizer.word_index))
print(tokenizer.word_index)
# 빈도수가 높은 단어 순서대로 index 부여


# In[482]:


threshold = 3
total_cnt = len(tokenizer.word_index)
rare_cnt=0 # threshold 보다 적은 빈도의 단어 수 count
total_freq = 0
rare_freq = 0

for k, v in tokenizer.word_counts.items():
    total_freq = total_freq + v
    
    if(v < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + v


# In[483]:


print(total_cnt)
print(rare_cnt)


# In[484]:


vocab_size = total_cnt - rare_cnt + 1
print(vocab_size)


# In[485]:


len(X_text)


# In[486]:


tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_text)
X_text = tokenizer.texts_to_sequences(X_text)

X_train, X_test, y_train, y_test = train_test_split(X_text, y_score, test_size=0.3, random_state=111)


# In[487]:


X_text


# In[488]:


len(X_train)


# In[489]:


y_test[:10]


# In[490]:


drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]


# In[491]:


drop_train # 비어 있는 샘플 확인


# In[492]:


for index in drop_train:
    del X_train[index]
    del y_train[index]


# In[493]:


len(X_train)


# In[494]:


len_result = [len(s) for s in X_train]

print('리뷰의 최대 길이 : {}'.format(np.max(len_result)))
print('리뷰의 평균 길이 : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()


# In[495]:


max_len=25


# In[496]:


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))


# In[497]:


below_threshold_len(max_len, X_train)


# In[498]:


unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))


# In[499]:


word_to_index = tokenizer.word_index
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key


# In[500]:


print('빈도수 상위 1등 단어 : {}'.format(index_to_word[4]))


# In[501]:


# (index 변환 전) X_train[0] 한국어 단어 표시
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token

print(' '.join([index_to_word[index] for index in X_train[0]]))


# In[502]:


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[503]:


X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


# In[504]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[505]:


# Naive bayes training
nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


# In[506]:


# Logistic Regression training

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
# Logistic Regression prediction
y_pred_logreg = logreg.predict(X_test)
print(classification_report(y_test, y_pred_logreg))


# In[630]:


# ANN
ANN = MLPClassifier(solver='lbfgs', alpha=2, hidden_layer_sizes=(30, 2), random_state=3)
ANN.fit(X_train, y_train)
y_pred_ANN = ANN.predict(X_test)
print(classification_report(y_test, y_pred_ANN))


# In[616]:


# k-Nearest Neighbor
knn  = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))


# In[509]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_DT = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred_DT))


# In[629]:


# SVM
svc = SVC(gamma='scale', kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(classification_report(y_test, y_pred_svc))


# In[ ]:




