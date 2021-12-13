#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import time
import re
import pandas as pd
from bs4 import BeautifulSoup
from konlpy.tag import Kkma
from konlpy.tag import Okt


# In[4]:


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


# In[5]:


def ind_reviews(code):
    url = "https://www.jobplanet.co.kr/reviews?&industry_id=" + str(code)
    response =  session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    num = soup.find('span', class_='num') # 리뷰 개수 확인
    num = int(num.get_text().strip())
    import math, random
    pages = math.ceil(num / 10)
    page = random.sample(range(1, pages),30)
    reviews = {}
    for i in page: # 페이지를 랜덤으로 하여 300개의 리뷰 추출
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


# In[6]:


# 각 산업군의 기업 리뷰 스크래핑
reviews1 = dict(ind_reviews(1001))
reviews2 = dict(ind_reviews(900))


# In[31]:


#특수문자 제거
def sub(dic):
    dic = [re.sub(r"[^가-힣A-Za-z0-9]", " ", str(content)) for content in dic]
    return dic


# In[32]:


#리뷰별 키워드 추출 후 중첩 리스트 생성  
def append(text):
    temp=[]
    for i in text:
        temp.append(okt.nouns(i))
    return temp


# In[33]:


#산업군 별 키워드 추출 후 리스트 생성
def extend(text):
    temp=[]
    for i in text:
        temp.extend(okt.nouns(i))
    return temp


# In[34]:


reviews_1 = sub(reviews1)
reviews_1


# In[35]:


okt=Okt()
okt.nouns(reviews_1[0])


# In[36]:


#KKMA를 사용할 경우 성능이 좋지 않음. okt랑 비교해봤을 때 okt를 사용하는 편이 좋다고 판단.
kkma=Kkma()
kkma.nouns(reviews_1[0])


# In[40]:


goverment= extend(reviews_1)
goverment[:6]


# In[38]:


reviews_2 = sub(reviews2)
reviews_2


# In[41]:


bank= append(reviews_2)
bank[:6]


# In[ ]:




