# 제주어-한국어 기계 번역  

### 1. 데이터 수집

**(1) train data** : jejufinaltrain.txt / koreafinaltrain.txt 

- 카카오브레인 제공 제주어-한국어 말뭉치 (**160,356개**) 
  - 제주어구술자료집 2017, 2018년 버전 가공

- 제주어구술자료집 2019, 2020년 버전 가공 (**127,861개**)
  1. pdf 파일 txt 파일로 변환 
  2. 필요한 인터뷰 데이터만 남기고 제주어, 한국어 분리 
  3. 제주어 데이터에 사용된 옛한글 한컴 입력기 통해 변환 
     - 옛한글 유니코드 : https://bit.ly/3oQSEQx
     - pyhwp 등은 옛한글 처리 x 
     - 아래아 제외한 옛한글은 한컴입력기를 통해 입력이 안 되고 카카오 데이터에서도 아래아만 확인이 되어 다른 옛한글 포함 문장은 제외 
  4. 불필요한 줄바꿈 처리 
- 위키 한국어 문장 데이터 셋 
  - 타겟 언어로만 이루어진 데이터를 추가하는 것이 기계 번역 성능 향상 도움이 됨
  - 관련 논문 : 'Improving neural machine translation models with monolingual data'
  - https://github.com/as9786/ParrotnlpJeju/blob/main/old_conv_list/monolingual_data.pdf

**(2) dev data** : je.dev / ko.dev

- 카카오브레인 제공 제주어-한국어 말뭉치 5000개 

**(3) test data** : je.test / ko.test 

- 카카오 브레인 제공 제주어-한국어 말뭉치 5000개 



### 2. 데이터 전처리 

- BPE 알고리즘으로 **단어 집합 생성** (by sentencepiece)

  - 처음에는 제주어 데이터, 한국어 데이터 따로 단어 집합 사용 

  - 한국어-한국어 말뭉치를 넣을 경우 **input과 output의 단어 인덱스가 같아야** 하는 문제 

  - [Neural Machine Translation of Rare Words with Subword Units(2016)] 논문 :

    source 언어와 target 언어의 단어집합을 따로 만들 경우 같은 단어가 다른 방식으로  분리되어 두 **단어를 매핑에 어려움**이 생길 수 있으므로 **하나의 단어 집합으로 인코딩을 하는 것이 단어들의 일관성을 높여줌 **

    - http://hiai.co.kr/wp-content/uploads/2019/12/%EB%85%BC%EB%AC%B8%EC%A6%9D%EB%B9%99_2019_05.pdf

    - https://www.aclweb.org/anthology/P16-1162/

  - 제주어와 한국어의 경우 겹치는 단어가 많기 때문에 하나의 단어 집합으로 통합하는 것이  쉬움

  - concatenate jejufinaltrain.txt, koreafinaltrain.txt => 단어 집합 생성 

  - 단어 집합 크기 : 4000 

- 토큰화 

  - START_TOKEN, END_TOKEN 추가 
  - 패딩 : MAXLEN 



### 3. transformer 모델 구현 

##### 1. wiki docs 참고 모델 

- 사전 학습이 된 모델이 아니다보니 성능이 잘 나오지 않음
- 'attention is all you need' 논문의 모델을 구현
- 모델 구조에 확신이 없음 

##### 2. pytorch fairseq 이용 

- 카카오브레인 논문과 비슷한 성능 
- 데이터 추가, 하이퍼 파라미터 조정으로 성능 향상 (할 것)

##### 3. 사전 학습 모델 

- kobart, 등.... https://github.com/SKT-AI/KoBART



### 4. BLEU score 

##### 1. nltk bleu 

- 질 낮은 번역에도 점수가 높게 나옴 
- smoothing_function = smoothie 설정해주니까 조금 개선되지만 괜찮은걸까..?
- https://www.nltk.org/_modules/nltk/translate/bleu_score.html

##### 2. sacrebleu

- 문자열을 토큰화하지 않은 채로 그대로 넣어야 함.
- 영어 문장은 괜찮겠지만 한국어 토큰화도 잘 할 수 있을까?!
- https://github.com/mjpost/sacrebleu

##### 3. multi-bleu.perl

- 사용해보는 중
- https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl

##### 4. fairseq bleu 



