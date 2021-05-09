# Kakao Jejueo

 한국어-제주어 기계 번역 

https://kakaobrain.com/publication/110

https://github.com/kakaobrain/jejueo/tree/master/translation

https://www.kakaobrain.com/blog/119

#### 1.데이터 셋 구축 

- <제주어 문장, 한국어 문장>으로 구성된 17만 개의 병렬 말뭉치

- 훈련 데이터 160,356쌍, 검증 데이터와 테스트 데이터 각각 5,000쌍

- 과정 

  (1) pdf 파일 txt 파일로 변환 

  (2) 인터뷰 데이터만 남김 

  (3) 제주어와 한국어 분리 

  - 제주어 (한국어) 형태로 저장되어 있는 문서를 분리 
  - 줄바꿈 중간중간 발생 => ^로 표시 => ^가 없는 단어가 다른 곳에 나오면 그 단어로 바꾸고 없으면 ^을 공백으로 바꿈 
  - punctuation 
  - 아래아와 같은 옛한글 standard 유니코드로 
  - train, test, dev 데이터로 split => dev, test 데이터는 한 문장에 단어 5개 이상 들어가는 문장들로 구성 



#### 2. 데이터 전처리 

- **BPE** : 서브워드 분리 알고리즘 (하나의 단어를 여러 서브워드로 분리해서 인코딩 => OOV 문제 완화 )

  - 기본 원리 : 연속적으로 가장 많이 등장한 글자의 쌍  찾아서 하나의 글자로 병합 

  - ```python
    # 기본 원리 
    aaabdaaabac
    ↓
    XdXac
    X=ZY
    Y=ab
    Z=aa
    ```

  - Bottom up 방식 (구현) : 유니코드 단위로 단어 집합 만든 후 가장 많이 등장하는 유니그램 하나의 유니그램으로 통합하여 단어 집합에 추가 

  - ```python
    # 초기 딕셔너리 
    l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3
    
    l, o, w, e, r, n, w, s, t, i, d
    ↓
    # dictionary에 (e, s) 쌍 가장 빈도수 높음 => 하나로 통합하여 단어 집합에 추가 
    l, o, w, e, r, n, w, s, t, i, d, es
    ↓
    l, o, w, e, r, n, w, s, t, i, d, es, est
    ↓
    ...
    ↓
    l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
    # =>  최종 딕셔너리
    low : 5,
    low e r : 2,
    newest : 6,
    widest : 3
    
    # 이렇게 하면 새로운 단어 lowest에 대해 bpe 알고리즘은 
    # 단어 집합에 있는 low와 est 두 단어로 인코딩 
    ```

- **SentencePiece** : BPE를 포함한 기타 서브워드 토크나이징 알고리즘 내장한 패키지 (by 구글)

  - https://github.com/google/sentencepiece

  - 사전 토큰화 작업 없이 단어 분리 토큰화 수행 

  - 제주어로 sentencepiece 모델 학습 

    ```python
    pip install sentencepiece
    import sentencepiece as spm 
    
    # sentencepiece 모델 학습
    train_je_bpe = '--input=./kakao_data/je.train --normalization_rule_name=identity --model_prefix=je_bpe --vocab_size=4000 --model_type=bpe --character_coverage=0.995'
    spm.SentencePieceTrainer.Train(train_je_bpe)
    
    # sentencepiece 학습을 통해 생성된 단어 집합 
    vocab_list = pd.read_csv('je_bpe.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    
    # 모델 적용 
    sp = spm.SentencePieceProcessor()
    vocab_file = "je_bpe.model"
    sp.load(vocab_file)
    
    # 인코딩
    line = "야 , ᄂᆞᆯ 낭 탈 때ᄁᆞ지"
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    #야 , ᄂᆞᆯ 낭 탈 때ᄁᆞ지
    #['▁야', '▁,', '▁ᄂᆞᆯ', '▁낭', '▁탈', '▁때', 'ᄁᆞ지']
    #[716, 5, 1042, 235, 2221, 26, 1036]
    
    # 디코딩 
    sp.DecodeIds([716, 5, 1042, 235, 2221, 26, 1036])
    # 야 , ᄂᆞᆯ 낭 탈 때ᄁᆞ지
    ```


https://colab.research.google.com/drive/1iyRN7lwSvGek0zc3C24T9usqQzfmV89N?usp=sharing



- **Optimal Vocabulary Size** : 4k 

![jejy](C:\Users\qhdud\OneDrive - Sogang\바탕 화면\jejy.PNG)





#### 3. 모델 : **Transformer** - deep seq2seq architecture 

- original parameter settings of the standard Transformer model
  => 6 encoder/decoder, 각각 512-2048 hidden units across 8 attention heads 

- pytorch 라이브러리 FAIRSEQ 사용 

  https://github.com/pytorch/fairseq

  ```
  export lang1="ko"
  export lang2="je"
  fairseq-train data/4k/${lang1}-${lang2}-bin \
      --arch transformer       \
      --optimizer adam \
      --lr 0.0005 \
      --label-smoothing 0.1 \
      --dropout 0.3       \
      --max-tokens 4000 \
      --min-lr '1e-09' \
      --lr-scheduler inverse_sqrt       \
      --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy       \
      --max-epoch 100 \
      --warmup-updates 4000 \
      --warmup-init-lr '1e-07'    \
      --adam-betas '(0.9, 0.98)'       \
      --save-dir train/4k/${lang1}-${lang2}/ckpt  \
      --save-interval 10
  ```

  

#### 4. 모델 성능 평가 

- BLEU : 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법

  - https://wikidocs.net/31695

- 문장을 그대로 출력하는 태스크보다 점수가 높은지를 기준으로

  ![gf](C:\Users\qhdud\OneDrive - Sogang\바탕 화면\gf.PNG)

  

  ```
  export lang1="ko"
  export lang2="je"
  
  fairseq-generate data/4k/${lang1}-${lang2}-bin \
    --ckpt CKPT \
    --subset {valid,test} \
    --beam-width 5
  ```

  

#### 5. 결과 

- 제주어 -> 한국어 번역이 한국어 -> 제주어 번역보다 점수가 훨씬 높음 
  - 제주어에서는 한국어가 많이 나타나지만 한국어에서는 제주어가 많이 나타나지 않기 때문인 것으로 보임 
    