from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import warnings
warnings.filterwarnings(action='ignore')


### 모델 훈련을 위한 함수 및 클래스

# 1. 모델의 input 데이터 생성
def convert_to_features(texts, labels, max_seq_len, tokenizer):
	# 모델 훈련에 필요한 input, attention_mask, type_id 생성
  input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
  for text, label in tqdm(zip(texts, labels), total=len(text)):
    input_id = tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True)
    padding_count = input_id.count(tokenizer.pad_token_id)
    attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
    token_type_id = [0] * max_seq_len

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    data_labels.append(label)
  # Numpy의 array로 변환
  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  data_labels = np.asarray(data_labels, dtype=np.int32)

  return (input_ids, attention_masks, token_type_ids), data_labels


# 2. 모델 클래스
class TFBertForSequenceClassification(tf.keras.Model):
	# model checkpoint : "klue/bert-base"
  def __init__(self, model_name):
    super(TFBertForSequenceClassification, self).__init__()
    self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
    self.classifier = tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),\
                                            activation='sigmoid',\
                                            name='classifier')
  def call(self, inputs):
    input_ids, attention_mask, token_type_ids = inputs
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    cls_token = outputs[1]
    prediction = self.classifier(cls_token)
    return prediction

# 3. 추론 함수
def sentiment_predict(sentences):
    result = []
    # 리뷰를 하나씩 꺼내 정제 후 추론. 0.5이상이면 1(긍정) else 0(부정)
    for sentence in sentences:
        sentence = re.sub(r"[^ㄱ-ㅎ|가-힣]", "", sentence)
        input_id = tokenizer.encode(sentence, max_length=max_seq_len,
                pad_to_max_length=True)

        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        token_type_id = [0] * max_seq_len

        input_ids = np.array([input_id])
        attention_masks = np.array([attention_mask])
        token_type_ids = np.array([token_type_id])

        encoded_input = [input_ids, attention_masks, token_type_ids]
        score = model.predict(encoded_input)[0][0]

        if(score > 0.5):
            result.append(1)
        else:
            result.append(0)
	  
    return result


if __name__ == "__main__":
  # 학습을 위한 Naver shopping review 데이터 로드 후 훈련셋 분리
    pretrain_data = pd.read_csv("./naver_review.csv")
    X_train, X_test, y_train, y_test = train_test_split(pretrain_data["reviews"], pretrain_data["label"], test_size=0.2, random_state=42)

    # 토크나이저와 패딩길이를 선언 후 적합한 input데이터로 변환 (위 함수사용)
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    max_seq_len = 128
    X_train, y_train = convert_to_features(X_train, y_train, max_seq_len=max_seq_len, tokenizer=tokenizer)
    X_test, y_test = convert_to_features(X_test, y_test, max_seq_len=max_seq_len, tokenizer=tokenizer)

    # 모델 훈련
    # 사전학습 모델 : "klue/bert-base"
    # optimizer : Adam / lr = 5e-5
    # loss : binary crossentropy
    # mecrics : Accuracy(정확도)
    model = TFBertForSequenceClassification("klue/bert-base")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs=3, batch_size=64)
    results = model.evaluate(X_test, y_test, batch_size=128)

    # 모델 저장 후 테스트셋 성능 출력
    model.save("./bert_model")
    print(results)

    # 컬리 리뷰데이터를 불러와 추론 함수를 통해 긍부정 분류. "emotion_label"컬럼으로 추가
    kurly_review = pd.read_csv("./kurly_review.csv")
    result = sentiment_predict(kurly_review ["review"])
    kurly_review ["emotion_label"] = result