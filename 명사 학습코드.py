import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 데이터 로드 및 전처리
df_noun = pd.read_csv('noun_dataset.csv')  # 명사 + 동사/형용사 포함 데이터셋

okt = Okt()
sentences = []
labels = []

# 문장을 문맥 전체로 처리하여 명사에 따른 조사 할당
for index, row in df_noun.iterrows():
    text = row['text']
    morphs = okt.pos(text)
    
    sentence = []
    label = []
    i = 0
    while i < len(morphs):
        morph, tag = morphs[i]
        sentence.append(morph)
        
        if tag == 'Noun':
            # 다음 토큰이 명사인지 확인
            if i + 1 < len(morphs):
                next_morph, next_tag = morphs[i + 1]
                if next_tag == 'Josa':
                    label.append(next_morph)  # 명사에 다음 조사를 레이블로 할당
                    i += 1  # 조사는 이미 처리했으므로 건너뜀
                else:
                    label.append('O')  # 조사가 없으면 'O'
            else:
                label.append('O')  # 문장 끝이라면 'O'
        else:
            label.append('O')  # 명사가 아닌 경우 'O'
        
        i += 1
    
    sentences.append(sentence)
    labels.append(label)

# 토크나이저 및 시퀀스 처리
max_len = 50  # 문장 최대 길이 설정

# 입력 시퀀스 토큰화 및 패딩
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
X_train = tokenizer.texts_to_sequences(sentences)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')

# 레이블 시퀀스 토큰화 및 패딩
tokenizer_label = Tokenizer(oov_token='<OOV>')
tokenizer_label.fit_on_texts(labels)
y_train = tokenizer_label.texts_to_sequences(labels)
y_train = pad_sequences(y_train, maxlen=max_len, padding='post')

# y_train을 numpy 배열로 변환
y_train = np.array(y_train)

vocab_size_input = len(tokenizer.word_index) + 1
vocab_size_output = len(tokenizer_label.word_index) + 1

# 트랜스포머 모델 정의 (시퀀스 레이블링)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(vocab_size, maxlen, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, head_size, input_length=maxlen)(inputs)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = LayerNormalization(epsilon=1e-6)(x)
    # 시퀀스 레이블링을 위해 TimeDistributed 사용
    x = TimeDistributed(Dense(mlp_units, activation="relu"))(x)
    x = Dropout(dropout)(x)
    outputs = TimeDistributed(Dense(vocab_size_output, activation="softmax"))(x)  # 시퀀스별 조사 예측
    
    return Model(inputs, outputs)

# 모델 빌드
transformer_model = build_transformer_model(
    vocab_size=vocab_size_input, 
    maxlen=max_len, 
    head_size=64, 
    num_heads=4, 
    ff_dim=64, 
    num_transformer_blocks=4, 
    mlp_units=128, 
    dropout=0.1
)

# 모델 컴파일
transformer_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# y_train의 shape을 (samples, max_len, 1)로 변경 (sparse_categorical_crossentropy를 위해)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

# 모델 요약 확인
transformer_model.summary()

# 모델 학습
transformer_model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2
)

# 모델 저장
transformer_model.save('transformer_model_with_context.keras')

# 토크나이저 저장
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer_label.pickle', 'wb') as handle:
    pickle.dump(tokenizer_label, handle, protocol=pickle.HIGHEST_PROTOCOL)