import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from konlpy.tag import Okt

# ----------------------------
# 1. CSV 파일에서 conjugation_dict 로드
# ----------------------------
def load_conjugation_dict(csv_file):
    """
    CSV 파일에서 동사/형용사 활용형 사전을 불러옵니다.
    """
    df = pd.read_csv(csv_file)
    conjugation_dict = {}
    for _, row in df.iterrows():
        word = row['word']
        particle = row['particle'] if pd.notna(row['particle']) else None
        conjugation = row['conjugation']
        conjugation_dict[(word, particle)] = conjugation
    return conjugation_dict

conjugation_dict = load_conjugation_dict('conjugation_dicts.csv')

# ----------------------------
# 2. 동사/형용사 활용형 변환
# ----------------------------
def conjugate_word(word, particle=None, is_sentence_end=False):
    """
    동사/형용사와 조사에 따라 활용형을 반환합니다.
    문장 끝일 때는 is_sentence_end를 True로 설정합니다.
    """
    if is_sentence_end:
        # 문장 끝일 때의 활용형 반환
        return conjugation_dict.get((word, '문장 끝'), word)
    elif particle is None or particle == '':
        # 조사가 없을 때 사전에서 단독 매핑을 찾아 반환
        return conjugation_dict.get((word, None), word)
    else:
        # 조사와 함께 매핑을 찾아 반환
        return conjugation_dict.get((word, particle), word)

# ----------------------------
# 3. 조사 예측 함수 학습 코드에서 모델이 학습한 패턴을 기반으로 실제 예측을 수행하는 역할
# ----------------------------
def predict_postpositions(nouns, verbs_adjs, model, tokenizer, tokenizer_label, max_len, okt):
    """
    명사에 적절한 조사를 예측하고 할당합니다.
    """
    # 입력 문장 생성
    combined_sentence = ' '.join(nouns + verbs_adjs)
    
    # 형태소 분석
    morphs = okt.pos(combined_sentence)
    tokens = [morph for morph, tag in morphs]
    tags = [tag for morph, tag in morphs]
    
    # 명사 위치 파악
    noun_indices = [i for i, tag in enumerate(tags) if tag == 'Noun']
    nouns = [tokens[i] for i in noun_indices]
    
    # 조사 예측을 위한 시퀀스 생성
    input_seq = tokenizer.texts_to_sequences([' '.join(tokens)])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    # 모델 예측
    output_seq = model.predict(input_seq)
    
    # 예측된 레이블 인덱스 추출
    pred_labels = np.argmax(output_seq, axis=-1)[0]  # (max_len,)
    
    # 레이블 역변환
    inverse_label_index = {v: k for k, v in tokenizer_label.word_index.items()}
    decoded_labels = [inverse_label_index.get(label, 'O') for label in pred_labels]
    
    # 명사에 조사 할당
    assigned_particles = []
    for idx in noun_indices:
        if idx < len(decoded_labels):
            particle = decoded_labels[idx]
            if particle.upper() != 'O':  # 대소문자 구분 없이 'O' 처리
                assigned_particles.append(particle)
            else:
                assigned_particles.append('')
        else:
            assigned_particles.append('')
    
    return nouns, assigned_particles, morphs

# ----------------------------
# 4. 최종 예측 함수 통합
# ----------------------------
def map_predictions(sentence):
    """
    예측된 문장에서 '사람'을 '청인'으로 변환합니다.
    """
    return sentence.replace('사람', '청인')

def predict_final_sentence(nouns, verbs_adjs, transformer_model, tokenizer_postposition, tokenizer_label_postposition, max_len_postposition, okt, original_input):
    """
    명사와 동사/형용사를 입력받아 최종 문장을 예측합니다.
    """
    # 단일 명사이고 동사/형용사가 없는 경우, 조사 없이 명사만 반환
    if len(nouns) == 1 and len(verbs_adjs) == 0:
        final_sentence = nouns[0]
        final_sentence = map_predictions(final_sentence)
        return final_sentence
    
    # 1. 조사 예측 및 명사에 조사 할당
    nouns, assigned_particles, morphs = predict_postpositions(
        nouns, 
        verbs_adjs, 
        transformer_model, 
        tokenizer_postposition, 
        tokenizer_label_postposition, 
        max_len_postposition, 
        okt
    )
    
    # 입력된 단어 순서에 따라 결과를 조합
    final_tokens = []
    noun_idx = 0
    verb_idx = 0

    for i, word in enumerate(original_input):
        is_sentence_end = (i == len(original_input) - 1)  # 문장의 마지막 단어인지 확인

        if noun_idx < len(nouns) and word == nouns[noun_idx]:
            # 명사일 경우, 조사와 결합
            particle = assigned_particles[noun_idx]
            combined_noun = f"{word}{particle}" if particle else word
            final_tokens.append(combined_noun)
            noun_idx += 1
        elif verb_idx < len(verbs_adjs) and word == verbs_adjs[verb_idx]:
            # 동사/형용사일 경우 활용형 변환
            conjugated = conjugate_word(word, assigned_particles[noun_idx - 1] if noun_idx - 1 >= 0 else None, is_sentence_end)
            final_tokens.append(conjugated)
            verb_idx += 1
        else:
            # 그 외의 단어는 그대로 추가
            final_tokens.append(word)
    
    final_sentence = ' '.join(final_tokens)

    # "사람"을 "청인"으로 변환
    final_sentence = map_predictions(final_sentence)

    return final_sentence

# ----------------------------
# 5. 사용자 입력 처리 및 예측
# ----------------------------
if __name__ == "__main__":
    okt = Okt()
    
    # 조사 예측 모델 로드
    transformer_model = load_model('transformer_model_with_context.keras')
    
    # 토크나이저 로드
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer_postposition = pickle.load(handle)

    with open('tokenizer_label.pickle', 'rb') as handle:
        tokenizer_label_postposition = pickle.load(handle)

    max_len_postposition = 50  # 조사 예측 모델의 최대 길이

    while True:
        user_input = input("단어를 쉼표로 구분하여 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            break

        words = [word.strip() for word in user_input.split(',') if word.strip()]

        morphs = okt.pos(' '.join(words))

        input_nouns = [morph for morph, tag in morphs if tag == 'Noun']
        input_verbs_adjs = [morph for morph, tag in morphs if tag in ['Verb', 'Adjective']]

        if not input_nouns and not input_verbs_adjs:
            print("입력된 단어에서 명사와 동사/형용사를 추출할 수 없습니다. 다시 입력해주세요.")
            continue

        predicted_sentence = predict_final_sentence(
            input_nouns, 
            input_verbs_adjs, 
            transformer_model, 
            tokenizer_postposition, 
            tokenizer_label_postposition, 
            max_len_postposition, 
            okt,
            words
        )

        print(f"입력: {user_input} -> 예측된 문장: {predicted_sentence}")