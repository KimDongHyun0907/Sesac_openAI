# 0 비속어, 민감정보 둘 다 없음  false, false  
# 1 비속어만 있음  true false  
# 2 민감정보만 있음  false true  
# 3 비속어, 민감정보 둘 다 있음  true true
# 

import re
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

model_name = 'sgunderscore/hatescore-korean-hate-speech'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        device = -1, # gpu: 0
        return_all_scores = True,
        function_to_apply = 'sigmoid')

# sample_text = "010-1234-5678 이 번호로 연락오면 받아라. 안 받으면 죽여버린다 개새끼야."
sample_text = "010-1234-5678 이 번호로 연락오면 받아라. aaa@naver.com, 971515-1234567"


def pattern_match(text):
    email_pattern = r'\b([A-Za-z0-9._%+-]{2})([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\d{3})-(\d{4})-(\d{4})'
    resident_id_pattern = r'(\d{6})-?([1-4])(\d{6})'

    # 이메일 마스킹
    def mask_email(match):
        username, domain = match.group().split('@')
        username = username[0]+'*'*len(username[1:])
        domain = '*'*len(domain.split('.')[0])+'.'+'*'*len(domain.split('.')[1])
        return f"{username}@{domain}"

    # 전화번호 마스킹
    def mask_phone(match):
        return match.group(1) + "-****-****"

    # 주민등록번호 마스킹
    def mask_resident_id(match):
        return match.group().split('-')[0] + '-'+ len(match.group().split('-')[1])*'*'
    
    # 이메일 마스킹 적용
    masked_text = re.sub(email_pattern, mask_email, text)

    # 전화번호 마스킹 적용
    masked_text = re.sub(phone_pattern, mask_phone, masked_text)

    # 주민등록번호 마스킹 적용
    masked_text = re.sub(resident_id_pattern, mask_resident_id, masked_text)

    return masked_text

def bad_words_find(text):
    label_and_score = pipe(text)[0]
    label_and_score = sorted(label_and_score, key=lambda x:-x['score'])
    return False if label_and_score[0]['label'] == 'None' else True

if bad_words_find(sample_text):
    print('비속어이므로 필터링합니다.')
else:
    masked_text = pattern_match(sample_text)
    print(masked_text)