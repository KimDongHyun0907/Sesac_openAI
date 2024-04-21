import torch
from hangeul_onehot_vector import make_input_vector
from CNN_badwords_filtering import TextCNN
from sensitive_info_masking import pattern_match

# 입력값을 받아 레이블을 출력하는 함수
def predict_label(input_data, model_path):
    # 모델 및 가중치 불러오기
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # 모델 평가 모드로 설정

    # 입력값을 텐서로 변환
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 배치 차원 및 채널 차원 추가
    # 모델에 입력값 전달하여 출력 계산
    with torch.no_grad():
        output = model(input_tensor)
    # 출력값을 확률로 변환하여 레이블 출력
    _, predicted = torch.max(output, 1)
    
    return predicted.item()  # 레이블 반환

def filtering_and_masking(text):
    input_vector = make_input_vector(text)
    predicted_label = predict_label(input_vector, "best_filtering_model.pth")

    if predicted_label:
        return pattern_match(text)
    else:
        return '비속어이므로 필터링합니다.'
    
if __name__=='__main__':
    # 예시 입력값
    sample_text = "이 새끼 와꾸 세월 정통으로 쳐맞았네"
    # sample_text = '개인정보를 확인해주세요. 휴대폰 번호 010-1234-5678. 이메일 주소 abcd@gmail.com. 주민등록번호 240423-3123456'
    print(filtering_and_masking(sample_text))