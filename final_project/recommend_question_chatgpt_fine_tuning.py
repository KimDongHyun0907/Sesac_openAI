import os, getpass
from openai import OpenAI

api_key = getpass.getpass(prompt='OpenAI API키 입력')

client = OpenAI(
    api_key=api_key
)

category_list = {'라이프스타일': ['건강과 웰빙', '여행', '음식과 요리', '취미', '패션과 뷰티', '개인 재정', '가정과 생활', '인간 관계', '자기 개발', '기술과 가젯'], '창의성과 예술': ['미술', '음악', '문학', '디자인', '영화 및 비디오', '공연 예술', '사진', '창작 글쓰기', '공예', '건축'], '사회': ['정치', '경제', '문화', '사회', '심리', '교육', '역사', '법학', '언론 및 커뮤니케이션', '인류'], '과학/기술': ['과학', '수학', '공학', '컴퓨터 과학', '의학', '환경과학', '신재생 에너지', '우주과학', '로보틱스', '나노기술']}

print('카테고리를 선택해 주세요')
text = ''
for idx, data in enumerate(category_list):
    text+=f'{idx+1}. {data}\n'

print(text)
select_category1 = input()

print('하위 카테고리를 선택해 주세요.')
text2=''
for idx, data in enumerate(category_list[select_category1]):
    text2+=f'{idx+1}. {data}\n'

print(text2)
select_category2 = input()


completion = client.chat.completions.create(
    model = 'ft:gpt-3.5-turbo-0125:personal::9EU6CDLe',
    messages=[
        {'role': 'user', 'content': f'{select_category1}, {select_category2}'}
    ]
)

print(completion.choices[0].message.content)
