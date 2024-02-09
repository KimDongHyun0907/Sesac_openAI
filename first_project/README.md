## 프로젝트 : 월클 김삼백
### 팀명 : 손흥민 황희찬 김삼백 Let's Go

### Why 월클 김삼백
- 축구는 세계적으로 가장 사랑받는 스포츠이다.
- 최근 예능 '골 때리는 그녀들'의 인기로 1030 여성 고객의 축구 용품 구매율이 37% 증가했다.
- 또한, 손흥민, 황희찬, 이강인, 김민재 등 여러 대한민국 선수들이 해외 리그에서 좋은 활약을 펼쳐 이전보다 축구를 즐기는 사람들은 증가했다.
- 축구는 득점을 해야 하는 스포츠이며, 골을 넣기 위해 또는 패스를 하기 위해서는 이에 맞는 자세가 중요하다.
- 따라서 슈팅 동작을 통해 공이 왼쪽 또는 오른쪽으로 예측하는 프로그램을 개발하게 되었다.
- 가상 골키퍼 캐릭터인 '김삼백'이 공의 방향을 예측하고 실제 슈팅 방향과 비교할 수 있다.

### 개발 과정
### Tech Stack
#### Language
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">

#### Model
<img src="https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white">

#### Web Front-End
<img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"><img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=white"><img src="https://img.shields.io/badge/css3-1572B6?style=for-the-badge&logo=css3&logoColor=white">  

#### Web Back-End
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white">

#### Tools
<img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"><img src="https://img.shields.io/badge/visual studio code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white">

- AI HUB의 데이터 API를 활용. ([비전영역, 축구 킥 동작 및 축구공 궤적 데이터 구축](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71406))
- 슛 자세 수집 이미지
  - 여성(f) - 인사이드 성인
  - 여성(f) - 인 스텝 성인
  - 여성(f) - 인프런트 성인
  - 남성(m) - 아웃프런트 성인
  - 남성(m) - 인사이드 성인
  - 남성(m) - 인스텝 성인
  - 남성(m) - 인프런트 성인
- 공의 이동 경로 수집을 위한 json
  - trajectory_후기-궤적
- 데이터 전처리
  - 공을 차기 직전의 Frame 사진만 분류 -> 이미지 파일 이름을 통해 분류 가능
  - 후기 궤적 json이 있는 사진만 분류 -> json 파일의 ball location의 마지막 y좌표를 통해 공의 방향을 알 수 있다. 모든 이미지 파일이 json 파일에 대한 정보가 있지 않다는 것을 확인.
  - 구글 Teachable Machine에 이미지 학습을 하기 위해 이미지를 224*224 크기로 resize한다.
  - 결과
    - 공의 방향 오른쪽
      - 왼발 2246장
      - 오른발 7096장
    - 공의 방향 왼쪽
      - 왼발 2759장
      - 오른발 5960장
    - 전처리 후 8 angle의 차기 직전 모습
      - ![image](https://github.com/KimDongHyun0907/Sesac_openAI/assets/88826811/730d2126-aff6-4f15-bd0c-bca37594054f)
- [Teachable Machine](https://teachablemachine.withgoogle.com/) 학습 (이미지 프로젝트)
  - Class를 공의 방향의 왼쪽과 오른쪽 2개를 생성.
  - 모델 학습
    - 차기 직전의 후면 모습과 RGB였던 사진을 Gray사진으로 변환 후 이미지 학습
    - ![image](https://github.com/KimDongHyun0907/Sesac_openAI/assets/88826811/f5969b93-1852-4919-ac4e-9646af25f9ba)
- 김삼백 캐릭터와 웹 페이지에 사용할 디자인 생성
  - 생성형 AI를 사용하여 이미지를 생성
  - [Capitol AI](https://www.capitol.ai/), [Ideogram](https://ideogram.ai/t/explore)
- Flow Chart - 페이지 구성 및 작동 방식
  - ![image](https://github.com/KimDongHyun0907/Sesac_openAI/assets/88826811/8f6cf490-0971-4b22-a7fd-5bf04388e81d)


