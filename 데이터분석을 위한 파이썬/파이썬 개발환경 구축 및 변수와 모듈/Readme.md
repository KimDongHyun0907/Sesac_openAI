### 가상환경 및 파이썬 작업 에디터
- 코드의 문서화가 편리한 jupyter Notebook을 vscode에서 실행함
- 이때 conda를 이용하여 파이썬을 설치하고, conda에서 가상환경(작업 폴더)를 작성하여 그 가상환경을 vs code와 연결하여 작업하는 것
- jupyter notebook 내에서 패키지 설치는 !pip install 패키지명

### 파이썬 모듈 (라이브러리, 패키지)는 별도로 작성된 프로그램 코드 (000.py로 작성됨)
- from PIL import Image  # PIL 폴더의 Image.py를 사용할 수 있게함.
  - Image.open(파일명)
  - 만약 from PIL하면 추후 PIL.Image.open(파일명)으로 사용가능
  - from PIL import Image, ImageFilter처럼 PIL 폴더에서 여러개의 py를 사용할 수 있음
- import numpy  # numpy 폴더의 __init__.py를 사용할 수 있게 함
  - 이때 모듈명을 간편하게 사용하기 위해 as를 넣어 별칭을 줌 (import numpy as np)

### print문 사용법 또는 옵션  
### input문 (input으로 들어온 자료는 문자취급으로, 숫자로 연산해야 할 시 형변환 (type 변환)이 필요함)
