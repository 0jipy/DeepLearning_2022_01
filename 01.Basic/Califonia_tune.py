# Califonia_tune try_one
# info 는 20640 rows × 9 columns / row index 20639
# 입력으로 (~~~) 값
# 캘리포니아 주택 가격의 실제값과 예측값, 그리고 오차를 보여주는 프로그램

# import part
import sys
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# argument 정리
# #
if len(sys.argv) <= 1:
    print(sys.argv[0], len(sys.argv))
    # print('사용법: python Califonia.py test_dataset_index(0~100) 2> /dev/null')
    print('사용법: python Califonia_tune.py test_dataset_index(0~20639) 2> /dev/null') 
    sys.exit()
try:
    index = int(sys.argv[1])
except:
    print('정수를 입력하세요.')
    # print('사용법: python Califonia.py test_dataset_index(0~100) 2> /dev/null')
    print('사용법: python Califonia_tune.py test_dataset_index(0~20639) 2> /dev/null')
    sys.exit()

if index < 0 or index > 20639:
    print('0 ~ 20639 사이의 정수를 입력하세요.')
    # print('사용법: python Califonia.py test_dataset_index(0~100) 2> /dev/null')
    print('사용법: python Califonia_tune.py test_dataset_index(0~20639) 2> /dev/null')
    sys.exit()

# 상수값 설정 등 변수 초기화
seed = 2022
model_filename = 'Califonia.h5'  # filename  'Califonia.h5' 에서 'Califonia.h5'??
np.random.seed(seed)
house = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    house.data, house.target, test_size=0.1, random_state=seed
)

# 저장된 메인 모델 불러오기
model = load_model(model_filename)

test = X_test[index].reshape(1,-1)
pred_value = model.predict(test)

# 최종 결과 출력
print(f'실제값 : {y_test[index]}, 예측값 : {pred_value[0,0]:.2f}, 오차 : {y_test[index] - pred_value[0,0]:.2f}')