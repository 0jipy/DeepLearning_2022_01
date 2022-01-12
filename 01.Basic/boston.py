# 입력값으로 0에서 50사이의 정수값을 받아서 
# 실제 가격과 보스톤 주택가격의 실제값과 예측값을 보여주는 프로그램 

# import part
import numpy as np 
import pandas as pd
import warnings
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 상수값 설정 등 변수 초기화
seed = 2022
warnings.filterwarnings('ignore')
np.random.seed(seed)
tf.random.set_seed(seed)  
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size = 0.1, random_state=seed
)

# 메인 모델 만들기 , 메인모델을 만들것이 아니라 미리 저장해두고 가져올것.
# 학습한 모델 자체를 save해서 이용할것. 미리학습모델 저장. 사용자 요청으로 바로 불러 사용하는 것.
model = Sequential([
    Dense(30, input_dim=13, activation='relu'),
    Dense(12, activation='relu'),
    Dense(1)    # 회귀라서 활성화 함수 없어
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_split=0.1, epochs=500, batch_size=60, verbose=0)


# 1. 입력값 받기
index = int(input('0 ~ 50 정수값을 입력하세요.> '))
test = X_test[index].reshape(1,-1)
pred_value = model.predict(test)


# 최종 결과 출력
print(f'실제값: {y_test[index]}, 예측값 : {pred_value[0,0]:.2f}')