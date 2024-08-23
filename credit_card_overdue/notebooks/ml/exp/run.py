import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from linear_model import LinearRegression

# 예제 데이터 생성
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# 데이터셋을 훈련 및 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화 및 학습
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
