#Cancer Patient

### [데이터 분석과 입력]

```python
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
```

딥러닝을 구동하는 데 필요한 케라스 함수 불러옴

케라스를 사용해 딥러닝 실행시킴

- 텐서플로와 케라스

    딥러닝 프로젝트를 **'여행'**으로 비유.

    텐서플로는 목적지까지 빠르게 이동시켜주는 **'비행기'**

    케라스는 비행기의 이륙 및 정확한 지점까지의 도착을 책임지는 **'파일럿'**

- Sequential

    Sequential : 딥러닝의 구조를 한 층 한 층 쉽게 쌓아올릴 수 있게 해줌

    Sequential 함수를 선언하고 나서 model.add() 함수를 사용해 필요한 층 차례로 추가함

```python
import numpy
import tensorflow as tf
```

필요한 라이브러리 불러옴

```python
# 실행할 때마다 같은 결과를 출력하기 위해 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)
```

시드 재설정(매번)을 사용할 때마다 동일한 숫자가 나옴

- random.seed

    numpy.random.seed()를 고정한 상태로 알고리즘 실행해, 난수의 생성 패턴을 동일하게 관리 →random성 제어

```python
Data_set = numpy.loadtxt("dataset/ThoraricSurgery.csv", delimiter=",")
```

Data_set이라는 임시 저장소 만들고, 넘파이 라이브러리 안에 있는 loadtxt( )라는 함수 사용 →외부 데이터셋 불러옴

- delimiter : ,로 구분해서 가져옴
![image](https://user-images.githubusercontent.com/71601986/126145573-02323636-6b11-4aed-aa9c-b9a046ebb122.png)


- 470명의 정보 각 라인은 18개 항목으로 구분 (환자 상태 정보임)
- 18번째 정보는 수술 후 생존 결과 (1:생존, 2:사망)
- 1~17번째 항목까지를 속성 / 18번째 항목을 클래스 (정답에 해당하는)

> 딥러닝 구동을 위해 **속성**만을 뽑아 데이터셋을 만들고, 클래스를 담은 데이터셋을 **따로** 만들어야함

```python
X = Data_set[:,0:17]
Y = Data_set[:,17]
```

17번째까지 속성(X) 18번째 클래스(Y)

### [딥러닝 실행]

```python
#딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

model.add()를 사용해서 두 개의 층을 쌓아올림

Dense : '조밀하게 모여있는 집합'

### [딥러닝 실행]

```python
#딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

model.add()를 사용해서 두 개의 층을 쌓아올림

Dense : '조밀하게 모여있는 집합' 

각 층이 제각각 어떤 **특성**을 가질 지 **옵션을 설정**하는 역할

```python
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)
```

compile( ) 함수를 이용해 실행시킴

- loss, optimizer, activation

### [결과 출력]

```python
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```

model.evaluate( ) 함수를 이용해 앞서 만든 딥러닝 모델이 어느 정도 정확히 예측하는지 점검

해당 예시에서 정확도 → 학습 대상인 기존 환자들의 데이터 중 일부를 랜덤 추출하여 새 환자처럼 가정하고 테스트하여 나온 결과

⇒ 신뢰할 수 있는 정확도를 위해선 테스트 셋을 미리 정하여 떼어낸 뒤 따로 저장하여 오직 이 테스트셋만으로 테스트

## [전체 코드]

```python
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

Data_set = numpy.loadtxt("dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:,0:17]
Y = Data_set[:,17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```

### [결과]

![image](https://user-images.githubusercontent.com/71601986/126146057-6e531778-d19f-4335-8e60-c10abc388a07.png)
