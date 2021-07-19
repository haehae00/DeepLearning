# DeepLearning

## [데이터 분석과 입력]

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/da785318-b2d7-48da-8bf1-dacba1d8b6c4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/da785318-b2d7-48da-8bf1-dacba1d8b6c4/Untitled.png)

- 470명의 정보 각 라인은 18개 항목으로 구분 (환자 상태 정보임)
- 18번째 정보는 수술 후 생존 결과 (1:생존, 2:사망)
- 1~17번째 항목까지를 속성 / 18번째 항목을 클래스 (정답에 해당하는)

> 딥러닝 구동을 위해 **속성**만을 뽑아 데이터셋을 만들고, 클래스를 담은 데이터셋을 **따로** 만들어야함

```python
X = Data_set[:,0:17]
Y = Data_set[:,17]
```

17번째까지 속성(X) 18번째 클래스(Y)

## [딥러닝 실행]

```python
#딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

model.add()를 사용해서 두 개의 층을 쌓아올림

Dense : '조밀하게 모여있는 집합'
