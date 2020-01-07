---
layout: default
title: Label Smoothing
parent: Interpretable AI
nav_order: 3
---

# Label Smoothing 이해하기

레이블 스무딩(Label Smoothing)은 데이터 정규화(regularization) 테크닉 가운데 하나로 간단한 방법이면서도 모델의 일반화 성능을 높여 주목을 받았습니다. 하지만 이 기법 역시 내부 작동 원리 등에 대해서는 거의 밝혀진 바가 없습니다. '해봤더니 그냥 잘 되더라' 정도였는데요. 제프리 힌튼 교수 연구팀이 2019 NeuraIPS에 제출한 [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629) 논문을 살펴볼까 합니다. 레이블 스무딩이 언제 잘 되고 그 이유에 대해 고찰해보는 내용이 핵심입니다. `Temperature Scaling`, `Knowledge Distillation` 등과도 비교한 점이 눈에 띄는데요. `One-hot-representation`과 `크로스 엔트로피(cross-entopy)` 일색인 딥러닝 학습 기법을 뒤집어 생각해볼 수 있는 좋은 계기가 될 거라 생각합니다.
{: .fs-4 .ls-1 .code-example }





## Label Smoothing이란

`레이블 스무딩`이란 [Szegedy et al. (2016)](https://arxiv.org/pdf/1512.00567.pdf)이 제안한 기법으로 말 그대로 레이블을 깎아서(스무딩) 모델 일반화 성능을 꾀합니다. `hard target`(one-hot-representation)을 `soft target`으로 바꾸는 것이 핵심입니다. $K$개 범주(class)에 관한 레이블 스무딩 벡터의 $k$번째 스칼라(sclar) 값은 다음 수식과 같습니다($\mathbf{y}_{k}$는 $k$번째 범주가 정답이면 1, 그렇지 않으면 0,  $\alpha$는 hyperparameter).



$$\mathbf{ y }_{ k }^\text{ LS }=\mathbf{ y }_{ k }(1-\alpha )+\alpha /K$$



예컨대 우리가 4개 범주를 분류하는 레이블을 만든다고 합시다. 기존 `hard target` 은 다음과 같을 겁니다(정답은 두번째 범주).



- $\begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}$



여기에 레이블 스무딩을 실시한 `soft target`은 다음과 같습니다($\alpha$를 0.1로 설정).



- $\begin{bmatrix} 0.025 & 0.925 & 0.025 & 0.025 \end{bmatrix}$





## LS를 시각화해서 이해하기

저자들은 로짓(logit)-소프트맥스 노드에 주목했습니다. $\mathbf{x}$는 이 노드의 입력 벡터=즉 직전 레이어의 출력 벡터(`the activations of the penultimate layer`), $\mathbf {w}_{l}$은 이 노드의 $l$번째 가중치(weight) 벡터에 해당합니다. 저자들은 후자를 템플릿(the template)이라고 부르고 있습니다.



$$\mathbf{ p }_{ k }=\frac { \exp(\mathbf{ x }^{ \top }\mathbf{w}_{k}) }{ \sum _{ l=1 }^{ K }{ \exp(\mathbf{ x }^{ \top }\mathbf{ w }_{ l }) }}$$



$k$번째 범주가 정답이라면 $\mathbf{x}^{\top} \mathbf{w}_{k}$ 내적(inner product) 값이 커져야 할 겁니다. 벡터의 내적는 코사인 유사도(cosine similarity)와 밀접한 관련이 있습니다. 이 뉴럴넷이 제대로 학습되고 있다면 $\mathbf{x}$는 정답 범주에 해당하는 템플릿($\mathbf{w}$)과 벡터공간 상 가까이에 위치하게 됩니다. 

저자들에 따르면 기존 `hard target`은 $\mathbf{x}$를 정답 템플릿과 가깝게 하는 데에만 관심을 둔다고 지적합니다. [소프트맥스-크로스 엔트로피(cross entropy)를 미분](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax)해 보면 정답 확률만을 높이는 쪽으로 그래디언트가 구해지기 때문에 그럴 것도 같습니다. 

그런데 레이블 스무딩을 실시하게 되면 양상이 조금 달라집니다. $\mathbf{x}$를 정답 템플릿과 가깝게 하는 한편, $\mathbf{x}$를 오답 템플릿과 **동일한 거리에 있도록** 멀게 만드는 효과를 낸다고 합니다. 



- *label smoothing encourages the activations of the penultimate layer to be close to the template of the correct class and equally distant to the templates of the incorrect classes*



이미 살펴봤듯 **레이블 스무딩을 실시하게 되면 오답 범주도 uniform하게 확률 값을 부여**하기 때문에 이 역시 그럴 듯한 설명입니다. 이를 살펴보기 위해 저자들은 다음과 같은 시각화 방식을 제안했습니다.



1. $K$개 범주 가운데 3개를 택한다.
2. 3개 범주에 해당하는 템플릿이 만들어내는 하이퍼플레인(hyperplane; ratsgo 註)의 정규 직교 기저(orthonormal basis)를 찾는다.
3. 정답이 3개 범주 중 하나인 다수의 $\mathbf{x}$를 2에서 찾은 기저에 사영(projection)한다.



이와 관련해서는 두 가지 의문점이 있습니다. (1) 3개의 벡터(여기서는 템플릿)가 만들어내는 하이퍼플레인의 정규 직교 기저의 최대의 수는 3(셋 모두 `선형 독립/linearly independent`일 때 성립)일 텐데요. 세 벡터가 선형 독립일 것이라는 가정조차 언급되어 있지 않습니다. (2) 아래 시각화 결과물을 보면 2개의 기저에 사영한 것을 볼 수 있는데요. 정규 직교 기저가 3개가 나왔다면 그 중 어떤 것을 취했는지 그 기준에 대한 언급이 없습니다. ~~힌튼 교수님 연구팀에서 나온 논문이니 제가 오독(誤讀)하고 있는 것이겠지요~~
{: .fs-3 .ls-1 .code-example }



다음은 위의 방식대로 시각화한 그림입니다. 첫번째 행은 `CIFAR10/AlexNet`, 두번째 행은 `CIFAR100/ResNet-56`, 세번째 행은 `ImageNet/Inception-v4`, 네번째 행 역시 `ImageNet/Inception-v4`이나 유사한 클래스만을 일부러 골라 다시 그린 것에 해당합니다. **전반적인 경향성을 보면 레이블 스무딩을 실시한 모델이 그렇지 않은 모델보다 동일한 범주끼리 잘 뭉치고 있습니다.** 

<img src="https://i.imgur.com/uA9s8eY.png" title="source: imgur.com" />



이와 관련해 저자들은 다음과 같이 언급하고 있습니다.

- *the clusters are much tighter, because label smoothing encourages that each example in training set to be equidistant from all the other class’s templates.*



## Temperature Scaling과 비교 : model calibration

Temperature Scaling은 [Guo et al. (2017)](https://arxiv.org/pdf/1706.04599.pdf)이 제안한 기법으로, 뉴럴넷이 예측 과정에서 과신(over-confident)하는 경향이 있어, 이를 완화해 일반화 성능을 높이는 기법입니다. 틀릴 때도 강한 확신으로 틀려버리니 이를 좀 막아보자는 겁니다. 학습된 뉴럴넷과 validation set 입력에 대해 다음 방식의 소프트맥스 확률을 구하고, 같은 validation set 레이블에 대해 크로스 엔트로피를 최소화하는 temperature $T$를 찾는 것이 목적입니다.



$$\mathbf{ p }_{ k }=\frac { \exp(\mathbf{ x }^{ \top }\mathbf{w}_{k}/T)}{ \sum _{ l=1 }^{ K }{ \exp(\mathbf{ x }^{ \top }\mathbf{ w }_{ l }/T) }}$$



TS를 제안한 저자들이 노리는 효과는 다음과 같습니다. 가장 이상적인 경우 모델의 출력 소프트맥스 확률의 최댓값이 그대로 accuracy가 되는 상황(아래 그림에서 점선)일 겁니다. 예컨대 모델이 어떤 입력을 받았을 때 $k$번째 범주일 확률이 제일 높다고 예측했는데, 그 값이 0.3이라면 해당 범주가 맞을 가능성(정확도) 역시 0.3으로 낮으므로 예측을 포기하는 것이 나을 겁니다. 

그러나 보통의 뉴럴넷은 정확도의 기댓값이 모델의 confidence(출력 소프트맥스 확률)보다 항상 낮은 경향(=`confidence가 accuracy보다 높은 over-confident 문제`)이 있습니다. 이에 TS 저자들은 로짓에 $T$만큼을 나누어주게 된 것입니다. 이렇게 되면 출력 소프트맥스 확률 분포가 약간 uniform(=`최고 높은 확률값이 낮아지고 낮은 값들이 높아진다`)해지게 되면서 **calibration이 됩니다**. 다시 말해 아래 그림에서 녹색 화살표 방향으로 옮겨지는 것입니다.



<img src="https://i.imgur.com/mX0mNcU.png" width="400px" title="source: imgur.com" />



**힌튼 교수 연구팀은 레이블 스무딩이 TS와 마찬가지로 calibration 효과를 볼 수 있다고 주장합니다.** TS가 로짓에 $T$를 나눠줘서 스케일하는 방식으로 calibration한다면 레이블 스무딩은 정답 레이블을 조금 깎고 오답 레이블은 조금 높이는 방식으로 calibration을 수행한다고 보는 것입니다. 다시 말해 TS는 로짓에, 레이블 스무딩은 레이블에 스무딩을 하는 셈이죠. 힌튼 연구팀이 `ResNet-56/CIFAR-100`에 대해 실험을 수행한 결과를 봅시다.



<img src="https://i.imgur.com/HfzNWKn.png" width="400px" title="source: imgur.com" />



위 그림에서 제일 아래 쪽에 있는 파란 실선은 레이블 스무딩, TS 모두 실시하지 않은 모델입니다. 위쪽에 있는 녹색 실선은 레이블 스무딩만, 파란색 선은 TS만 실시한 모델입니다. 레이블 스무딩이 TS처럼 모델의 출력을 calibration하고 있음을 간접적으로 확인할 수 있습니다.  



## Knowledge Distillation과 비교 : information erase

Knowledge Distillation(이하 KD)은 방대한 teacher network로부터 핵심적인 지식을 전수받은 경량화된 student network를 만들기 위해 제안됐습니다. 그런데 student network는 teacher network의 출력 소프트맥스 확률을 정답 삼아 학습을 하게 되는데요. student network 입장에서는 `soft target`을 가지고 배우는 셈입니다. 힌튼 연구팀은 이런 관점에서 보면 레이블 스무딩이 KD와 밀접한 관련이 있다고 보았습니다. 레이블 스무딩 역시 `soft target`이니까요.

다음은 `CIFAR-10` 데이터셋에 대해 teacher network를 `ResNet-56`, student network를 `AlexNet`로 두고 한 실험 결과입니다. 두번째 아래 그림에서 보라색 (4)에 해당하는 것이 TS만 실시한 student network의 정확도 양상, 붉은색 (3)에 해당하는 것이 레이블 스무딩만 실시한 student network의 정확도 양상입니다. 레이블 스무딩을 실시한 것이 TS보다 정확도가 훨씬 낮은 경향을 보이고 있습니다.



<img src="https://i.imgur.com/xfqgIpP.png" width="400px" title="source: imgur.com" />



힌튼 교수 연구팀은 이 원인으로 **정보 손실**을 언급하고 있습니다. 로짓 벡터에 스케일을 하는 TS와는 달리, 레이블 스무딩은 기존 `hard target`에 인위적인 조작을 가해 정보량이 줄어들 수 있습니다. 이와 관련해 저자들이 실험한 결과는 다음 그림과 같습니다. 아래 그림은 위 그림의 (1) teacher network를 대상으로 `hard target`과 teacher network의 로짓(logit) 사이의 상호정보량(mutual information)을 근사한 결과입니다. 레이블 스무딩을 적용한 모델의 상호정보량이 적습니다. 레이블 스무딩 탓에 teacher network가 받아들이는 정보가 감소했음을 방증합니다. 레이블 스무딩으로 teacher network의 일반화 성능을 높일 수 있다 하더라도, KD에 있어서는 이렇게 학습 과정에서 손실이 발생한 그대로 따라하는 student network의 성능이 떨어질 수 있음을 보여준 셈입니다.



<img src="https://i.imgur.com/jkWeLGq.png" width="400px" title="source: imgur.com" />