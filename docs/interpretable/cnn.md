---
layout: default
title: Convolutional Neural Networks
parent: Interpretable AI
nav_order: 1
---

# Convolutional Neural Networks 이해하기

컨볼루션 뉴럴넷(Convolutilal Neural Networks)은 컴퓨터 비전 분야에서 제안된 이후 자연어 처리에서도 널리 사용되고 있는 모델입니다. 높은 성능과 빠른 계산 속도 덕분인데요. 하지만 그 인기에 비해 내부 작동 원리를 잘 몰랐던 것도 사실입니다. 이와 관련해 Yoav Goldberg 연구팀이 [Understanding Convolutional Neural Networks for Text Classification](https://arxiv.org/pdf/1809.08037.pdf)이라는 논문을 냈습니다. 2018 EMNLP의 `BlackBoxNLP` 워크샵에서 발표된 내용인데요. 간략하게 그의 아이디어를 살펴보겠습니다. 좀 더 자세히 보고 싶은 분은 [코드](https://github.com/sayaendo/interpreting-cnn-for-text)도 공개되어 있으니 살펴보시면 좋을 것 같습니다.
{: .fs-4 .ls-1 .code-example }



## conv filter : ngram detector

콘볼루션 필터(convolutional filter)는 문서 분류에 중요한 특정 `ngram`을 디텍트(detect)하는 역할을 수행합니다. 그도 그럴 것이 문서 분류를 위한 CNN의 입력값은 해당 문서에 속한 토큰의 벡터들(즉, 행렬)인데요. 텍스트 CNN은 이들 토큰 임베딩 행렬에 1D convolution 연산을 수행합니다. 만일 임베딩 차원수가 $d$이고 토큰 수 기준 문서 최대 길이를 $n$이라고 둔다면 토큰 임베딩 행렬은 $\mathbb{ R }^{d \times n}$이 됩니다. 이때 컨볼루션 필터의 크기는 $\mathbb{ R }^{l \times d}$가 될텐데요. 한번 컨볼루션을 수행할 때 고려 대상이 되는 단어 벡터의 수가 $l$개라는 이야기가 됩니다.  

Yoav Goldberg 연구팀은 콘볼루션 필터의 역할을 조사하기 위해 `Slot Activation Vector`라는 개념을 도입했습니다. 콘볼루션 필터는 $l$개 단어를 한꺼번에 보게 되는데요. 컨볼루션 필터를 행(row) 기준으로 슬라이싱해 작은 단위(`slot`)로 나눕니다. 이를 입력 단어 벡터와 내적(inner product)합니다.  $l$개 단어 각각에 대해 이를 모두 수행하면 $\mathbb{ R }^{l}$ 크기의 벡터가 나옵니다. 이를 `Slot Activation Vector`라고 합니다. 이렇게 슬롯 단위로 처리하는 이유는 **개별 단어**의 영향력을 자세히 알아보기 위해서입니다. 만일 1번 슬롯의 값이 가장 크다면 첫번째 단어-1번 슬롯 사이의 연관성이 해당 문서 분류에 제일 중요하다는 뜻입니다. 우선 표1을 봅시다.



<a href="https://imgur.com/wsK1ey4"><img src="https://i.imgur.com/wsK1ey4.png" title="source: imgur.com" /></a>



위의 표1은 전자제품 리뷰(Elec) 데이터셋으로 학습된 CNN 모델을 바탕으로 만든 것입니다. 표1의 0~8번 필터는 모두 3-gram을 봅니다. `slot` 역시 3개가 됩니다. 0번 필터에서 가장 높은 slot score를 가지는 단어는 `poorly`로 나타났습니다. 0번 필터는 아마도 부정적인 어구(phrase)를 잡아내기 위해 학습된 것이 아닌가 합니다. 0번 필터가 가장 높은 slot score를 내는 n-gram은 `poorly designed junk`로 역시 극성이 부정에 가깝습니다. 



## conv filter : 다양한 의미범주 캐치

다음 표2는 여러 문장을 동일한 필터에 입력한 후 피처맵(feature map vector)을 뽑아서 `Mean Shift Clustering`이라는 기법으로 군집화를 수행한 결과입니다. 같은 필터인데도 `긍정(positive)`, `부정(negative)` 극성과 관련한 slot score가 모두 높습니다. 이와 관련해 저자는 "각각의 필터는 복수의 의미 범주를 잡아낸다(each filter captures multiple semantic classes)"고 언급하고 있습니다. 다시 말해 하나의 필터가 단 하나의 의미 범주만 담당하는 게 아니라는 말입니다.



<a href="https://imgur.com/vpNSu8u"><img src="https://i.imgur.com/vpNSu8u.png" width="380px" title="source: imgur.com" /></a>





## conv filter : 특정 단어/구의 부재 확인

콘볼루션 필터는 어떤 단어/구의 존재를 체크하기도 하지만 `부재(lack of existence)`를 확인하기도 합니다. 다음 표3을 봅시다. 표3은 동일한 필터를 가지고 도출한 것으로, 우선 slot score가 높은 ngram을 뽑습니다. 이후 이 ngram 가운데 단어 벡터 기준으로 해밍 거리(hamming distance)가 작은 단어로 교체합니다. 혹은 해밍 거리가 작은 반대 극성 단어(아래 표에서 볼드 표시)로 교체합니다. 이렇게 만든 새로운 ngram을 가지고 slot score를 다시 계산합니다. 



<a href="https://imgur.com/iFlBmAj"><img src="https://i.imgur.com/iFlBmAj.png" width="380px" title="source: imgur.com" /></a>



위 표3의 첫번째 행을 봅시다. 이 콘볼루션 필터는 `pleased`로 끝나는 3gram에 높은 점수를 주기는 하지만 중간에 `not`이 포함되어 있다면 점수를 짜게 줍니다. 다시 말해 이 콘볼루션 필터는  `pleased`의 존재와 `not`의 부재를 동시에 확인하는 역할을 한다는 이야기입니다.



## max-pooling : threshold filter

콘볼루션 뉴럴네트워크는 입력 단어 행렬에 콘볼루션을 수행한 뒤 ReLU(Rectified Linear Unit)와 맥스 풀링(max-pooling)을 순차적으로 적용합니다. 맥스 풀링을 하면 가장 큰 콘볼루션 스코어를 가지는 ngram만 살아남게 됩니다. 우리가 감성분석을 수행하고 있다면 **긍/부정 예측에 도움이 되는 ngram만 남기고 나머지 ngram은 무시**하게 되는 것입니다. 이에 Yoav Goldberg 연구팀은 threshold 값을 정하는 새로운 기법을 제시하고 이보다 낮은 값을 가지는 ngram의 콘볼루션 스코어는 0으로 바꿔서 인위적으로 모델이 무시하게 만들었습니다. 표4를 봅시다.



<a href="https://imgur.com/MewLX9Q"><img src="https://i.imgur.com/MewLX9Q.png" width="300px" title="source: imgur.com" /></a>



*purity*가 0.0~0.4 정도라면 테스트 성능에 전혀 영향이 없습니다. 맥스 풀링을 수행하고 있기 때문입니다(긍/부정 예측에 도움이 안되는 ngram은 이미 무시하고 있음). 그런데 주목할 점은 *purity*가 0.75 정도가 되면 테스트 성능이 되레 오른다는 사실입니다. 저자에 따르면 *purity*가 0.75라면 44% 정도의 ngram이 인위적으로 무시된다고 합니다. 이렇게 높은 비율로 ngram을 무시한다고 해도 모델이 높은 테스트 성능을 유지할 수 있는 건, 맥스 풀링 이후 계층(feedforward neural network)이 긍/부정 예측에 도움이 되는 ngram들만으로 문서 분류를 하는 데 최적화되어 있기 때문이라고 봅니다.

이와 관련해 저자는 다음과 같이 언급하고 있습니다. 맥스 풀링은 사실상 threshold 역할을 수행한다는 것입니다. 단 최댓값을 취하기 때문에 그 threshold 값은 절대적이지 않고 유동적(soft)입니다.

- Global max-pooling indeed induces a functionality of separating imporant and not important activation signals using a latent (presumably soft) threshold.