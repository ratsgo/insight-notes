---
layout: default
title: The Bottom-up Evolution of Representations in the Transformer
parent: Transformer
nav_order: 1
---

# 밑바닥부터 이해해 보는 Transformer

[The Bottom-up Evolution of Representations in the Transformer](https://arxiv.org/pdf/1909.01380.pdf)는 트랜스포머의 내부 작동 원리를 고찰한 논문입니다. 2019년 EMNLP에 소개됐습니다. 다양한 시각에서 트랜스포머를 살펴보며 시각화가 정말 잘 되어 있어서 좋습니다. 논문도 훌륭하지만 저자가 직접 논문을 소개한 [블로그 포스팅](https://lena-voita.github.io/posts/emnlp19_evolution.html)도 참고할 만 합니다. 이 포스팅의 모든 그림은 저자 논문이나 블로그를 참조한 것입니다.
{: .fs-4 .ls-1 .code-example }



## MT, LM, MLM

트랜스포머를 학습시키는 방법으로 크게 세 가지가 있습니다. 하나는 기계 번역(Machine Translation), 언어 모델(Language Model), 마스크 언어 모델(Masked Language Model)이 바로 그것입니다. 기계 번역 모델은 소스 언어 문장을 타겟 언어 문장으로 바꾸는 과정에서 학습합니다. 언어 모델은 이전 단어들로 다음 단어를 맞추는 태스크를 수행합니다. 마스크 언어 모델은 마스크된 토큰을 맞추도록(복원하도록) 설계됐습니다. 각각 다음과 같습니다.


|Machine Translation|Language Model|Masked Language Model|
|---|---|---|
|<img src="https://i.imgur.com/1IJh2a8.png" width="200px" title="source: imgur.com">|<img src="https://i.imgur.com/nfWdsDI.png" width="200px" title="source: imgur.com">|<img src="https://i.imgur.com/hdBwRZj.png" width="200px" title="source: imgur.com">|



## Mutual Information

논문 저자들은 MT, LM, MLM 각각의 학습 방식이 모델에 어떤 변화를 가져오는지 고찰합니다. 이와 관련해 저자들은 `상호정보량(Mutual Information)`이라는 지표를 사용합니다. 상호정보량이란 두 확률변수가 서로 어떤 관계를 가지고 있는지 수치화한 것입니다. 두 확률변수가 완전히 독립인 경우(`사건 A가 일어나는 것이 사건 B가 일어날 확률에 전혀 영향을 주지 않고, 그 역도 마찬가지`) 그 값은 0이 되고, 둘이 서로 밀접한 관련(`사건 A가 일어날수록 B가 일어날 확률이 높아진다`)이 있을 경우 커지고, 역의 방향으로 관련이 있을 경우(`사건 A가 일어날수록 B가 일어날 확률이 낮아진다`) 값이 작아집니다.

저자들은 다음 그림과 같이 입력 토큰 벡터들($\mathbf{X}$)과 트랜스포머 레이어별 출력 벡터들($\mathbf{\hat{X}}$) 사이의 상호정보량을 계산했습니다. 물론 트랜스포머 같이 방대한 네트워크에서 $\mathbf{X}$와 $\mathbf{\hat{X}}$의 분포와 상호정보량을 구하는 건 매우 어렵습니다. 이 때문에 저자들은 근사치를 구했다고 설명하고 있습니다(자세한 내용은 논문 참고).



<img src="https://i.imgur.com/cV1JnGR.png" width="300px" title="source: imgur.com"/>

 

언어모델(LM)의 경우 레이어가 거듭될 수록 $\mathbf{X}$와 $\mathbf{\hat{X}}$ 간 상호정보량이 떨어지고 있음을 확인할 수 있습니다. 다시 말해 높은 레이어일 수록 $\mathbf{\hat{X}}$가 $\mathbf{X}$와 관련이 없어진다, 혹은 $\mathbf{X}$의 정보를 잊는다(forget)고 해석할 수 있겠습니다. 언어 모델은 이전 토큰 정보를 입력 받아 다음 토큰을 예측해야하기 때문에 높은 레이어에서는 입력 토큰과는 관계 없는 정보를 생성해야 할 것입니다. 이 점을 고려하면 당연한 결과라고 해석할 수 있겠습니다.

마스크 언어 모델(MLM)의 경우 상호정보량이 떨어졌다가 다시 복원되고 있는 점을 볼 수 있습니다. 마스크 토큰을 원래 토큰으로 복원해야 하기 때문에 이 같은 결과가 나오는 것으로 풀이됩니다. 저자들은 MLM에서 상호정보량이 떨어지는 구간을 `context encoding`, 다시 오르는 과정을 `token reconstruction`이라고 설명하고 있습니다.





## Distance between tasks

저자들은 태스크별 $\mathbf{\hat{X}}$의 거리 역시 측정해 봤습니다. 거리 지표는 `Projection Weighted Canonical Correlation Analysis(PWCCA)`라는 걸 사용했습니다. 다음 그림에서 점선 지표들은 같은 태스크로 학습했지만 랜덤 초기화 지점이 다른 두 모델 간 거리를 나타냅니다. 같은 태스크를 수행했기 때문에 태스크가 다른 모델과 비교해보면 거리가 가까운 걸 확인할 수 있습니다. 당연한 결과입니다.



<img src="https://i.imgur.com/ZLmTz9Y.png" width="400px" title="source: imgur.com" />



위의 그래프를 보면 두 가지 사실을 확인할 수 있습니다.

- 기계 번역(MT)과 마스크 언어 모델(MLM)이 가깝다.
- 언어 모델(LM)과 기계 번역(MT)이 멀다.

그 이유로 저자들은 두 가지 점을 언급했습니다. 첫째 언어 모델(LM)은 예측시 이전 컨텍스트 정보를 쓰지만, 기계 번역(MT)과 마스크 언어 모델(MLM)은 문장 전체 정보(MT=소스 문장, MLM=입력 문장)를 활용합니다. 둘째 기계 번역(MT)과 마스크 언어 모델(MLM)은 토큰을 예측하거나 번역하기 위해 개별 토큰에 집중하는 반면, 언어 모델(LM)은 다음 토큰을 예측하는 데 필요한 representation을 생성하는 데 강조점을 둡니다.





## amount of change

저자들은 레이어별로 representation이 얼마나 변화하는지를 측정했습니다. 레이어별 토큰 벡터들 간 PWCCA 거리를 측정하는 방식으로 말이죠. 다음 그림과 같습니다.



<img src="https://i.imgur.com/zOuq2wN.png" title="source: imgur.com" />