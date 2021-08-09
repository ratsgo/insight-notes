---
layout: default
title: Semi-supervised learning
parent: Retrieval model
grand_parent: Question Answering
nav_order: 4
permalink: /docs/qa/ssl
---

# Semi-supervised learning
{: .no_toc }

준지도학습(semi-supervised learning)이란 레이블이 달려 있는 데이터와 레이블이 달려있지 않은 데이터를 동시에 사용해서 더 좋은 모델을 만드는 일련의 방법들을 가리킵니다. 이전 글에서 살펴본 Data Augmentation과 준지도학습 모두 레이블된 데이터의 의존도를 줄인다는 점에서 동일한 목적을 가지고 있으나, Data Augmentation은 기존에 없는 새로운 데이터를 자동적으로 생성하는 방법에 집중한다는 특징을 가집니다. 이 글에서는 Unsupervised Data Augmentation(UDA)와 MixMatch 등 최근 제안된 준지도학습 기법들을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Unsupervised Data Augmentation

[Xie et al.(2019)](https://arxiv.org/pdf/1904.12848)는 *Unsupervised data augmentation for consistency training(UDA)*이라는 방법론을 제시했습니다. [Consitency training](https://ratsgo.github.io/insight-notes/docs/qa/consistency)은 입력(input) 또는 은닉 상태(hidden states)에 노이즈를 추가해 학습하는 기법, [Data Augmentation](https://ratsgo.github.io/insight-notes/docs/qa/augmentation)은 원본 데이터와 같은 레이블을 가지는 새로운 데이터를 만드는 기법을 가리킵니다.

*unlabeled data*에 대해 data augmentation을 수행(unsupervised data augmentation)해 결과적으로 Consitency training이 되도록 학습한다는 취지에서 저자들이 이렇게 논문 제목을 지은 것 같습니다. UDA의 전체 도식은 그림1과 같습니다.

## **그림1** UDA
{: .no_toc .text-delta }
<img src="https://i.imgur.com/CFkkQpO.png" width="700px" title="source: imgur.com" />

UDA의 학습 과정은 다음과 같습니다. UDA가 상정하는 모델은 분류기(classifier)입니다.

1. *labeled data*에 대해 통상의 방법으로 Cross Entropy를 최소화하는 방식으로 학습합니다. 다시 말해 현 스텝의 분류 모델 $\theta$에 *labeled input*인 $\mathbf{x}$을 넣어 그 출력이 *label*인 $\mathbf{y}^{\*}$와 최대한 유사해 지도록 모델을 업데이트합니다.
2. *unlabeled data*에 대해 Consistency Loss를 최소화하는 방식으로 학습합니다. 구체적으로는 다음과 같습니다.
  - $p_{\theta}(\mathbf{y}\|\mathbf{x})$ 계산 : 현 스텝의 분류 모델 $\theta$에 *labeled input*인 $\mathbf{x}$을 넣어 범주 확률을 계산합니다.
  - $p_{\theta}(\mathbf{y}\|\mathbf{\hat{x}})$ 계산 : 현 스텝의 분류 모델 $\theta$에 *unlabeled input* $\mathbf{x}$으로부터 augmentation된 $\mathbf{\hat{x}}$을 넣어 범주 확률을 계산합니다.
  - 위의 두 확률분포 사이의 KL-Divergence를 계산합니다. 이것이 바로 Consistency Loss입니다.

UDA 기법에서 텍스트 분류기를 학습할 때 augmentation 기법으로 사용하는 것은 다음 두 가지입니다. 각각의 자세한 내용은 해당 글을 살펴보면 좋겠습니다.

- [Back translation](https://ratsgo.github.io/insight-notes/docs/qa/augmentation#back-translation)
- [TF-IDF를 활용한 단어 대치](https://ratsgo.github.io/insight-notes/docs/qa/augmentation#%ED%82%A4%EC%9B%8C%EB%93%9C%EA%B0%80-%EC%95%84%EB%8B%8C-%EB%8B%A8%EC%96%B4%EB%A5%BC-%EB%8C%80%EC%B9%98)

UDA가 노리는 효과는 다음과 같습니다.

1. **data augmentation을 통한 consitency training** : *unlabeled data*에 대해 Consistency Loss를 최소화하는 과정에서 노이즈(문장의 변형)에 강건한 모델 생산.
2. **semi-supervised learning** : Supervised Cross Entropy Loss와 Unsupervised Consistency Loss를 동시에 최소화하는 과정에서 *labeled data*의 정보가 *unlabeled data* 전파되도록 모델 업데이트.


# MixMatch

[Berthelot el al.(2019)](https://arxiv.org/pdf/1905.02249)는 MixMatch라는 준지도학습 기법을 제안했습니다. 여기에서 두 가지 개념이 중요할 것 같은데요. 다음과 같습니다.

- **Match** : *unlabeled data*를 data augmentation을 통해 '그럴싸한' label을 얻고 모델 관점에서 일관적인 예측이 나오도록 유도. consitency training 혹은 consitency regularization.
- **Mix** : *unlabeled data*와 *unlabeled data*를 섞어서(mixing) 학습. 

이 개념에 비추어 보면 MixMatch는 물론 앞서 살펴본 UDA도 *Mix-and-Match-based Semi-supervised leaning* 기법이라고 할 수 있을 것 같습니다. MixMatch의 Match 과정을 도식적으로 나타낸 그림은 다음과 같습니다.

## **그림2** MixMatch의 Match
{: .no_toc .text-delta }
<img src="https://i.imgur.com/hn9OutJ.png" width="500px" title="source: imgur.com" />

MixMatch의 학습 과정은 다음과 같습니다. 아래에서 1\~3까지가 Match, 4\~5가 Mix에 해당하는 것 같습니다. MixMatch가 상정하는 모델은 이미지 분류기입니다.

1. **data augmentation** : *labeled data*에 대해서는 1회, *unlabeled data*에 대해서는 $K$회 augmentation을 수행합니다.
2. **label guessing** : 1에서 augmentation된 *unlabeled data*를 현 스텝의 모델 $\theta$에 넣어 범주 확률을 계산합니다. 이를 $K$개 인스턴스 모두에 대해 각각 수행하고 평균을 취합니다.
3. **sharpening** : 2에서 구한 범주 확률 평균에 temperature scaling을 수행해 분포를 뾰족하게 만듭니다. temperature $t$가 0에 가까울 수록 이 분포는 원핫(one-hot) 레이블처럼 됩니다. 2, 3을 수행한 확률 분포를 *guessed label*이라고 합니다. 이렇게 만든 *guessed label*은 $K$개 인스턴스 모두에 동일하게 붙여줍니다.
4. **mixup** : 지금까지 만든 input과 label을 mixup합니다. 그 대상과 갯수는 다음과 같습니다.
  - (*augmented labeled input*, *original label*) $\times$ batch size : *labeled data* 인스턴스 하나당 1회만 augmentation
  - (*augmented unlabeled input*, *guessed label*) $\times$ batch size $\times K$ : *unlabeled data*에 대해서는 $K$회 augmentation
5. **loss** : supervised 인스턴스에 대해서는 Cross Entropy, unsupervised 인스턴스에 대해서는 L2 loss(4번)를 각각 구하고 더해줍니다. 

개인적으로는 MixMatch가 강력한 이유는 위의 과정 4번에 있는 것 같습니다. *labeled/unlabled input*, *original/guessed label*을 넘나들며 다양한 mixing이 이뤄지기 때문입니다. 레이블 정보가 *unlabeled data*에 전파되는 한편 *original/augmented sample*들이 서로 영향을 주고 받습니다. 이 과정에서 이전보다 강건한 모델이 탄생하는 것 아닌가 합니다. MixMatch가 노리는 효과를 저자들이 정리한 내용은 다음과 같습니다.

1. **Consistency Regularization** : $K$개 augmentation 샘플들이 저마다 다른 값을 지니더라도 동일한 레이블(*guessed label*)로 예측되도록 유도.
2. **Entropy Minimization** : *sharpening*을 수행함으로써 *unlabeled/unseen data*에 대해서도 컨피던스가 높도록(high-confidence, low-entropy) 유도.
3. **Traditional Regularization** : mixup을 적용해 달성.

한편 [MixText](https://arxiv.org/pdf/2004.12239)는 MixMatch의 후속 논문인데요. 텍스트 분류에 대해 MixMatch를 적용한 논문입니다. Data augmentation 기법으로 [Back translation](https://ratsgo.github.io/insight-notes/docs/qa/augmentation#back-translation)을 사용했습니다.


---


# References

- [Xie, Q., Dai, Z., Hovy, E., Luong, M. T., & Le, Q. V. (2019). Unsupervised data augmentation for consistency training. arXiv preprint arXiv:1904.12848.](https://arxiv.org/pdf/1904.12848)
- [Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., & Raffel, C. (2019). Mixmatch: A holistic approach to semi-supervised learning. arXiv preprint arXiv:1905.02249.](https://arxiv.org/pdf/1905.02249)
- [Chen, J., Yang, Z., & Yang, D. (2020). Mixtext: Linguistically-informed interpolation of hidden space for semi-supervised text classification. arXiv preprint arXiv:2004.12239.](https://arxiv.org/pdf/2004.12239)

---