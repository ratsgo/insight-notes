---
layout: default
title: Metric learning 
parent: Retrieval model
grand_parent: Question Answering
nav_order: 1
permalink: /docs/qa/metric
---

# Metric learning
{: .no_toc }

뉴럴네트워크 기반 검색 모델(retrieval model)을 학습하기 위한 방법론으로 메트릭 러닝(metric learning) 기법이 널리 쓰입니다. 이 글에서는 메트릭 러닝의 개념을 검색 모델 중심으로 살펴보고, 지도학습(supervised learning) 계열 방법과 자기지도학습(self-supervised learning) 계열 방법을 차례대로 살펴봅니다. 전자는 레이블 데이터(상대적으로 소수)를, 후자는 다량의 레이블 없는 데이터를 바탕으로 학습합니다. 
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Metric Learning: Concept


**메트릭 러닝(metric learning)**이란 데이터의 표상(representation)을 학습하는 방법의 일종입니다. 두 데이터 representation 사이의 거리(distance) 혹은 유사도(similarity)를 수치화하기 위해 다양한 지표(metric)을 사용한다는 취지에서 이런 이름이 붙은 것 같습니다. 의미상 유사한 데이터는 그 representation의 거리가 가깝게(=유사도가 높게) 업데이트하는 것이 핵심 아이디어입니다.


# Loss functions

메트릭 러닝의 손실 함수(loss function)는 아래처럼 다양하게 불리고 있습니다. 용어 정리 차원에서 기록해 둡니다. 자세한 내용은 [이 글](https://gombru.github.io/2019/04/03/ranking_loss)을 참고하시면 좋을 것 같습니다.

- **랭킹 로스(ranking loss)** : 데이터 인스턴스를 줄세운다(ranking)는 취지를 강조한 용어, 정보 검색(information retreival) 분야에서 이 용어 사용.
- **컨트라스티브 로스(contrastive loss)** : 두 개 혹은 그 이상의 데이터 인스턴스를 두고 손실을 계산한다는 취지에서 붙은 용어(contrastive: 대조하는).
- 기타 : 마진 로스(margin loss), 힌지 로스(hinge loss) 등으로도 불림. 랭킹 로스든 컨트라스티브 로스든 손실을 구할 때 마진(margin) $m$을 반영하기 때문. 


## Triplet ranking loss

그림1은 랭킹 로스 가운데 널리 쓰이는 **트리플렛 랭킹 로스(triplet ranking loss)**를 도식화한 것입니다. 우선 앵커(anchor, $r_a$) 샘플을 선정합니다. 보유 데이터 가운데 하나 뽑은 결과입니다. 그림1에서 포지티브(positive, $r_p$)는 앵커와 유사한 혹은 관련성이 높은 샘플을 가리킵니다. 네거티브(negative, $r_n$)는 앵커와 유사하지 않은 혹은 관련성이 없는 샘플을 의미합니다.

## **그림1** triplet ranking loss
{: .no_toc .text-delta }
<img src="https://i.imgur.com/q22suVg.png" width="500px" title="source: imgur.com" />

## **수식1** triplet ranking loss
{: .no_toc .text-delta }

$$
L(r_a, r_p, r_n) = \max(0, m+\text{d}(r_a, r_p) - \text{d}(r_a, r_n))
$$

트리플렛 랭킹 로스의 계산 과정과 지향하는 바를 직관적으로 설명하면 이렇습니다. 

1. 앵커 샘플을 모델(그림1에서 CNN이라고 적었지만 그 어떤 딥러닝 모델도 가능함)에 넣어 모델 중간 혹은 최종 출력 결과를 가지고 앵커 샘플의 representation으로 삼음.
2. 포지티브, 네거티브 샘플 역시 같은 방식으로 representation을 만듬.
3. 앵커-포지티브, 앵커-네거티브의 거리 혹은 유사도를 계산함.
4. 앵커-포지티브는 거리가 가깝게(=유사도가 높게), 앵커-네거티브는 거리가 멀게(=유사도가 낮게) representation과 모델을 업데이트.
5. 앵커-포지티브, 앵커-네거티브 업데이트는 한 스텝에서 동시에 이루어짐.

한편 수식1에서 $m$은 마진(margin)입니다. 마진의 효과는 다음과 같습니다.

- 포지티브 쌍과 네거티브 쌍 사이의 거리를 최소 $m$ 이상으로 벌려서 구분이 잘 되도록 합니다.
- 특정 포지티브 쌍과 네거티브 쌍 사이의 거리가 $m$ 이상일 때 손실을 0으로 무시함으로써 모델로 하여금 다른 쌍 사이의 차이를 벌리는 데 집중하도록 유도해 성능을 개선합니다.



## Negative log-likelihood

일반적인 딥러닝 모델 학습에 쓰이는 네거티브 로그라이클리후드(the negative log-likelihood)는 엄밀히 말해 랭킹 로스는 아닙니다만, **샘플 쌍 사이의 유사도에 적용할 경우** 이 역시 메트릭 러닝에 사용될 수 있습니다. 수식2와 같이 모델의 최종 출력인 유사도 벡터에 네거티브 로그라이클리후드를 계산하고 이를 최소화하면 포지티브 쌍(positive pair)의 유사도는 높이고 네거티브 쌍(negative pair)의 유사도는 낮추게 됩니다.

## **수식2** the negative log-likelihood
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HNXuaXk.png" width="400px" title="source: imgur.com" />

수식1에서 로그라이클리후드 계산 대상이 되는 모델 최종 출력($p$)이 로짓(logit)이 아닌 유사도(similarity)라는 점에 주목할 필요가 있습니다. 로짓은 범주 확률에 관계가 깊고 유사도는 샘플 쌍 사이에 코사인 유사도 혹은 내적(inner product)로 계산됩니다.


# Training technics

메트릭 러닝을 잘하기 위해서는 다양한 테크닉이 필요합니다. 널리 쓰이는 방식으로는 In-batch-training, Negative Sampling 등이 있습니다. 차례대로 살펴봅니다. 

## In-batch-training

In-batch-training은 [Dense Passage Retrieval(DPR)](https://arxiv.org/pdf/2004.04906) 저자들이 사용한 기법입니다. 배치 크기가 3일 경우 포지티브 쌍 3개를 선택합니다. 그리고 이 포지티브 쌍은 배치 내에서 **서로의 네거티브 쌍**이 됩니다. 

그림2를 보면 `언제 배송되나요`라는 쿼리는 `오늘 출고 예정입니다`라는 문서와 포지티브 관계에 있지만 `충분합니다`, `네 정사이즈입니다`라는 문서와는 네거티브 관계를 가집니다. 마찬가지로 `재고 있나요`는 `충분합니다`와 포지티브, `오늘 출고 예정입니다`, `네 정사이즈입니다`와 네거티브 관계입니다. 이렇게 서로는 서로의 포지티브 관계이면서 네거티브 관계를 갖습니다.

## **그림2** In-Batch-training
{: .no_toc .text-delta }
<img src="https://i.imgur.com/T4iHP39.png" width="400px" title="source: imgur.com" />

In-Batch-training 기법이 노리는 것은 모델로 하여금 포지티브 쌍과 네거티브 쌍 사이의 관계를 다이내믹하게 볼 수 있도록 하는 것입니다. 개별 포지티브 쌍 입장에서 (배치 크기 - 1)개 쌍의 네거티브 관계가 동시에 고려됩니다.


## Negative Sampling

네거티브 쌍을 어떻게 만드느냐가 모델 성능에 큰 영향을 끼칩니다. 보통 네거티브 샘플은 데이터 전체에서 랜덤으로 선택합니다. 하지만 모델이 포지티브 쌍으로 헷갈려할 만한 샘플을 네거티브로 줄 수록 모델 성능이 높아지는 경향이 있습니다. 이 같은 쌍을 **하드 네거티브(hard negative)**라고 합니다. [DPR](https://arxiv.org/pdf/2004.04906) 저자들의 경우 쿼리와 BM25 스코어가 높은데(용어가 쿼리와 많이 겹치는데) 쿼리와 관계 없는 문서를 하드 네거티브로 부여했고 실제로 검색 모델 성능이 확 높아졌다고 합니다.


# Supervised setting

메트릭 러닝을 지도학습(supervised learning) 방식으로 학습할 수 있습니다. 여기에서 지도학습이라고 함은 포지티브 쌍과 네거티브 쌍이 레이블 형태로 이미 주어져 있다는 뜻입니다. 예컨대 Natural Language Inference(NLI) 데이터를 가지고 메트릭 러닝을 한다고 가정해 봅시다. 그렇다면 NLI의 데이터는 다음과 같이 메트릭 러닝 예제들로 바꿀 수 있습니다.

- **positive pair** : 레이블이 entailment인 premise, hypothesis
- **negative pair** : 레이블이 contradiction 혹은 neutral인 premise, hypothesis

**주의!!!**
우리가 [Cross-Encoder](http://ratsgo.github.io/insight-notes/docs/qa/retriever#cross-encoder) 모델을 가지고 파인튜닝하는 상황이라고 가정해봅시다. 동일한 NLI 데이터로 `[CLS] premise [SEP] hypothesis [SEP]`를 입력하고 마지막 레이어 CLS 벡터를 활용해 범주 확률을 만들고 여기에 네거티브 로그라이클리후드 손실 함수를 적용해 `entailment`, `contradiction`, `neutral` 세 가지 범주 가운데 하나를 맞추는 NLI 모델을 만들 수도 있습니다. 이 경우 모델은 메트릭 러닝의 목표인 '데이터 사이의 유사도 혹은 거리'를 이해하는 것이 아니라, 입력이 주어졌을 때 특정 레이블을 맞추는 분류기(classifier) 역할을 수행하는 것이 됩니다. 요컨대 입력과 손실 함수가 동일하다 하더라도 "모델의 출력이 범주 확률이냐, 유사도/거리이냐", "레이블을 범주로 주느냐, positive/negative로 주느냐"에 따라서 본질적으로 다른 모델이 탄생하게 됩니다.
{: .code-example }

NLI는 포지티브 쌍뿐 아니라 네거티브 쌍도 레이블로 주어져 있지만, 경우데 따라서는 데이터에 포지티브 쌍만 있는 경우가 있습니다. 예컨대 검색 모델 기반의 답변 모델을 만드려고 하는데, 우리가 확보한 데이터는 사용자와 AI의 대화 이력뿐이라고 가정해 봅시다. 그렇다면 이 데이터를 다음과 같이 메트릭 러닝 예제들로 바꿀 수 있습니다.

- **positive pair** : 쿼리, 쿼리 직전의 대화 이력들
- **negative pair** : 쿼리, 전체 세션 가운데 랜덤으로 뽑은 대화 이력들

네거티브 쌍이 레이블로 주어지지 않은 케이스라면 앞선 네거티브 샘플링을 수행해주어야 합니다. 얼마나 잘 뽑느냐에 따라 검색 모델 성능이 확 달라지기 때문에 고민을 많이 해야 합니다.


# Self-supervised setting

자기지도학습(self-supervised learning) 세팅은 네거티브 쌍은 물론 포지티브 쌍도 주어지지 않은 경우에 수행합니다. 네거티브 샘플을 가장 나이브하게 뽑는 건 전체 데이터에서 포지티브 관계가 아닌 인스턴스를 랜덤으로 뽑으면 됩니다만, 어디까지나 포지티브 쌍이 주어져 있을 때 네거티브를 뽑을 수 있습니다. 여기서는 각 기법별로 포지티브 쌍을 어떻게 구성하는지 살펴보겠습니다.

## Word2Vec

[Word2Vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) 저자들은 Skip-Gram 모델을 만들 때 포지티브 샘플을 다음과 같이 뽑았습니다. 우선 말뭉치에서 단어를 랜덤으로 선택합니다(그림3에서는 `거기`). 그리고 해당 단어 주변에 등장한 단어들을 포지티브 샘플 취급합니다.

## **그림3** Skip-Gram Positive sample
{: .no_toc .text-delta }
<img src="https://i.imgur.com/yDcSFhE.png" width="400px" title="source: imgur.com" />

그림3에서 뽑힌 포지티브 쌍은 다음과 같습니다. 분포 가정(distributional hypothesis)에 따르면 자연어의 의미는 문맥(context)에서 드러나기 때문에 이러한 샘플링 전략을 사용한 것 같습니다.

- 거기-카페
- 거기-갔었어
- 거기-사람
- 거기-많더라


## Wav2Vec

[Wav2Vec](https://arxiv.org/abs/2006.11477) 역시 Word2Vec과 유사합니다. 계산 대상 이번 프레임 주변 프레임을 포지티브 샘플 취급합니다. 음성 인식에서는 보통 밀리세컨드(ms) 단위의 굉장히 짧은 시간으로 프레임을 나누는데요. 실제로 특정 프레임 주변의 프레임들은 물리적 특성이 비슷할 가능성이 높기 때문에 이같은 샘플링은 합리적인 방식이라는 생각이 듭니다.

## **그림4** Wav2vec Positive sample
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6sMQzUY.png" width="400px" title="source: imgur.com" />

## TF-IDF 활용하기

TF-IDF로 포지티브 샘플을 만들 수도 있습니다. [DPR](https://arxiv.org/pdf/2004.04906) 저자들은 다음과 같이 포지티브 쌍을 구성했습니다.

- 질문(question)은 이미 존재하는 상태이다.
- 다량의 문서(document) 역시 존재한다. 예: 위키피디아
- 질문, 그리고 질문과 BM25가 가장 높은 단락(paragraph)를 포지티브 쌍으로 취급한다. 


## Inverse Cloze Task

Inverse Cloze Task는 [Lee et al.(2019)](https://arxiv.org/abs/1906.00300)가 검색 기반의 오픈 도메인 질의응답 시스템(Open-Retrieval Question Answering, ORQA)을 만들 때 제안한 자기지도(self-supervised) 기반 학습 방법입니다. 큰 얼개는 그림4와 같습니다.

1950년대 제안된 *Cloze Task*는 주변 문맥(context)을 보고 마스킹 처리한 단락을 맞추는 과제라고 합니다. Lee et al.(2019)이 제안한 *Inverse Cloze Task*은 *Cloze Task*의 역(inverse)입니다. 즉, 문장이 주어졌을 때 그 주변 문맥을 예측하는 것입니다.

예를 들어보겠습니다. 우리가 가진 학습데이터의 원래 단락이 다음과 같이 되어 있다고 가정하겠습니다. 각 문장 앞에 번호는 이후 설명의 편의를 위해 제가 붙인 것입니다.

```
(1) ...Zebras have four gaits: walk, trot, canter and gallop.
(2) They are generally slower than horses, but their great stamina helps them outrun predators.
(3) When chased, a zebra will zigzag from side to side...
```

위의 원본 데이터에서 랜덤으로 문장 하나를 선택해 이를 쿼리 취급(pseudo-query)합니다. (2)를 선택했다고 칩시다. 그리고 나서 나머지 문장(1, 3)을 문서 취급(pseudo-evidence)합니다. 이렇게 만든 쿼리와 문서 쌍을 포지티브 쌍으로 보고, 랜덤으로 선택한 네거티브 쌍과의 네거티브 로그라이클리후드(수식2)를 최소화하는 방향으로 검색 모델을 학습하는 것입니다. 그림6에서는 $BERT_B(0)$이 포지티브, 나머지가 네거티브 pseudo-evidence입니다.


## **그림4** Inverse Cloze Task
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9qlVVyy.png" width="400px" title="source: imgur.com" />

요컨대 *Inverse Cloze Task*는 원본 문서/단락에서 문장 하나를 랜덤으로 뽑고 해당 문장을 제외한 나머지 문서/단락을 포지티브 쌍으로 취급합니다. 대개 문서/단락은 주제, 문체, 맥락 등이 유지되기 마련이므로 ICT 방식으로 만든 가짜 레이블(pseudo-label) 역시 유효하다 할 수 있겠습니다. 원본 데이터만 있으면 ICT로 얼마든지 많은 데이터를 만들어낼 수 있어 검색 모델 성능을 제법 올릴 수 있는 것으로 알려져 있습니다.

그림4에서 우리는 원본 문서에서 pseudo-query를 완전히 제거하는 방식으로 pseudo-evidence를 만들었는데요. 이같이 문서에서 쿼리를 제거하는 방식으로만 ICT를 수행하게 되면 "토큰 중복이 쿼리-문서 관련성에 중요한 특징이다"는 사실을 모델이 배우지 못할 가능성이 높습니다. 사실 관련 있는 쿼리와 문서 사이에는 토큰이 겹칠 가능성이 꽤 있기 때문입니다(텍스트 검색 분야에서 BM25가 아직도 높은 성능을 보이는 이유입니다). 그림5는 원본 문서에서 pseudo-query를 완전히 제거(ICT masking rate=1.0)했을 때 검색 모델의 성능 하락함을 보여주고 있습니다.

## **그림5** ICT 마스킹 비율별 성능
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ufYp11w.png" width="400px" title="source: imgur.com" />


# Metric Learning 성능 높이기

메트릭 러닝의 성능을 높이기 위해서는 다양한 기법을 활용할 수 있습니다. 메트릭 러닝과 직접 관련은 없지만, 성능 개선과 관련해 좀 더 확장해볼 수 있는 개념으로 다음과 같은 접근이 있습니다. 각 장에서 살펴봅니다.

- **[Consistency training](http://ratsgo.github.io/insight-notes/docs/qa/consistency)** : 입력(input), 히든 스테이트(hidden state)에 노이즈 등 작은 변화를 주어서 강건한 모델 만들기
- **[Data augmentation](http://ratsgo.github.io/insight-notes/docs/qa/augmentation)** : 레이블이 존재하는 데이터에 변화를 주어서 원본 데이터와 같은 레이블을 가지는 새로운 데이터를 만들어 데이터를 불리고 이를 통해 모델 성능 증대 도모하기
- **[Semi-supervised learning](http://ratsgo.github.io/insight-notes/docs/qa/ssl)** : 다량의 unlabeled dataset을 활용해서 소량의 labeled dataset 효과를 극대화 하기


---


# References

- [Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.](https://arxiv.org/pdf/2004.04906)
- [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names](https://gombru.github.io/2019/04/03/ranking_loss)
- [Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
- [Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. arXiv preprint arXiv:2006.11477.](https://arxiv.org/abs/2006.11477)
- [Lee, K., Chang, M. W., & Toutanova, K. (2019). Latent retrieval for weakly supervised open domain question answering. arXiv preprint arXiv:1906.00300.](https://arxiv.org/abs/1906.00300)


---