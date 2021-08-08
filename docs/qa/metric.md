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

metric learning이란
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Metric Learning: Concept


## **그림1** Metric Learning
{: .no_toc .text-delta }
<img src="https://i.imgur.com/q22suVg.png" width="300px" title="source: imgur.com" />


# Supervised setting

- NLI를 유사(entailment), 비유사(not entailment) 두 그룹으로 나눠 학습
- 학습데이터에 등장한 질문-답변 쌍을 positive, 랜덤으로 추출한 쌍을 negative로 학습
- 주의! NLI를 세 범주로 나누는 것은 기존 방식의 cross entropy 학습

## minimize the negative log-likelihood

위에서 언급된 검색모델들은 수식2와 같은 네거티브 로그라이클리후드(the negative log-likelihood)를 최소화하는 방식으로 학습합니다. 다시 말해 포지티브 쌍(positive pair)의 스코어는 높이고 네거티브 쌍(negative pair)의 스코어는 낮추는 것입니다.

## **수식2** the negative log-likelihood
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HNXuaXk.png" width="400px" title="source: imgur.com" />

여기에서 포지티브 쌍이란 **쿼리(query)**와 그에 대응하는 **문서(document, passage)** 쌍을 가리킵니다. 네거티브 쌍은 쿼리, 그리고 이 쿼리와 관계가 없는 문서 쌍을 의미합니다. 네거티브 샘플(문서)은 보통 전체 말뭉치에서 랜덤으로 선택합니다만 모델이 포지티브 쌍으로 헷갈려할 만한 쿼리-문서 쌍을 네거티브 쌍으로 줄 수록 검색 모델의 성능이 높아진다고 합니다. 이 같은 쌍을 **하드 네거티브(hard negative)**라고 합니다. [DPR](https://arxiv.org/pdf/2004.04906) 저자들의 경우 쿼리와 BM25 스코어가 높은데(용어가 쿼리와 많이 겹치는데) 쿼리와 관계 없는 문서를 하드 네거티브로 부여했고 실제로 검색 모델 성능이 확 높아졌다고 합니다.

## In-Batch-training

In-Batch-training은 [DPR](https://arxiv.org/pdf/2004.04906) 저자들이 제안한 기법입니다. 배치 크기가 3일 경우 포지티브 쌍 3개를 선택합니다. 그리고 이 포지티브 쌍은 배치 내에서 **서로의 네거티브 쌍**이 됩니다. 

다시 말해 `언제 배송되나요`라는 쿼리는 `오늘 출고 예정입니다`라는 문서와 포지티브 관계에 있지만 `충분합니다`, `네 정사이즈입니다`라는 문서와는 네거티브 관계를 가집니다. 마찬가지로 `재고 있나요`는 `충분합니다`와 포지티브, `오늘 출고 예정입니다`, `네 정사이즈입니다`와 네거티브 관계입니다. 이렇게 서로는 서로의 포지티브 관계이면서 네거티브 관계를 갖습니다.

## **그림6** In-Batch-training (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/T4iHP39.png" width="400px" title="source: imgur.com" />

In-Batch-training 기법이 노리는 것은 모델로 하여금 포지티브 쌍과 네거티브 쌍 사이의 관계를 다이내믹하게 볼 수 있도록 하는 것입니다. 개별 포지티브 쌍 입장에서 (배치 크기 - 1)개 쌍의 네거티브 관계가 동시에 고려됩니다.


# Self-supervised setting

## Word2Vec

## **그림6** Skip-Gram Positive sample
{: .no_toc .text-delta }
<img src="https://i.imgur.com/yDcSFhE.png" width="400px" title="source: imgur.com" />

## Wav2Vec

## **그림6** Wav2vec Positive sample
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6sMQzUY.png" width="400px" title="source: imgur.com" />


## Inverse Cloze Task

Inverse Cloze Task는 [Lee et al.(2019)](https://arxiv.org/abs/1906.00300)가 검색 기반의 오픈 도메인 질의응답 시스템(Open-Retrieval Question Answering, ORQA)을 만들 때 제안한 비지도(unsupervised) 기반 학습 방법입니다. 큰 얼개는 그림6과 같습니다.

1950년대 제안된 *Cloze Task*는 주변 문맥(context)을 보고 마스킹 처리한 단락을 맞추는 과제라고 합니다. Lee et al.(2019)이 제안한 *Inverse Cloze Task*은 *Cloze Task*의 역(inverse)입니다. 즉, 문장이 주어졌을 때 그 주변 문맥을 예측하는 것입니다.

예를 들어보겠습니다. 우리가 가진 학습데이터의 원래 단락이 다음과 같이 되어 있다고 가정하겠습니다. 각 문장 앞에 번호는 이후 설명의 편의를 위해 제가 붙인 것입니다.

```
(1) ...Zebras have four gaits: walk, trot, canter and gallop.
(2) They are generally slower than horses, but their great stamina helps them outrun predators.
(3) When chased, a zebra will zigzag from side to side...
```

위의 원본 데이터에서 랜덤으로 문장 하나를 선택해 이를 쿼리 취급(pseudo-query)합니다. (2)를 선택했다고 칩시다. 그리고 나서 나머지 문장(1, 3)을 문서 취급(pseudo-evidence)합니다. 이렇게 만든 쿼리와 문서 쌍을 포지티브 쌍으로 보고, 랜덤으로 선택한 네거티브 쌍과의 네거티브 로그라이클리후드(수식2)를 최소화하는 방향으로 검색 모델을 학습하는 것입니다. 그림6에서는 $BERT_B(0)$이 포지티브, 나머지가 네거티브 pseudo-evidence입니다.


## **그림6** Inverse Cloze Task
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9qlVVyy.png" width="400px" title="source: imgur.com" />

요컨대 *Inverse Cloze Task*는 원본 문서/단락에서 문장 하나를 랜덤으로 뽑고 해당 문장을 제외한 나머지 문서/단락을 포지티브 쌍으로 취급합니다. 대개 문서/단락은 주제, 문체, 맥락 등이 유지되기 마련이므로 ICT 방식으로 만든 가짜 레이블(pseudo-label) 역시 유효하다 할 수 있겠습니다. 원본 데이터만 있으면 ICT로 얼마든지 많은 데이터를 만들어낼 수 있어 검색 모델 성능을 제법 올릴 수 있는 것으로 알려져 있습니다.

그림6에서 우리는 원본 문서에서 pseudo-query를 완전히 제거하는 방식으로 pseudo-evidence를 만들었는데요. 이같이 문서에서 쿼리를 제거하는 방식으로만 ICT를 수행하게 되면 "토큰 중복이 쿼리-문서 관련성에 중요한 특징이다"는 사실을 모델이 배우지 못할 가능성이 높습니다. 사실 관련 있는 쿼리와 문서 사이에는 토큰이 겹칠 가능성이 꽤 있기 때문입니다(텍스트 검색 분야에서 BM25가 아직도 높은 성능을 보이는 이유입니다). 그림7은 원본 문서에서 pseudo-query를 완전히 제거(ICT masking rate=1.0)했을 때 검색 모델의 성능 하락함을 보여주고 있습니다.

## **그림7** ICT 마스킹 비율별 성능
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ufYp11w.png" width="400px" title="source: imgur.com" />


# Metric Learning 성능 높이기

- consistency training : 입력, 히든에 변화를 주어서 강건한 모델 만들기
- semi-supervised learning : 다량의 unlabeled 데이터셋을 활용해서 소량의 labeled셋 효과를 극대화 하기


---


# References

- [Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.](https://arxiv.org/pdf/2004.04906)
- [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names](https://gombru.github.io/2019/04/03/ranking_loss)
- [Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
- [Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. arXiv preprint arXiv:2006.11477.](https://arxiv.org/abs/2006.11477)
- [Lee, K., Chang, M. W., & Toutanova, K. (2019). Latent retrieval for weakly supervised open domain question answering. arXiv preprint arXiv:1906.00300.](https://arxiv.org/abs/1906.00300)


---