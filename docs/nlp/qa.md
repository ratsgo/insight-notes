---
layout: default
title: Open-Domain Question Answering
parent: Natural Language Processing
nav_order: 2
permalink: /docs/nlp/qa
---

# Open-Domain Question Answering
{: .no_toc }


오픈도메인 질의응답(Open Domain Question Answering)이란 특정 분야에 국한되지 않고 다양한 질문에 대한 답을 하는 과제입니다. BERT, GPT 등 트랜스포머(transformer) 계열 언어모델의 등장과 발전, 검색(retrieval) 기술 향상 등을 등에 업고 최근 들어 눈에 띄게 성능이 좋아지고 있습니다. [Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 오픈도메인 질의응답 모델의 성능은 Knowledge Graph 등을 활용한 방법보다, 질의(query)에 적절한 문서를 찾아(retrieve) 이를 생성 모델(generation model)에 넣어 답변을 얻어내는 방식의 성능이 더 좋다고 합니다. 이후 살펴볼 [Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282), [Retrieval-Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401) 등 현존 최고 성능을 내는 오픈도메인 질의응답 모델이 이 방식을 채택하고 있습니다. 특히 답변 생성에 기초가 되는 문서를 검색해 생성 모델에 넣어주는 이 방식은 생성 모델의 고질적인 문제인 **hallucination**을 완화하는 것으로 알려져 있습니다. 다시 말해 `검색+생성` 기법이 '아무말 생성'에 가까운 결과를 사실에 근거한 답변으로 탈바꿈시키는 효과가 있다는 것입니다. 이에 이 글에서는 검색 모델 아키텍처와 그 학습 방법, 그리고 FiD와 RAG 모델을 중심으로 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

# 검색 모델 아키텍처

TF-IDF(BM25)는 오랫동안 검색(retrival) 분야의 최강자로 군림해 왔습니다. BERT 등 대규모 말뭉치로 프리트레인한 트랜스포머 계열 모델들이 제안되면서 최근 들어서야 뉴럴넷 기반의 검색 모델이 주목을 받고 있습니다. 뉴럴넷 기반의 검색 모델의 대표적인 아키텍처는 `Bi-encoder`, `Bi-encoder`, `Cross-encoder`, `Poly-encoder`, `ColBERT` 네 종류가 있습니다. 각각의 특징을 살펴봅시다.


## Bi-encoder

`Bi-encoder`의 구성은 그림1과 같습니다. 이른바 [Dense Passage Retrieval(DPR)](https://arxiv.org/pdf/2004.04906) 모델입니다. 그 구성요소는 다음과 같습니다.

- **쿼리 인코더($\text{E}_{\text{Q}}$)** : 쿼리($q$)를 벡터로 인코딩합니다. 프리트레인을 마친 BERT 계열 모델을 파인튜닝해서 사용합니다.
- **문서 인코더($\text{E}_{\text{P}}$)** : 문서($p$)를 벡터로 인코딩합니다. 프리트레인을 마친 BERT 계열 모델을 파인튜닝해서 사용합니다.
- **유사도 계산** : 쿼리 벡터와 문서 벡터 사이의 유사도를 수식1처럼 계산합니다. 

## **그림1** Bi-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6IcBkD1.png" width="300px" title="source: imgur.com" />

## **수식1** Bi-encoder 유사도 계산
{: .no_toc .text-delta }

$$
\text{sim}(q,p)=\text{E}_{\text{Q}}(q)^{\top}\text{E}_{\text{P}}(p)
$$

`Bi-encoder`의 장점은 검색 대상 문서를 미리 인코딩해둘 수 있다는 것입니다. 인퍼런스(inference) 단계에서 쿼리가 들어왔을 때 쿼리만 벡터화하고, 미리 인코딩해둔 문서 벡터들 사이의 유사도를 계산하고 가장 높은 유사도를 가진 문서를 검색 결과로 리턴합니다. [Faiss](https://github.com/facebookresearch/faiss) 같은 Maximum Inner Product Search 기법을 사용해 유사도 계산시 계산 속도 최적화가 가능합니다.

하지만 `Cross-encoder`, `Poly-encoder`, `ColBERT` 등과 달리 쿼리와 문서 사이의 토큰 레벨 상호작용(interaction)을 고려하지 않아 검색 성능이 비교적 낮은 편입니다.


## Cross-encoder

`Cross-encoder`의 구조는 그림2와 같습니다. BERT 계열의 모델이 바로 이 방식입니다. `[CLS] Query [SEP] Document [SEP]`를 모델에 입력하고 마지막 레이어 `[CLS]` 벡터를 선형변환(linear transformation) 등을 적용해 둘 사이의 적절성(relevance) 스코어를 출력으로 리턴합니다.

## **그림2** Cross-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/XPDS9Ze.png" width="300px" title="source: imgur.com" />

`Cross-encoder`는 프리트레인을 마친 BERT 계열 모델을 포지티브 쌍(positive pair)과 네거티브 쌍(negative pair) 쌍 사이의 네거티브 로그라이클리후드(the negative log-likelihood, 아래 챕터 참조)를 최소화하는 방향으로 파인튜닝한 것입니다. 또는 STS(Semantic Textual Similarity) 데이터로 지도학습(supervised learning) 파인튜닝을 하기도 합니다.

`Cross-encoder`는 모든 레이어에서 토큰 레벨 상호작용을 고려하기 때문에 검색 성능이 비교적 좋은 편입니다. 하지만 계산량이 그만큼 많아 속도가 느리고, 쿼리가 주어졌을 때 검색 대상 모든 문서에 대해 일일이 그림2 같은 계산을 수행해야 하기 때문에 문서를 미리 인코딩하거나 Faiss를 사용하는 등 인퍼런스 속도를 최적화하기 어렵습니다.


## Poly-encoder

[Poly-encoder](https://arxiv.org/pdf/1905.01969)는 그림3과 같은 구조입니다. 토큰 레벨 상호작용 고려 정도, 인퍼런스 속도와 성능이 `Bi-encoder`와 `Cross-encoder`의 중간점에 위치합니다.

## **그림3** Poly-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SwKUrY2.png" width="300px" title="source: imgur.com" />

`Poly-encoder`는 `Cross-encoder`처럼 검색 대상 문서를 미리 인코딩할 수 없다는 단점이 있습니다. 하지만 검색 성능이 비교적 좋기 때문에, BM25나 `Bi-encoder` 같은 검색 모델로 1차로 걸러낸 문서들을 다시 랭킹하는 데 `Poly-encoder`를 쓰기도 합니다. [Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 `Poly-encoder`로 리랭킹했을 때 성능 개선을 확인할 수 있다고 합니다.


## ColBERT

[ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=We_ZWu3bPtsAAAAA:5-Wv7Zyu5tLcB4KFF6crgYuYbVJPD06M8DzbYux8a0L4CW8wSjiqNDPVPmFDJ-SB90pqTInUYItk) 구조는 그림4와 같습니다. (1) 토큰 레벨 상호작용을 고려하기 때문에 검색 성능이 상대적으로 좋고 (2) 문서를 미리 인코딩해둘 수 있고 (3) Faiss 등 라이브러리를 사용할 수 있다는 장점이 있습니다.

## **그림4** ColBERT
{: .no_toc .text-delta }
<img src="https://i.imgur.com/cvnc4WN.png" width="400px" title="source: imgur.com" />

`ColBERT`는 수식2처럼 유사도를 계산합니다. 쿼리($q$)가 2개의 단어($q_1$, $q_2$)로 구성되어 있고 문서($p$)에 3개의 단어($p_1$, $p_2$, $p_3$)가 있다면 계산 예시는 수식3과 같습니다. 

## **수식2** ColBERT 유사도 계산 (1)
{: .no_toc .text-delta }

$$
\text{sim}(q,p):=\sum_{i\in[\text{E}_{\text{Q}}(q)]}\max_{j\in[\text{E}_{\text{P}}(p)]}\text{E}_{\text{Q}}(q_i)^{\top} \text{E}_{\text{P}}(p_j)
$$


## **수식3** ColBERT 유사도 계산 (2)
{: .no_toc .text-delta }

$$
\text{sim}(q,p)=\max[\text{E}_{\text{Q}}(q_1)^{\top} \text{E}_{\text{P}}(p_1), \text{E}_{\text{Q}}(q_1)^{\top} \text{E}_{\text{P}}(p_2), \text{E}_{\text{Q}}(q_1)^{\top} \text{E}_{\text{P}}(p_3)]
\\+ \max[\text{E}_{\text{Q}}(q_2)^{\top} \text{E}_{\text{P}}(p_1), \text{E}_{\text{Q}}(q_2)^{\top} \text{E}_{\text{P}}(p_2), \text{E}_{\text{Q}}(q_2)^{\top} \text{E}_{\text{P}}(p_3)]
$$


수식2와 수식3을 보면 `ColBERT`는 쿼리와 문서에 속한 모든 토큰에 대해 상호작용을 고려합니다. `Bi-encoder`와 달리 문서 인코딩시 토큰 레벨 벡터를 저장해두어야 하기 때문에 공간 복잡도는 약간 높은 편입니다. **하지만 `ColBERT`는 상대적으로 나은 검색 성능, 인퍼런스 속도 등으로 인해 위의 네 가지 선택지 가운데 가장 나은 선택지라는 생각이 듭니다.**


---


# 검색 모델 학습 테크닉


## minimize the negative log-likelihood

위에서 언급된 검색모델들은 수식2와 같은 네거티브 로그라이클리후드(the negative log-likelihood)를 최소화하는 방식으로 학습합니다. 다시 말해 포지티브 쌍(positive pair)의 스코어는 높이고 네거티브 쌍(negative pair)의 스코어는 낮추는 것입니다.

## **수식2** the negative log-likelihood
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HNXuaXk.png" width="400px" title="source: imgur.com" />

여기에서 포지티브 쌍이란 **쿼리(query)**와 그에 대응하는 **문서(document, passage)** 쌍을 가리킵니다. 네거티브 쌍은 쿼리, 그리고 이 쿼리와 관계가 없는 문서 쌍을 의미합니다. 

보통 전체 말뭉치에서 랜덤으로 선택합니다만 모델이 포지티브 쌍으로 헷갈려할 만한 쿼리-문서 쌍을 네거티브 쌍으로 줄 수록 검색 모델의 성능이 높아진다고 합니다. 이 같은 쌍을 **하드 네거티브(hard negative)**라고 합니다.


## In-Batch-training


## **그림6** In-Batch-training (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/T4iHP39.png" width="400px" title="source: imgur.com" />


## Inverse Cloze Task

## **그림6** Inverse Cloze Task
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9qlVVyy.png" width="400px" title="source: imgur.com" />

## **그림1** ICT 마스킹 비율별 성능
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ufYp11w.png" width="400px" title="source: imgur.com" />

---


# 답변 모델 아키텍처


## Fusion-In-Decoder

## **그림5** Fusion-In-Decoder (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tQQvl9y.png" width="400px" title="source: imgur.com" />


## **그림1** Fusion-In-Decoder (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1ScrHuN.png" width="400px" title="source: imgur.com" />


## Retrieval Augmented Generation


## **그림1** RAG
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Ze6ulwi.png" width="400px" title="source: imgur.com" />

## **그림1** RAG-Sequence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qfflEGL.png" width="400px" title="source: imgur.com" />

## **그림1** RAG-Sequence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/FVVB2CJ.png" width="400px" title="source: imgur.com" />


---

# 질의-답변 쌍 자동으로 생성

## **그림1** Knowledge Graph vs Language Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Lt2HCVR.png" width="400px" title="source: imgur.com" />


---



## References

- [Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.](https://arxiv.org/pdf/2004.04906)
- [Khattab, O., & Zaharia, M. (2020, July). Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39-48).](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=We_ZWu3bPtsAAAAA:5-Wv7Zyu5tLcB4KFF6crgYuYbVJPD06M8DzbYux8a0L4CW8wSjiqNDPVPmFDJ-SB90pqTInUYItk)
- [Izacard, G., & Grave, E. (2020). Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282.](https://arxiv.org/pdf/2007.01282)
- [Lewis, P., Wu, Y., Liu, L., Minervini, P., Küttler, H., Piktus, A., ... & Riedel, S. (2021). Paq: 65 million probably-asked questions and what you can do with them. arXiv preprint arXiv:2102.07033.](https://arxiv.org/pdf/2102.07033)
- [Humeau, S., Shuster, K., Lachaux, M. A., & Weston, J. (2019). Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969.](https://arxiv.org/pdf/1905.01969)
- [Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval Augmentation Reduces Hallucination in Conversation. arXiv preprint arXiv:2104.07567.](https://arxiv.org/pdf/2104.07567)
- [Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401.](https://arxiv.org/pdf/2005.11401)




---