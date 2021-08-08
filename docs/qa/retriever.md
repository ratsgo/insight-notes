---
layout: default
title: Retrieval model
parent: Question Answering
nav_order: 1
has_children: true
permalink: /docs/qa/retriever
---

# 뉴럴넷 기반의 검색 모델
{: .no_toc }

TF-IDF(BM25)는 오랫동안 검색(retrival) 분야의 최강자로 군림해 왔습니다. BERT 등 대규모 말뭉치로 프리트레인한 트랜스포머 계열 모델들이 제안되면서 최근 들어서야 뉴럴넷 기반의 검색 모델이 주목을 받고 있습니다. 뉴럴넷 기반의 검색 모델의 대표적인 아키텍처는 `Bi-encoder`, `Cross-encoder`, `Poly-encoder`, `ColBERT` 네 종류가 있습니다. 각각의 특징을 살펴봅시다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Bi-encoder

`Bi-encoder`의 구성은 그림1과 같습니다. 이른바 [Dense Passage Retrieval(DPR)](https://arxiv.org/pdf/2004.04906) 모델입니다. 그 구성요소는 다음과 같습니다.

- **쿼리 인코더($\text{E}_{\text{Q}}$)** : 쿼리($q$)를 벡터로 인코딩, 프리트레인을 마친 BERT 계열 모델을 파인튜닝해서 사용.
- **문서 인코더($\text{E}_{\text{P}}$)** : 문서($p$)를 벡터로 인코딩, 프리트레인을 마친 BERT 계열 모델을 파인튜닝해서 사용.
- **유사도 계산** : 쿼리 벡터와 문서 벡터 사이의 유사도를 수식1처럼 계산합니다. 

## **그림1** Bi-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6IcBkD1.png" width="300px" title="source: imgur.com" />

## **수식1** Bi-encoder 유사도 계산
{: .no_toc .text-delta }

$$
\text{sim}(q,p)=\text{E}_{\text{Q}}(q)^{\top}\text{E}_{\text{P}}(p)
$$

`Bi-encoder`는 프리트레인을 마친 BERT 계열 모델 두 개를 포지티브 쌍(positive pair)과 네거티브 쌍(negative pair) 쌍 사이의 네거티브 로그라이클리후드(the negative log-likelihood, 아래 챕터 참조)를 최소화하는 방향으로 파인튜닝하는 방식으로 학습합니다. 

`Bi-encoder`의 장점은 검색 대상 문서를 미리 인코딩해둘 수 있다는 것입니다. 인퍼런스(inference) 단계에서 쿼리가 들어왔을 때 쿼리만 벡터화하고, 미리 인코딩해둔 문서 벡터들 사이의 유사도를 계산하고 가장 높은 유사도를 가진 문서를 검색 결과로 리턴합니다. [Faiss](https://github.com/facebookresearch/faiss) 같은 Maximum Inner Product Search 기법을 사용해 유사도 계산시 계산 속도 최적화가 가능합니다.

하지만 `Cross-encoder`, `Poly-encoder`, `ColBERT` 등과 달리 쿼리와 문서 사이의 토큰 레벨 상호작용(interaction)을 고려하지 않아 검색 성능이 비교적 낮은 편입니다.


# Cross-encoder

`Cross-encoder`의 구조는 그림2와 같습니다. BERT 계열의 모델이 바로 이 방식입니다. `[CLS] Query [SEP] Document [SEP]`를 모델에 입력하고 마지막 레이어 `[CLS]` 벡터를 선형변환(linear transformation) 등을 적용해 둘 사이의 적절성(relevance) 스코어를 출력으로 리턴합니다.

## **그림2** Cross-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/XPDS9Ze.png" width="300px" title="source: imgur.com" />

`Cross-encoder`는 `Bi-encoder`와 마찬가지로 포지티브 쌍(positive pair)과 네거티브 쌍(negative pair) 쌍 사이의 네거티브 로그라이클리후드(the negative log-likelihood, 아래 챕터 참조)를 최소화하는 방향으로 파인튜닝하는 방식으로 학습합니다. 또는 STS(Semantic Textual Similarity) 데이터로 지도학습(supervised learning) 파인튜닝을 하기도 합니다.

`Cross-encoder`는 모든 레이어에서 토큰 레벨 상호작용을 고려하기 때문에 검색 성능이 비교적 좋은 편입니다. 하지만 계산량이 그만큼 많아 속도가 느리고, 쿼리가 주어졌을 때 검색 대상 모든 문서에 대해 일일이 그림2 같은 계산을 수행해야 하기 때문에 문서를 미리 인코딩하거나 Faiss를 사용하는 등 인퍼런스 속도를 최적화하기 어렵습니다.


# Poly-encoder

[Poly-encoder](https://arxiv.org/pdf/1905.01969)는 그림3과 같은 구조입니다. 토큰 레벨 상호작용 고려 정도, 인퍼런스 속도와 성능이 `Bi-encoder`와 `Cross-encoder`의 중간점에 위치합니다.

## **그림3** Poly-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SwKUrY2.png" width="300px" title="source: imgur.com" />

`Poly-encoder`는 `Cross-encoder`처럼 검색 대상 문서를 미리 인코딩할 수 없다는 단점이 있습니다. 하지만 검색 성능이 비교적 좋기 때문에, BM25나 `Bi-encoder` 같은 검색 모델로 1차로 걸러낸 문서들을 다시 랭킹하는 데 `Poly-encoder`를 쓰기도 합니다. [Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 `Poly-encoder`로 리랭킹했을 때 성능 개선을 확인할 수 있다고 합니다. **아울러 '현재 쿼리를 포함한 대화 이력'-'답변 후보' 등과 같이 복잡한 의미 관계에 해당하는 검색에도 자주 활용됩니다.**


# ColBERT

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


수식2와 수식3을 보면 `ColBERT`는 쿼리와 문서에 속한 모든 토큰에 대해 상호작용을 고려합니다. 인코딩시 문서 레벨 벡터를 저장해 두는 `Bi-encoder`와 달리 토큰 레벨 벡터를 저장해두어야 하기 때문에 공간 복잡도는 약간 높은 편입니다. **하지만 `ColBERT`는 상대적으로 나은 검색 성능, 인퍼런스 속도 등으로 인해 위의 네 가지 선택지 가운데 상대적으로 경쟁력 있는 선택지라는 생각이 듭니다.**


---


# References

- [Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.](https://arxiv.org/pdf/2004.04906)
- [Khattab, O., & Zaharia, M. (2020, July). Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39-48).](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=We_ZWu3bPtsAAAAA:5-Wv7Zyu5tLcB4KFF6crgYuYbVJPD06M8DzbYux8a0L4CW8wSjiqNDPVPmFDJ-SB90pqTInUYItk)
- [Humeau, S., Shuster, K., Lachaux, M. A., & Weston, J. (2019). Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969.](https://arxiv.org/pdf/1905.01969)



---