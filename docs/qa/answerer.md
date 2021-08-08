---
layout: default
title: Answer model
parent: Question Answering
nav_order: 2
permalink: /docs/qa/answerer
---

# 답변 생성 모델
{: .no_toc }

[Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 오픈도메인 질의응답 모델의 성능은 Knowledge Graph 등을 활용한 방법보다, 질의(query)에 적절한 문서를 찾아(retrieve) 이를 생성 모델(generation model)에 넣어 답변을 얻어내는 방식의 성능이 더 좋다고 합니다. [Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282), [Retrieval-Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401) 등 현존 최고 성능을 내는 오픈도메인 질의응답 모델이 이 방식을 채택하고 있습니다.
[Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282)와 [Retrieval Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401)는 쿼리에 적절한 문서를 검색해 그 결과를 생성 모델 입력에 넣어서 답변을 생성합니다. 이는 생성 모델의 고질적인 문제인 **hallucination**을 완화하는 것으로 알려져 있습니다. 다시 말해 `검색+생성` 기법이 '아무말 생성'에 가까운 생성 모델의 인퍼런스 결과를 사실에 근거한 답변으로 탈바꿈시키는 효과가 있다는 것입니다. 이 글에서는 FiD와 RAG 모델을 중심으로 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Fusion-in-Decoder

오픈 도메인 질의 응답 시스템을 생성 모델(generative model)로 구축할 수는 있습니다. [생성 모델 안에 다양한 지식이 내재](http://localhost:4000/insight-notes/docs/nlp/commonsense)해 있음이 여러 연구를 통해 입증된 바 있기도 합니다. 하지만 생성 모델에 세상의 모든 지식을 다 넣을 수 없을 뿐더러 설령 가능하다 하더라도 원하는 성능을 내려면 모델 크기가 기하급수적으로 커져야 할 겁니다.

[Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282)는 이같은 문제를 보완하기 위해 생성 모델 입력에 검색 결과를 넣어 활용합니다. 그 도식은 그림8과 같습니다. 우선 쿼리(`Where was Alan Turing born?`)에 적절한 문서를 검색 모델을 통해 검색합니다. $N$개 검색 결과를 생성 모델 인코더에 넣어주고, 생성모델 디코더에서 답변(`Maida Vale, London`)을 생성하는 방식입니다.


## **그림8** Fusion-In-Decoder (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tQQvl9y.png" width="300px" title="source: imgur.com" />

FiD 저자들은 검색 모델은 BM25, [DPR](https://arxiv.org/pdf/2004.04906)을 활용했습니다. 생성 모델로는 T5 같은 트랜스포머(transformer) 계열 Sequence-to-Sequence 모델을 썼습니다. 그림8 예시에서 검색 모델이 쿼리와 관련성이 가장 높다고 판단한 문서를 가지고 FiD 생성 모델 인코더 입력 값을 만들면 다음과 같습니다.

- question: Where was Alan Turing born? context: Alan Turing was a British computer scientist. Born in Maida Vale, London.

FiD의 답변 생성 방식은 그림9와 같습니다. 쿼리와의 관련성이 높은 순서대로 $N$개 입력을 만들고 각각 인코더에 태워서 벡터로 인코딩합니다. 이 $N$개의 인코딩 벡터들을 합쳐(concatenate) 디코더에 넣고 답변을 생성하는 방식입니다.

## **그림9** Fusion-In-Decoder (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1ScrHuN.png" width="500px" title="source: imgur.com" />

FiD 저자들은 프리트레인을 마친 T5 모델을 질의-답변 쌍 데이터로 파인튜닝해서 모델을 구축했습니다. 검색 모델(BM25, DPR)은 파인튜닝하지 않았다고 합니다.


# Retrieval Augmented Generation


[Retrieval Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401)는 FiD와 마찬가지로 답변을 생성할 때 검색 결과를 활용합니다. 하지만 검색 모델을 파인튜닝하지 않는 FiD와 달리 RAG는 검색 모델과 생성 모델 모두 학습한다는 차이가 있습니다. 그 전체적인 아키텍처는 그림10과 같으며 구성요소는 다음과 같습니다.

- **검색 모델(retriever)** $P_{\eta}(z\|x)$ : 쿼리 $x$가 주어졌을 때 관련성 있는 문서 $z$의 분포(상위 $K$개).
- **생성 모델(generater)** $P_{\theta}(y_i\|x, z, y_{1:i-1})$ : 쿼리 $x$, 검색된 문서 $z$, $i-1$번째까지의 답변 토큰이 주어졌을 때 $i$번째 답변 토큰의 분포.


## **그림10** RAG
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Ze6ulwi.png" width="600px" title="source: imgur.com" />

RAG 저자들은 크게 두 가지 변형을 만들었습니다. 첫번째는 `RAG-Sequence`입니다. 수식4와 같습니다.


## **수식4** RAG-Sequence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ipwBgjx.png" width="600px" title="source: imgur.com" />

1. 검색 모델로 쿼리와 관련 있는 $K$개 문서를 찾는다.
2. $K$개 문서 각각에 대해 정답 시퀀스 $y$를 끝까지 생성한다.
3. 2에서 구한 확률 분포 시퀀스를 합친다(sequence-level marginalize).

`RAG-Sequence`을 도식화한 그림은 그림11입니다. 그림11에서 맨앞 회색 네모는 Top1 검색 문서($z_1$)이 고정되었을 때 생성 모델이 계산한 다음 토큰 확률 분포 시퀀스(정답 시퀀스 길이 $N \times$ 어휘 집합 크기 $\|V\|$)입니다. 두번째 보라색 네모는 Top2 검색 문서($z_2$)가 고정되었을 때 나온 다음 토큰 확률 분포 시퀀스입니다. 이런 확률 분포 행렬(matrix)가 $K$개만큼 만들어진 셈입니다. 이후 이 행렬들을 행(row) 단위로 합칩니다(각 행렬의 $i$번째 행들을 element-wise sum한 뒤 합이 1이 되도록 normalize). 

## **그림11** RAG-Sequence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qfflEGL.png" width="200px" title="source: imgur.com" />


두번째는 `RAG-Token`입니다. 수식5와 같습니다.

## **수식5** RAG-Token
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9W30lew.png" width="450px" title="source: imgur.com" />

1. 검색 모델로 쿼리와 관련 있는 $K$개 문서를 찾는다.
2. $i$번째 토큰을 생성할 때 $K$개 문서 각각에 대해 구한다.
3. 2에서 구한 확률 분포 시퀀스를 각 time-step마다 합친다(token-level marginalize).
4. 이를 $N$번 반복한다.

`RAG-Token`을 도식화한 그림은 그림12입니다. 그림12에서 최상단에 위치한 회색 네모는 Top1 검색 문서($z_1$)이 주어졌을 때 첫번째 토큰 확률 분포입니다(차원수는 어휘 집합 크기 $\|V\|$). 최상단 보라색 네모는 Top2 문서($z_2$)가 주어졌을 때 첫번째 토큰 확률 분포입니다. 이를 $K$개 문서 모두에 대해 구하고, 각 확률 분포를 합칩니다(marginalize, 즉 element-wise sum한 뒤 합이 1이 되도록 normalize). 그리고 이 모든 과정을 $N$번 반복합니다.


## **그림12** RAG-Token
{: .no_toc .text-delta }
<img src="https://i.imgur.com/FVVB2CJ.png" width="200px" title="source: imgur.com" />

만약 시퀀스 길이 $N$이 1이라면 `RAG-Sequence`와 `RAG-Token`은 동치(equivalent)입니다.

RAG 저자들은 검색 모델로 [DPR](https://arxiv.org/pdf/2004.04906), 생성 모델로 BART를 사용했습니다. 생성 모델이 생성한 결과와 정답 답변 토큰 시퀀스 사이의 네거티브 로그라이클리후드를 최소화하는 방식으로 검색 모델, 생성 모델을 동시에 학습했습니다. 이 때 학습 대상은 DPR의 쿼리 인코더, BART입니다. 다시 말해 DPR의 문서 인코더는 고정해 두었습니다.

한편 [Shuster et al.(2021)](https://arxiv.org/pdf/2104.07567)에 따르면 RAG로 학습한 검색 모델을 FiD의 검색 모델로 사용하면 FiD의 성능을 높일 수 있다고 합니다.


---


# References

- [Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval Augmentation Reduces Hallucination in Conversation. arXiv preprint arXiv:2104.07567.](https://arxiv.org/pdf/2104.07567)
- [Izacard, G., & Grave, E. (2020). Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282.](https://arxiv.org/pdf/2007.01282)
- [Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401.](https://arxiv.org/pdf/2005.11401)


---