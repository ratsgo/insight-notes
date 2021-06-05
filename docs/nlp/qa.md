---
layout: default
title: Open-Domain Question Answering
parent: Natural Language Processing
nav_order: 2
permalink: /docs/nlp/qa
---

# Open-Domain Question Answering
{: .no_toc }


오픈도메인 질의응답(Open Domain Question Answering)이란 특정 분야에 국한되지 않고 다양한 질문에 대한 답을 하는 과제입니다. BERT, GPT 등 트랜스포머(transformer) 계열 언어모델의 등장과 발전, 검색(retrieval) 기술 향상 등을 등에 업고 최근 들어 눈에 띄게 성능이 좋아지고 있습니다. [Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 오픈도메인 질의응답 모델의 성능은 Knowledge Graph 등을 활용한 방법보다, 질의(query)에 적절한 문서를 찾아(retrieve) 이를 생성 모델(generation model)에 넣어 답변을 얻어내는 방식의 성능이 더 좋다고 합니다. [Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282), [Retrieval-Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401) 등 현존 최고 성능을 내는 오픈도메인 질의응답 모델이 이 방식을 채택하고 있습니다. 이 글에서는 검색 모델과 이를 활용한 답변 모델 아키텍처 전반을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

# 검색 모델 아키텍처

TF-IDF(BM25)는 오랫동안 검색(retrival) 분야의 최강자로 군림해 왔습니다. BERT 등 대규모 말뭉치로 프리트레인한 트랜스포머 계열 모델들이 제안되면서 최근 들어서야 뉴럴넷 기반의 검색 모델이 주목을 받고 있습니다. 뉴럴넷 기반의 검색 모델의 대표적인 아키텍처는 `Bi-encoder`, `Cross-encoder`, `Poly-encoder`, `ColBERT` 네 종류가 있습니다. 각각의 특징을 살펴봅시다.


## Bi-encoder

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


## Cross-encoder

`Cross-encoder`의 구조는 그림2와 같습니다. BERT 계열의 모델이 바로 이 방식입니다. `[CLS] Query [SEP] Document [SEP]`를 모델에 입력하고 마지막 레이어 `[CLS]` 벡터를 선형변환(linear transformation) 등을 적용해 둘 사이의 적절성(relevance) 스코어를 출력으로 리턴합니다.

## **그림2** Cross-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/XPDS9Ze.png" width="300px" title="source: imgur.com" />

`Cross-encoder`는 `Bi-encoder`와 마찬가지로 포지티브 쌍(positive pair)과 네거티브 쌍(negative pair) 쌍 사이의 네거티브 로그라이클리후드(the negative log-likelihood, 아래 챕터 참조)를 최소화하는 방향으로 파인튜닝하는 방식으로 학습합니다. 또는 STS(Semantic Textual Similarity) 데이터로 지도학습(supervised learning) 파인튜닝을 하기도 합니다.

`Cross-encoder`는 모든 레이어에서 토큰 레벨 상호작용을 고려하기 때문에 검색 성능이 비교적 좋은 편입니다. 하지만 계산량이 그만큼 많아 속도가 느리고, 쿼리가 주어졌을 때 검색 대상 모든 문서에 대해 일일이 그림2 같은 계산을 수행해야 하기 때문에 문서를 미리 인코딩하거나 Faiss를 사용하는 등 인퍼런스 속도를 최적화하기 어렵습니다.


## Poly-encoder

[Poly-encoder](https://arxiv.org/pdf/1905.01969)는 그림3과 같은 구조입니다. 토큰 레벨 상호작용 고려 정도, 인퍼런스 속도와 성능이 `Bi-encoder`와 `Cross-encoder`의 중간점에 위치합니다.

## **그림3** Poly-encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SwKUrY2.png" width="300px" title="source: imgur.com" />

`Poly-encoder`는 `Cross-encoder`처럼 검색 대상 문서를 미리 인코딩할 수 없다는 단점이 있습니다. 하지만 검색 성능이 비교적 좋기 때문에, BM25나 `Bi-encoder` 같은 검색 모델로 1차로 걸러낸 문서들을 다시 랭킹하는 데 `Poly-encoder`를 쓰기도 합니다. [Shuster et al.(2020)](https://arxiv.org/pdf/2104.07567)에 따르면 `Poly-encoder`로 리랭킹했을 때 성능 개선을 확인할 수 있다고 합니다. **아울러 '현재 쿼리를 포함한 대화 이력'-'답변 후보' 등과 같이 복잡한 의미 관계에 해당하는 검색에도 자주 활용됩니다.**


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


수식2와 수식3을 보면 `ColBERT`는 쿼리와 문서에 속한 모든 토큰에 대해 상호작용을 고려합니다. 인코딩시 문서 레벨 벡터를 저장해 두는 `Bi-encoder`와 달리 토큰 레벨 벡터를 저장해두어야 하기 때문에 공간 복잡도는 약간 높은 편입니다. **하지만 `ColBERT`는 상대적으로 나은 검색 성능, 인퍼런스 속도 등으로 인해 위의 네 가지 선택지 가운데 상대적으로 경쟁력 있는 선택지라는 생각이 듭니다.**


---


# 검색 모델 학습 테크닉


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

---


# 답변 모델 아키텍처

이 챕터에서 살펴볼 [Fusion-in-Decoder(FiD)](https://arxiv.org/pdf/2007.01282)와 [Retrieval Augmented Generation(RAG)](https://arxiv.org/pdf/2005.11401)는 쿼리에 적절한 문서를 검색해 그 결과를 생성 모델 입력에 넣어서 답변을 생성하는 방식입니다. 이는 생성 모델의 고질적인 문제인 **hallucination**을 완화하는 것으로 알려져 있습니다. 다시 말해 `검색+생성` 기법이 '아무말 생성'에 가까운 생성 모델의 인퍼런스 결과를 사실에 근거한 답변으로 탈바꿈시키는 효과가 있다는 것입니다. FiD와 RAG 모델을 중심으로 살펴보겠습니다.


## Fusion-in-Decoder

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


## Retrieval Augmented Generation


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

# 질의-답변 쌍 자동으로 생성


[PAQ](https://arxiv.org/pdf/2102.07033) 저자들은 위키피디아 문서들을 가지고 자동으로 질문-답변 쌍을 만들기도 했습니다. 물론 사람이 만든 것 대비 품질은 낮겠지만, 이렇게 자동으로 만들어진 문서-질문-답변 데이터는 여러모로 쓰임이 많을거라 생각합니다.


## **그림13** Knowledge Graph vs Language Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Lt2HCVR.png" width="600px" title="source: imgur.com" />


---



## References

- [Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.](https://arxiv.org/pdf/2004.04906)
- [Khattab, O., & Zaharia, M. (2020, July). Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39-48).](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=We_ZWu3bPtsAAAAA:5-Wv7Zyu5tLcB4KFF6crgYuYbVJPD06M8DzbYux8a0L4CW8wSjiqNDPVPmFDJ-SB90pqTInUYItk)
- [Izacard, G., & Grave, E. (2020). Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282.](https://arxiv.org/pdf/2007.01282)
- [Lewis, P., Wu, Y., Liu, L., Minervini, P., Küttler, H., Piktus, A., ... & Riedel, S. (2021). Paq: 65 million probably-asked questions and what you can do with them. arXiv preprint arXiv:2102.07033.](https://arxiv.org/pdf/2102.07033)
- [Humeau, S., Shuster, K., Lachaux, M. A., & Weston, J. (2019). Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969.](https://arxiv.org/pdf/1905.01969)
- [Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval Augmentation Reduces Hallucination in Conversation. arXiv preprint arXiv:2104.07567.](https://arxiv.org/pdf/2104.07567)
- [Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401.](https://arxiv.org/pdf/2005.11401)
- [Lee, K., Chang, M. W., & Toutanova, K. (2019). Latent retrieval for weakly supervised open domain question answering. arXiv preprint arXiv:1906.00300.](https://arxiv.org/abs/1906.00300)



---