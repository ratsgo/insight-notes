---
layout: default
title: Commonsense
parent: Casuality & Reasoning
nav_order: 1
permalink: /docs/causality/commonsense
---

# 딥러닝에 상식을 가르칠 수 있을까
{: .no_toc }


ELMo, BERT 등 임베딩 기법이 제안되면서 자연어 처리 모델의 성능이 비약적으로 올라갔습니다. 그런데 여전히 상식(commonsense), 인과관계(causality), 추론(inference) 등 영역은 갈 길이 멉니다. 다행인 것은 관련 연구 성과들이 조금씩 나오고 있다는 점입니다. 이번 포스트에서는 단어나 문장 수준 임베딩에 상식이 얼마나 포함되어 있는지 혹은 상식을 내재화할 수 있는지에 대한 연구들을 간략히 살펴 보도록 하겠습니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---



## Language Models as Knowledge Bases?

[Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066)는 2019년 페이스북 AI Research에서 발간한 논문으로 ELMo나 BERT 같은 문장 수준 임베딩에 상식(common sense) 내지 지식(knowledge)이 얼마나 내재해 있는지를 평가해 보고자 했습니다. 그 컨셉은 그림1과 같습니다.

그림1 상단은 기존의 지식 그래프 형태를 나타냅니다. 우선 지식 그래프(Knowledge Graph)를 메모리 형태로 저장해 둡니다. 지식 그래프는 인간의 상식을 컴퓨터가 다룰 수 있는 형태의 구조화된 데이터 베이스로 표현한 것입니다. 이후 쿼리가 들어왔을 때 지식 그래프에서 해당 노드를 찾아 가장 그럴듯한 답변을 출력합니다. 이른바 시맨틱 검색(semantic search)입니다.

그림2 하단은 언어 모델(Language Model)을 형상화한 것입니다. 우선 대규모 말뭉치(unlabeled corpus)로부터 언어 모델(ELMo) 혹은 마스크 언어 모델(Masked Language Model, BERT)을 학습합니다. 전자는 이전 단어가 주어졌을 때 다음 단어를 맞추는 과정에서, 후자는 마스킹된 단어가 어떤 단어일지 복원하는 과정에서 학습됩니다. 이들 모델은 마스킹한 문장(혹은 이전 문장)을 쿼리로 주면 해당 마스킹 단어(혹은 다음 단어)가 무엇일지 답변할 수 있습니다. 

이러한 방식으로 기존 지식 그래프처럼, **언어모델을 지식 베이스(Knowledge Base)로 활용할 수 있는지** 여부가 저자들의 주요 관심사항이 되겠습니다.


## **그림1** Knowledge Graph vs Language Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/q0mAsWn.png" width="400px" title="source: imgur.com" />


그림2는 지식 그래프 관련 말뭉치인 T-Rex, ConceptNet을 학습한 BERT-large 모델에 각각의 쿼리를 던져 답변을 얻은 결과입니다. BERT는 분명 랜덤 토큰을 마스킹하고 해당 토큰이 무엇인지 맞추는 과정에서 학습했을텐데요. 쿼리(마스크 토큰이 포함된 문장)에 대해 적절한 답변(마스크 토큰이 어떤 토큰인지 예측)을 하고 있음을 확인할 수 있습니다. 다시 말해 **BERT는 말뭉치의 어휘 구조, 문법 구조 등은 물론 지식이나 상식에 이르는 영역에 이르기까지 폭넓게 학습**을 하고 있다는 걸 방증합니다. ~~역시 사고력의 원천은 암기인가 봅니다.~~


## **그림2** BERT-large 모델에 쿼리를 던진 결과
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qQVSxEH.png" title="source: imgur.com" />


---


## Commonsense Transformers


[COMET: Commonsense Transformers for Automatic Knowledge Graph Construction](https://arxiv.org/pdf/1906.05317)의 저자들은 한발 더 나아갔습니다. 트랜스포머(Transformer)가 인간의 상식을 만들어(generate) 낼 수 있는지를 검증해 보고자 했습니다. 그림1은 언어학/심리학 전문가들이 구축해 놓은 지식 베이스(knowledge base)를 개념화한 것입니다. 

`어떤 사람이 상점에 간다(PersonX goes to the store)`라는 문장이 있다면 해당 문장의 주어(`PersonX`)가 해당 행동(`go to the store`)을 하려는 의도(`xIntent`)는 식료품 구매(`to get food`)입니다. 이 관계(relation)는 지식 베이스에 명시적으로 나타나 있기 때문에 실선으로 표시가 되어 있습니다.

점선은 지식 베이스에 명시적으로 나타나지 않은 관계를 가리킵니다. 예컨대 식료품 구매를 하려면 `PersonX`가 지갑을 지참(`bring a wallet`)하는 것이 필요(`xNeed`)합니다. 어떤 물건을 사려면 돈(지갑)이 있어야 하는 건 상식이니까요. 인간의 상식이나 지식에 기초해 작은 학습데이터로 많은 추론을 이끌어 내는 것, 이것이 어쩌면 딥러닝의 한계를 뛰어넘을 수 있는 단초가 될지 모르겠습니다.


## **그림3** Knowledge Base
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1gVJwde.png" width="400px" title="source: imgur.com" />


COMET 저자들은 트랜스포머 아키텍처를 그대로 활용해 트랜스포머로 상식을 만들어 낼 수 있는지를 확인했습니다. 우선 저자들은 트랜스포머 인코더(encoder) 입력으로 $\[ X_s, X_r \]$, 디코더(decoder) 출력으로 $\[ X_o \]$가 되도록 설계했습니다. $X_s$는 문장의 주요 부분(subject, 더 정확한 번역어는 추천해 주시면 감사하겠습니다), $X_o$는 $X_s$와 관련한 부수적인 효과(object), $X_r$은 이 둘 사이의 관계를 가리킵니다.

예컨대 $X_s$가 어떤 사람이 쇼핑몰에 간다(`PersonX goes to the mall`), $X_r$이 이 사람의 의도(`xIntent`), $X_o$가 옷 사기(`to buy clothes`)라면 트랜스포머 입/출력 구성이 수식1과 같을 겁니다. COMET 저자들은 여기에 마스크 토큰을 추가하여 입력 템플릿을 만들었습니다. 그림4와 같습니다.

## **수식1** 트랜스포머 입출력
{: .no_toc .text-delta }
$$\left[ X_s, X_r \right] \rightarrow  \left[ X_o \right]\\\left[ \text{PersonX goes to the mall}, \text{xIntent} \right] \rightarrow \left[ \text{to buy clothes} \right]$$


## **그림4** COMET 입력 개념도
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DoKKKcH.png" width="400px" title="source: imgur.com" />


그림5는 COMET이 테스트 데이터를 예측한 결과를 사람이 평가한 점수 표입니다. 기존 모델(주로 Sequence-to-Sequence 기반) 대비 그 성능이 월등함을 확인할 수 있습니다. 맨 마지막 라인의 COMET 모델은 파인튜닝 전 프리트레인 모델로 GPT를 사용했는데요(COMET 모델 아키텍처가 GPT와 완전 동일). 랜덤 가중치로 학습한 모델(COMET - pretrain)보다 14% 가량 성능이 나음을 확인할 수 있습니다. 저자들은 이를 두고 "언어에 대한 일반적인 정보(어휘, 문법 등)를 지식이나 상식으로 확장할 수 있지 않을까"라고 해석했습니다. 저도 동의하는 부분입니다.


## **그림5** COMET Performance
{: .no_toc .text-delta }
<img src="https://i.imgur.com/slAZEPM.png" title="source: imgur.com" />


---


## Do NLP Models Know Numbers?


[Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://arxiv.org/pdf/1909.07940) 저자들은 단어 혹은 문장 임베딩에 숫자의 의미가 얼마나 내포되어 있는지 확인해보고자 했습니다. 이에 제시한 것이 그림6과 같은 프로빙 모델(probing model)입니다. 

좌측은 최댓값을 맞추는 모델입니다. 우선 입력 단어(숫자를 영어로 표시)를 임베딩으로 바꿉니다. 이때 임베더는 ELMo BERT 등 문장 수준, Word2Vec GloVe 등 단어 수준 임베딩 등 다양한 것이 올 수 있습니다. 이후 양방향 LSTM(Bidirectional LSTM) 레이어에 태우고 최댓값에 해당하는 입력 단어 인덱스를 맞추는 분류 문제를 학습합니다.

가운데는 디코딩 모델입니다. 입력 단어를 임베딩으로 바꾼 뒤 멀티레이어퍼셉트론(Multilayer perceptron, MLP) 레이어에 태웁니다. 이를 타겟 숫자와의 MSE(mean squared error)를 최소화하는 회귀 문제를 학습합니다. 마지막으로 우측은 덧셈 모델인데요. 하위 레이어는 가운데 디코딩 모델과 동일하고요. 두 개 숫자를 입력 받아 타겟 숫자(덧셈)와의 MSE를 최소화하는 회귀 문제를 풉니다. 


## **그림6** numeracy probing model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/etD6pb7.png" width="500px" title="source: imgur.com" />


그림7은 그림6의 디코딩 모델(가운데)을 학습한 결과입니다. -500 이상, 500 이하 숫자들만 학습을 했는데요. 문자(character) 단위 모델들이 입력과 출력이 정확히 선형 관계를 가지고 있어 다른 모델 대비 좋은 성능을 보여주고 있습니다. 다만 실험 모델 전부 학습 대상($\[-500, 500\]$)을 제외한 구간에서는 일반화 성능이 떨어집니다.


## **그림7** experiment (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Nj64OQD.png" width="400px" title="source: imgur.com" />


그림8은 프로빙 모델, 임베딩별 성능을 보여줍니다. `Pre-trained` 모델들의 성능이 `Random Vectors`보다 훨씬 높아 이들 임베딩에 숫자의 의미가 어느 정도는 내포되어 있음을 간접적으로 확인할 수 있습니다. 

특이한 것은 `Pre-trained ELMo(프리트레인된 ELMo를 프로빙 모델 학습시 프리즈)`와 `Learned Char-CNN(프리트레인 없이 프로빙 모델과 동시에 학습)`가 상대적으로 높은 성능을 기록하고 있는 점인데요. 저자들은 그 배경으로 CNN이 지역적 피처 추출에 유리하다는 사실을 꼽았습니다. 


## **그림8** experiment (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bTEjxIl.png" title="source: imgur.com" />


한편 `BERT`는 `Learned Char-LSTM`보다 성능이 낮은데요. 저자들에 따르면 숫자 인식 문제에 있어서는 문자 단위가 서브워드(subword)보다 유리한 것 아닌가 하는 해석을 내놓고 있습니다.


---