---
layout: default
title: Consistency training
parent: Retrieval model
grand_parent: Question Answering
nav_order: 2
permalink: /docs/qa/consistency
---

# Consistency training
{: .no_toc }

Consistency training이란 입력 또는 은닉 상태에 노이즈를 추가해 모델의 성능 높이는 기법입니다. 검색 모델 구축 관점에서 도움이 될 만한 기법을 소개합니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Concept

Consistency training이란 레이블(label)이 없는 데이터의 입력 또는 은닉 상태(hidden state)에 노이즈(noise)가 추가되어도 모델이 강건한 예측을 할 수 있도록 학습하는 방법론입니다. 다시 말해 *unlabeled data* 원본의 모델 출력과 여기에 노이즈가 추가된 데이터의 출력이 유사하도록 제약을 만드는 것입니다. 이 글에서 소개하는 Dialogue History Perturbation는 입력에, SimCSE는 은닉 상태에서 노이즈를 추가하는 기법입니다.

# Dialogue History Perturbation

[Zhou el al.(2021)](https://arxiv.org/pdf/2105.15171)은 사용자와 AI의 대화 이력 데이터에서 단어, 문장 수준의 변화를 주는 방식으로 pertubation을 주었습니다. 다음과 같습니다. 

1. Word-level pertubation
  - **word-shuffle** : 두 단어의 순서를 바꾸기
  - **reverse** : 문장 내 모든 단어를 역순으로 배치하기
  - **word-drop** : 문장 내 특정 단어를 삭제하기
  - **noun-drop** : 문장 내 특정 명사를 삭제하기
  - **verb-drop** : 문장 내 특정 동사를 삭제하기
2. Utterance-level pertubation
  - **shuf** : 두 발화의 순서를 바꾸기
  - **reverse** : 세션 내 모든 발화를 역순으로 배치하기
  - **drop** : 세션 내 특정 발화를 삭제하기
  - **truncate** : 세션을 앞뒤로 자르기
  - **repl** : 세션 내 특정 발화를 임의의 다른 발화로 대치하기

**주의!!!**
여기서 주목할 것은 Word-level pertubation은 [Data augmentation](http://ratsgo.github.io/insight-notes/docs/qa/augmentation) 파트에서 소개한 Easy Data Augmentation과 거의 유사해보이는데요. 방식은 유사하나 목적이 다릅니다. [Zhou el al.(2021)](https://arxiv.org/pdf/2105.15171)의 목적은 입력 변화에도 강건한 모델을 만들기 위한 **Consistency training**, EDA는 원본 데이터와 같은 레이블을 가지는 새로운 데이터를 만드는 **Data Augmentation**이라는 점이 다른 것 같습니다.
{: .code-example }

[Zhou el al.(2021)](https://arxiv.org/pdf/2105.15171)은 `원본 대화 이력을 입력으로 한 뒤 답변이 나타날 우도(likelihood)`가 `단어, 문장 수준의 pertubation을 준 대화 이력을 입력으로 하는 우도` 대비 크게 떨어지는 경우 높은 점수, 반대의 경우 낮은 점수를 갖도록 리워드(reward)를 설계했습니다. 그리고 목적 함수(objective function)는 답변이 나타날 우도(LM likelihood)에 이 리워드를 곱한 값이 최대화되도록 했습니다. 그림1과 같습니다.

## **그림1** Learning from Perturbations
{: .no_toc .text-delta }
<img src="https://i.imgur.com/zy1hdrK.png" width="700px" title="source: imgur.com" />

[Zhou el al.(2021)](https://arxiv.org/pdf/2105.15171)은 (1) "리워드 $\times$ LM likelihood"와 별개로 (2) Pertubation Penalty 역시 제안했는데요. Pertubation Penalty는 수식1과 같습니다. [Zhou el al.(2021)](https://arxiv.org/pdf/2105.15171)는 (1)은 최대화, (2)는 최소화하도록 고안했습니다.

## **수식1** Pertubation Penalty
{: .no_toc .text-delta }

$$
P = \max(0, m + S_\text{adv} - S_\text{orig})
$$


수식1은 **네거티브 쌍에 대응하는 pairwise ranking loss**와 깊은 관련이 있습니다. Pertubation dialogue history에 대응하는 스코어($S_\text{adv}$)와 Original dialogue history 스코어($S_\text{orig}$) 사이의 차이가 마진($m$) 이상이 되도록 유도합니다. 이와 관련해 자세한 내용은 앞선 챕터인 [Metric learning](http://ratsgo.github.io/insight-notes/docs/qa/metric)과 [이 글](https://gombru.github.io/2019/04/03/ranking_loss)을 참고하시면 좋을 것 같습니다.

사용자의 현재 질문을 쿼리(query)로 하고 사용자-AI 간 대화 이력을 문서(document)로 하는 검색 모델을 학습한다고 가정해 봅시다. 수식1의 직관적 의미는 이렇습니다. Pertubation dialogue history 스코어 $S_\text{adv}$는 사용자의 현재 질문과 Pertubation이 수행된 대화 이력 사이의 유사도 점수입니다. Original dialogue history 스코어 $S_\text{orig}$는 사용자의 현재 질문과 이 질문의 직전 대화 이력 사이의 유사도 점수입니다. 

**수식1을 최소화하려면 사용자의 현재 질문과 Pertubation이 수행된 대화 이력 사이의 유사도는 낮아져야 하고, 사용자의 현재 질문과 이 질문의 직전 대화 이력 사이의 유사도는 높아져야 합니다.** 요컨대 Consistency training 원래 목적대로 원본 데이터에 노이즈(pertubation)가 들어가 있어도 모델이 강건한 예측을 수행하게 됩니다.

실험(사용자의 현재 질문을 쿼리로 하고 대화 이력을 문서로 하는 검색 모델 학습) 결과 [In-batch-training](http://ratsgo.github.io/insight-notes/docs/qa/metric#in-batch-training) + [네거티브 로그라이클리후드(negative log-likelihood)](http://ratsgo.github.io/insight-notes/docs/qa/metric#negative-log-likelihood) 상황에서 수식1의 Pertubation Penalty를 함께 사용하면 검색 품질의 향상을 꾀할 수 있다고 합니다. 다시 말해 In-batch-training 상황에서 "네거티브 로그라이클리후드 + Pertubation Penalty"를 손실 함수로 사용한다는 것입니다. 이렇게 되면 모델은 1회 스텝에서 다음을 모두 수행하기 때문에 성능이 개선되는 것 같습니다.

1. In-batch-training 상황에서의 네거티브 로그라이클리후드 최소화 
  - 포지티브 쌍(positive pair, 현재 질문-직전 대화 이력) 사이의 거리를 좁힌다(=유사도를 높인다).
  - 네거티브 쌍(negative pair, 현재 질문-랜덤 대화 이력) 사이의 거리를 벌린다(=유사도를 낮춘다).
  - In-batch-training이기 때문에 포지티브 쌍은 배치 내에서 서로의 네거티브 쌍이 되고 서로가 서로의 의미를 고려하게 됨.
2. Pertubation Penalty 최소화
  - Original dialogue history(현재 질문-직전 대화 이력, 1의 포지티브 쌍과 동일)와 Perturbed dialogue history(현재 질문-pertubation 수행한 대화 이력) 사이의 거리를 $m$ 이상으로 벌린다(=유사도를 낮춘다).
  - 사용자의 현재 질문과 Pertubation이 수행된 대화 이력 사이의 거리는 멀게(=유사도는 낮아지게) 하고, 사용자의 현재 질문과 이 질문의 직전 대화 이력 사이의 거리는 가깝게(=유사도는 높아지게) 해야 함.


# SimCSE

[SimCSE](https://arxiv.org/pdf/2104.08821)는 문장 인코더(sentence encoder)의 새로운 학습 방법을 제안했습니다. 여기서 문장 인코더는 문장을 벡터로 임베딩하는 역할을 수행합니다. 우선 레이블이 있는 데이터가 있는 경우(supervised setting) 먼저 살펴보겠습니다. 

Natural Language Inference(NLI) 데이터를 가지고 문장 인코더를 학습하는 상황이라고 가정해 보겠습니다. 그림2처럼 premise를 벡터화하는 인코더와 hypothesis를 벡터화하는 인코더를 따로 둡니다. supervised setting이기 때문에 우리는 NLI 데이터의 레이블 정보 역시 활용할 수 있습니다. 

레이블이 entailment인 premise(`Two dogs are running.`)와 hypothesis(`There are animals outdoors`)를 포지티브 쌍으로 둡니다. 레이블이 contradiction인 premise(`Two dogs are running.`), hypothesis(`The pets are sitting on a couch.`)는 네거티브 쌍 취급합니다. 아울러 [In-batch-training](http://ratsgo.github.io/insight-notes/docs/qa/metric#in-batch-training)를 사용하기 때문에 premise와 전혀 관계 없는 hypothesis(`There is a man.`, `A kid is skateboarding.` 등) 역시 네거티브 쌍이 됩니다.

## **그림2** supervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RczwsgU.png" width="600px" title="source: imgur.com" />

**주의!!**
물론 NLI 데이터 가지고 entailment, contradiction, neutral 3개 범주 가운데 하나를 맞추는 분류기를 학습할 수 있습니다. 하지만 동일한 데이터로도 "모델의 출력이 범주 확률이냐, 유사도/거리이냐", "레이블을 범주로 주느냐, positive/negative로 주느냐"에 따라서 본질적으로 다른 모델이 탄생하게 됩니다. [이전 챕터](http://ratsgo.github.io/insight-notes/docs/qa/metric#supervised-setting) 참고.
{: .code-example }

[SimCSE](https://arxiv.org/pdf/2104.08821)의 핵심 기여는 unsupervised setting입니다. 그림3과 같습니다. 그림2와 동일한 NLI 데이터를 쓴다고 해도 레이블 없이 문장만 가지고 학습해야 합니다. 다시 말해 포지티브 쌍이 전혀 존재하지 않는다는 뜻입니다.

[SimCSE](https://arxiv.org/pdf/2104.08821) 저자들은 포지티브 쌍을 그림3처럼 부여했습니다. **드롭아웃(dropout)을 켠 채로 동일한 문장을 두 번 순전파(forward computation)하고 이 둘의 representation을 포지티브 쌍으로 간주합니다.** 다시 말해 드롭아웃을 노이즈로 하는 Consistency training인 셈입니다. 한편 unsupervised setting에서 네거티브 쌍은 배치 내 특정 인스턴스(`The dogs are running.`)와 해당 인스턴스를 제외한 모든 인스턴스(`A man surfing on the sea` 등)가 됩니다.

## **그림3** unsupervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Rq0nPaM.png" width="400px" title="source: imgur.com" />

한편 [SimCSE](https://arxiv.org/pdf/2104.08821) 저자들 역시 모든 실험에서 [In-batch-training](http://ratsgo.github.io/insight-notes/docs/qa/metric#in-batch-training) + [네거티브 로그라이클리후드(negative log-likelihood)](http://ratsgo.github.io/insight-notes/docs/qa/metric#negative-log-likelihood) 조합을 사용하였습니다. 이밖에 학습 디테일 역시 참고하기 좋은 것 같습니다.


---


# References

- [Zhou, W., Li, Q., & Li, C. (2021). Learning from Perturbations: Diverse and Informative Dialogue Generation with Inverse Adversarial Training. arXiv preprint arXiv:2105.15171.](https://arxiv.org/pdf/2105.15171)
- [Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. arXiv preprint arXiv:2104.08821.](https://arxiv.org/pdf/2104.08821)


---