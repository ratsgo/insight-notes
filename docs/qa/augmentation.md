---
layout: default
title: Data Augmentation
parent: Retrieval model
grand_parent: Question Answering
nav_order: 3
permalink: /docs/qa/augmentation
---

# Data Augmentation
{: .no_toc }

Data Augmentation은 원본 데이터를 불려서 모델 성능을 높이는 기법입니다. 텍스트를 대상으로 수행하는 기법을 중심으로 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Concept

**Data augmentation**이란 일반적으로 지도 학습(Supervised Learning)에서 사용되는 방법으로, 레이블이 존재하는 데이터에 변화를 줘 원본 데이터와 같은 레이블을 가지는 새로운 데이터를 만드는 기법입니다. 이렇게 불린 데이터를 가지고 모델을 학습하면 이전 대비 강건한(robust) 모델을 만들 수 있는 것으로 알려져 있습니다. 비전(vision) 분야에서는 이미지 회전, 확대 등, 음성(speech) 분야에서는 피치(pitch) 주파수(frequency) 변화 등이 대표적입니다. 이 글에서는 텍스트 데이터에 대한 Data Augmentation 기법을 살펴봅니다.


# Easy Data Augmentation

가장 쉽게 해볼 수 있는 augmentation 기법입니다. 그래서 이름하여 [Easy Data Augmentation](https://arxiv.org/pdf/1901.11196). 내용은 표1과 같습니다.

## **표1** Easy Data Augmentation
{: .no_toc .text-delta }

|구분|내용|예시|
|---|---|---|
|원본 문장|-|A sad, superior human comedy played out on the back roads of life.|
|Synonym Replacement|랜덤하게 유의어로 대치|A **lamentable**, superior human comedy played out on the **backward** road of life.|
|Random Insertion|랜덤한 위치에 랜덤한 단어를 삽입|A sad, superior human comedy played out on **funniness** the back roads of life.|
|Random Swap|랜덤하게 단어 순서 변경|A sad, superior human comedy played out on **roads** back **the** of life.|
|Random Deletion|랜덤하게 단어 삭제|A sad, superior human out on the roads of life.|

# 키워드가 아닌 단어를 대치

일반적으로 키워드(keyword)는 해당 문장의 의미에서 핵심적인 역할을 담당하는 경우가 많습니다. 따라서 문장 내 키워드(아래 예시에서 `꿀잼`)를 아무 단어로 대치(replacement)하면 문장의 의미가 달라질 염려가 있습니다. 아래 예시처럼 말이죠. 이 경우 **원본 데이터와 같은 레이블을 가지는 새로운 데이터를 만든다**는 Data augmentation의 철학에서 벗어나게 됩니다.

- `이 영화 꿀잼` > `이 영화 노잼`

[Xie et al.(2019)](https://arxiv.org/pdf/1904.12848)는 이에 문장에서 정보량이 낮은 단어를 대상으로만 대치하는 전략을 택했습니다. 아울러 위의 예시처럼 대치 후 단어(`노잼`)가 정보량이 높은 단어, 즉 키워드라면 이 역시 레이블이 달라질 가능성을 배제할 수 없기 때문에 대치 후 단어 역시 정보량이 낮은 단어들 가운데서 선택되도록 했습니다. 여기에서 정보량 지표로는 TF-IDF를 활용했습니다. 자세한 내용은 [Xie et al.(2019)](https://arxiv.org/pdf/1904.12848)의 Appendix A.2를 살펴보시기 바랍니다. 


# Back Translation

Back Translation 역시 텍스트 분야에서 널리 사용되는 Data Augmentation 기법입니다. 우선 augmentation 대상 문장을 특정 언어로 번역합니다. 이후 번역된 문장을 다시 원래 언어로 다시 번역합니다. 그림1과 같습니다.

## **그림1** Back Translation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/vVTr5xd.png" width="700px" title="source: imgur.com" />

잘 학습된 번역 모델이 있다면 Back Translation을 수행한다고 해도 원래 문장의 의미를 보존한 상태로 새로운 문장을 만들 수 있을 겁니다. 한편 번역 모델을 디코딩할 때 탑-$k$ 샘플링 등 다양성(diversity)을 높이는 기법을 적용한다면 좀 더 다양한 문장으로 augmentation을 수행할 수 있습니다.

# LM 파인튜닝

[Kumar et al.(2020)](https://arxiv.org/pdf/2003.02245)은 프리트레인을 마친 언어 모델(language model)을 다운스트림 태스크 데이터로 파인튜닝해서 Data augmentation을 수행했습니다. 예컨대 [NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)를 augmentation하는 상황이라고 가정해 봅시다. 원본 데이터는 다음과 같다고 치겠습니다.

- 이 영화 졸잼, 긍정
- 이 영화 꿀잼, 긍정
- 이 영화 노잼, 부정
- ...

배치 크기가 3일 때 파인튜닝 데이터는 다음과 같이 구성합니다. 파인튜닝 대상 모델은 GPT2 같은 트랜스포머 디코더(transformer decoder) 등을 사용했습니다.

- 긍정 [SEP] 이 영화 졸잼 [EOS] 긍정 [SEP] 이 영화 꿀잼 [EOS] 부정 [SEP] 이 영화 노잼 [EOS]

파인튜닝을 마치면 다음과 같은 방식의 프롬프트(prompt)를 주어서 conditional generation이 되도록 합니다. 원하는 레이블과 함께 문장 앞쪽 $k$개 단어를 프롬프트로 구성합니다(아래는 $k=2$인 예시). 이렇게 하면 해당 레이블에 맞는 새로운 문장을 다량 생성할 수 있습니다. 이 역시 다양성을 높이는 디코딩 기법으로 새로운 문장을 다수 만들어낼 수 있습니다.

- **긍정 문장 augmentation** : 긍정 [SEP] 이 영화
- **부정 문장 augmentation** : 부정 [SEP] 이 영화


# SentAugment

[SentAugment](https://arxiv.org/pdf/2010.02194) 저자들은 검색(retrieval) 기반의 augmentation 기법을 제안했습니다. 도식도는 그림2와 같습니다.

## **그림2** SentAugment
{: .no_toc .text-delta }
<img src="https://i.imgur.com/far3fxb.png" width="700px" title="source: imgur.com" />

SentAugment 적용 순서는 다음과 같습니다.

1. 문장 인코더(문장을 representation으로 변환)를 가지고 *labeled sentence*(다운스트림 태스크 데이터)는 물론 *unlabeled sentence*를 모두 벡터로 변환.
2. *labeled sentence*의 representation을 가지고 다음 3가지 유형의 임베딩을 만든다.
  - **per-sent** : *labeled sentence* 각각의 임베딩.
  - **label-avg** : *labeled sentence* 임베딩의 레이블별 평균 벡터.
  - **all-avg** : *labeled sentence* 임베딩 전체의 평균 벡터.
3. 2에서 구한 세 부류의 벡터와 코사인 유사도가 높은 *unlabeled sentence* 임베딩을 추리고 그에 해당하는 문장을 augmentation 결과로 취함.
4. *labeled sentence*로만 학습한 분류기에 3에서 구한 문장들을 입력해 레이블을 새로 달아줌.


---


# References

- [Wei, J., & Zou, K. (2019). Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196.](https://arxiv.org/pdf/1901.11196)
- [Xie, Q., Dai, Z., Hovy, E., Luong, M. T., & Le, Q. V. (2019). Unsupervised data augmentation for consistency training. arXiv preprint arXiv:1904.12848.](https://arxiv.org/pdf/1904.12848)
- [Xie, Q., Dai, Z., Hovy, E., Luong, M. T., & Le, Q. V. (2019). Unsupervised data augmentation for consistency training. arXiv preprint arXiv:1904.12848.](https://arxiv.org/pdf/1904.12848)
- [Kumar, V., Choudhary, A., & Cho, E. (2020). Data augmentation using pre-trained transformer models. arXiv preprint arXiv:2003.02245.](https://arxiv.org/pdf/2003.02245)
- [Du, J., Grave, E., Gunel, B., Chaudhary, V., Celebi, O., Auli, M., ... & Conneau, A. (2020). Self-training improves pre-training for natural language understanding. arXiv preprint arXiv:2010.02194.](https://arxiv.org/pdf/2010.02194)


---