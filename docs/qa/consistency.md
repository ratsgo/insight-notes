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

Consistency training이란
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


# Concept


Consistency training이란, label이 존재하지 않는 데이터의 입력 또는 은닉상태(hidden state)에 noise가 추가되어도 강건한 예측을 할 수 있도록 학습하는 방법론입니다. 즉, unlabeled 데이터와 noise가 추가된 unlabeled 데이터의 예측 분포가 유사하도록 제약을 만드는 것입니다. Noise는 대표적으로 Gaussian noise가 적용됩니다. 적은 labeled 데이터로 학습한 모델이 unlabeled 데이터를 예측할 경우 발생하는 불확실성을 보완하기 위한 방법인 것 같습니다. 


자연어 처리에서 대표적으로 쓰이는 Consistency training 방법은 학습 데이터 문장을 뻥튀기하는 'data augmentation', 히든에 노이즈를 주는 'pertubation' 등 방식이 있습니다. 전자는 EDA, back-translation 등이 있고 후자는 


# Data Augmentation

## Easy Data Augmentation

- Learning from Perturbations : 상환 님 시도와 연관

## **그림1** Learning from Perturbations
{: .no_toc .text-delta }
<img src="https://i.imgur.com/zy1hdrK.png" width="300px" title="source: imgur.com" />

## Back Translation

- UDA에서도 사용

## **그림1** Back Translation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/vVTr5xd.png" width="300px" title="source: imgur.com" />

## TF-IDF 활용한 키워드 대치

- UDA에서 사용

## LM 파인튜닝

- Conditional LM : Kumar, 학습 입력/프롬프트 입력 명시
- Query Converter와 연관, 학습 입력/프롬프트 입력 명시

## SentAugment

- Self-training

## **그림1** SentAugment
{: .no_toc .text-delta }
<img src="https://i.imgur.com/far3fxb.png" width="300px" title="source: imgur.com" />


## 질의-답변 쌍 자동으로 생성

- consistency와 직접 연관 없어 제외해야할듯

[PAQ](https://arxiv.org/pdf/2102.07033) 저자들은 위키피디아 문서들을 가지고 자동으로 질문-답변 쌍을 만들기도 했습니다. 물론 사람이 만든 것 대비 품질은 낮겠지만, 이렇게 자동으로 만들어진 문서-질문-답변 데이터는 여러모로 쓰임이 많을거라 생각합니다.

## **그림13** Knowledge Graph vs Language Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Lt2HCVR.png" width="600px" title="source: imgur.com" />


# 히든에 노이즈 주기

## SimCSE


## **그림1** supervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RczwsgU.png" width="600px" title="source: imgur.com" />

## **그림1** unsupervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Rq0nPaM.png" width="600px" title="source: imgur.com" />


---


# References

- [Wei, J., & Zou, K. (2019). Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196.](https://arxiv.org/pdf/1901.11196)
- [Zhou, W., Li, Q., & Li, C. (2021). Learning from Perturbations: Diverse and Informative Dialogue Generation with Inverse Adversarial Training. arXiv preprint arXiv:2105.15171.](https://arxiv.org/pdf/2105.15171)
- [Xie, Q., Dai, Z., Hovy, E., Luong, M. T., & Le, Q. V. (2019). Unsupervised data augmentation for consistency training. arXiv preprint arXiv:1904.12848.](https://arxiv.org/pdf/1904.12848)
- [Kumar, V., Choudhary, A., & Cho, E. (2020). Data augmentation using pre-trained transformer models. arXiv preprint arXiv:2003.02245.](https://arxiv.org/pdf/2003.02245)
- [Du, J., Grave, E., Gunel, B., Chaudhary, V., Celebi, O., Auli, M., ... & Conneau, A. (2020). Self-training improves pre-training for natural language understanding. arXiv preprint arXiv:2010.02194.](https://arxiv.org/pdf/2010.02194)
- [Lewis, P., Wu, Y., Liu, L., Minervini, P., Küttler, H., Piktus, A., ... & Riedel, S. (2021). Paq: 65 million probably-asked questions and what you can do with them. arXiv preprint arXiv:2102.07033.](https://arxiv.org/pdf/2102.07033)
- [Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. arXiv preprint arXiv:2104.08821.](https://arxiv.org/pdf/2104.08821)


---