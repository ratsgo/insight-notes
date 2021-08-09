---
layout: default
title: Consistency training
parent: Retrieval model
grand_parent: Question Answering
nav_order: 3
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


# SimCSE


## **그림1** supervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RczwsgU.png" width="600px" title="source: imgur.com" />

## **그림1** unsupervised setting
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Rq0nPaM.png" width="600px" title="source: imgur.com" />


---


# References

- [Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. arXiv preprint arXiv:2104.08821.](https://arxiv.org/pdf/2104.08821)


---