# Weighted Adversarial Domain Adaptation for Machine Remaining Useful Life Prediction 
Kangkai Wu , Jingjing Li, Lin Zuo, Ke Lu, Heng Tao Shen

![Idea.jpg](https://s2.loli.net/2023/02/20/RlUytN91F7jIfrM.jpg)

Abstract: In industry, accurate remaining useful life (RUL) prediction is critical in improving system reliability, reducing downtime and accident risk. Numerous deep learning approaches have been proposed and achieved impressive performance in RUL prediction. Nevertheless, most of them are based on an unrealistic assumption, i.e., the training (source) and testing (target) data follow similar distributions. In real-world applications, the source and target domains usually have different data distributions, which degrades the model performance on the unlabeled target domain. Existing adversarial domain adaptive RUL prediction methods fail to consider the label information, which potentially aligns samples with very different semantic information and ultimately leads to negative transfer. To address the above issue, this paper proposes a weighted adversarial loss (WAL) for cross domain RUL prediction. To be specific, WAL utilizes the ground truth labels of source domain and the pseudo labels of target domain to calculate a weight and then obtain the weighted adversarial loss. This proposed loss forces the adversarial model to align samples with similar RULs from the source and target domains. Therefore, WAL can enhance positive transfer while alleviating negative transfer. Extensive experiments demonstrate that the proposed loss can be effectively plugged into existing adversarial domain adaptation methods and yields state-of-the-art results.

## Usage

* For Pretraining

    `python pretrain_phm.py`

* For Cross-domain Training

    `python main.py`
