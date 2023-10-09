# Weighted Adversarial Domain Adaptation for Machine Remaining Useful Life Prediction 
Kangkai Wu , Jingjing Li, Lin Zuo, Ke Lu, Heng Tao Shen

![Idea.jpg](https://s2.loli.net/2023/02/20/RlUytN91F7jIfrM.jpg)

Abstract: In industry, accurate remaining useful life (RUL) prediction is critical in improving system reliability, reducing downtime and accident risk. Numerous deep learning approaches have been proposed and achieved impressive performance in RUL prediction. Nevertheless, most of them are based on an unrealistic assumption, i.e., the training (source) and testing (target) data follow similar distributions. In real-world applications, the source and target domains usually have different data distributions, which degrades the model performance on the unlabeled target domain. Existing adversarial domain adaptive RUL prediction methods fail to consider the label information, which potentially aligns samples with very different semantic information and ultimately leads to negative transfer. To address the above issue, this paper proposes a weighted adversarial loss (WAL) for cross domain RUL prediction. To be specific, WAL utilizes the ground truth labels of source domain and the pseudo labels of target domain to calculate a weight and then obtain the weighted adversarial loss. This proposed loss forces the adversarial model to align samples with similar RULs from the source and target domains. Therefore, WAL can enhance positive transfer while alleviating negative transfer. Extensive experiments demonstrate that the proposed loss can be effectively plugged into existing adversarial domain adaptation methods and yields state-of-the-art results.

## Usage

* Conda Enviroment

    `conda env create -f environment.yaml`

* For Pretraining

    `python pretrain_phm.py`

* For Cross-domain Training

    `python main.py`

The processed data can be downloaded from this [LINK](https://drive.google.com/drive/folders/12vxOBouxJlrdfDTa0jCCTb5MQ6ccZ-2O?usp=sharing).

## Citation
```
@article{wu2022weighted,
  title={Weighted adversarial domain adaptation for machine remaining useful life prediction},
  author={Wu, Kangkai and Li, Jingjing and Zuo, Lin and Lu, Ke and Shen, Heng Tao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--11},
  year={2022},
  publisher={IEEE}
}
```
