# Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis

This repository contains the official implementation of the paper:  
**"[Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis](https://ieeexplore.ieee.org/document/11048667)"**  
published in IEEE Transactions on Instrumentation and Measurement.

### Overview
The intelligent fault diagnosis (IFD) methods obtain superior performance in ensuring the safety of mechanical systems, but varying working conditions degrade the performance of intelligent models. Fortunately, unsupervised domain adaptation (UDA) has been used to handle the bias and unannotated data. High-quality features contribute to facilitating subsequent domain alignment and enhancing diagnostic performance. This article aims to address the contradiction between using complex neural networks to extract better fault features and the resulting longer inference time. Specifically, a heterogeneous extractor is designed by integrating a pure CNN-based main network in parallel with a hybrid auxiliary network. The auxiliary network consists of a CNN and a ViT, connected in series, which extract local and global fault knowledge, respectively. Then, a training strategy is proposed to help the main branch enrich the extracted features, where the pure CNN is optimized to distinguish the hard-to-transfer samples identified by auxiliary CNN-ViT extractor. Finally, an information filter mechanism is introduced to facilitate mutual feature learning between the two branches. Experiments are constructed on two diagnosis datasets and a practical platform, where the comparison studies manifest the superiority of our method in fault diagnosis tasks under varying working conditions.
<img width="6104" height="4929" alt="fig3" src="https://github.com/user-attachments/assets/c53d05d9-bf7a-41c4-b52c-c3c947feb682" />

### Usage Guidelines
1. **Dataset Configuration**  
   Modify the `data_name` parameter to switch between different datasets, including Jiangnan University dataset, HUST bearing dataset.

2. **File Path Setup**  
   Update the `data_direction` variable in `train_util_HETPM.py` to point to your local dataset directory.
   
3. **Run HETPM_auto.py**  
   
### Citation
If you find this work useful, please cite our paper:

X. Yang and Y. Li, "Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis," in IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1-12, 2025, Art no. 3546612, doi: 10.1109/TIM.2025.3582302.

### Contact
For questions regarding the code or methodology, please contact yangxilin@sjtu.edu.cn.
