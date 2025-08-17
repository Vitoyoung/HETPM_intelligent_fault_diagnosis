# Heterogeneous Heterogeneous Heter Heterogeneous## Heter Heterogeneous Extractor for Cross-Domain Bearing Fault Diagnosis

This repository contains the official implementation of the paper:  
**"Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis"**  
published in IEEE Transactions on Instrumentation and Measurement.

### Overview
The code implements the proposed Heterogeneous extractor framework for cross-domain rolling bearing fault diagnosis, enabling knowledge transfer between different operating conditions or bearing types. The model integrates local feature extraction and global pattern learning to enhance diagnostic performance across domains.

### Usage Guidelines
1. **Dataset Configuration**  
   Modify the `data_name` parameter to switch between different datasets.

2. **File Path Setup**  
   Update the `data_direction` variable in `train_util_HETPM.py` to point to your local dataset directory.

3. **Training & Evaluation**  
   Follow the example scripts to run experiments. The main training pipeline supports automatic domain adaptation and performance logging.

### Citation
If you find this work useful, please cite our paper:

X. Yang and Y. Li, "Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis," in IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1-12, 2025, Art no. 3546612, doi: 10.1109/TIM.2025.3582302.

### Contact
For questions regarding the code or methodology, please contact the corresponding author.
