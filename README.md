**Acknowledgement**

This code for FedSSL-AMC is adapted from Yiyue Chen's work on Evolutionary Clustering in Federated Learning [https://arxiv.org/abs/2509.07198]

**Libraries and their versions used for simulation**

- joblib==1.5.1
- matplotlib==3.10.3
- multiprocess==0.70.18
- numpy==2.1.2
- scikit-learn==1.6.1
- scipy==1.15.3
- torch==2.7.0+rocm6.3
- torchaudio==2.7.0+rocm6.3
- torchvision==0.22.0+rocm6.3

**Running the simulations**

- To generate the dataset, run python3 Generate_Dataset.py

- To Train the encoder, execute python3 Train_Encoder_Server.py

- To evaluate the train encoder and train the personalized SVMs, run python3 Output_layer_Personlized_Main
