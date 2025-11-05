**autointMLP.ipynb**
- Grid search 를 이용한 하이퍼파라미터 튜닝
    - learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    - dropout = hp.Choice('dropout', [0.3, 0.4, 0.5])
    - embed_dim = hp.Choice('embed_dim', [8, 16, 32])
    - hidden_units = hp.Choice('hidden_units', [32, 64, 128])
    - batch_size = hp.Choice('batch_size', [512, 1024, 2048])

    <img width="850" height="510" alt="image" src="https://github.com/user-attachments/assets/40125b9e-5fd7-4f3e-867b-2cdc1d0ebb06" />


<img width="207" height="760" alt="image" src="https://github.com/user-attachments/assets/e7fed7bd-467b-482c-a0c6-e0d888cf66fa" />

**Trial 09 summary**
**Hyperparameters:**
- learning_rate: 0.0005
- dropout: 0.3
- embed_dim: 16
- hidden_units: 128
- batch_size: 1024
- Score: 0.5112023949623108
