**autointMLP.ipynb**
- Grid search 를 이용한 하이퍼파라미터 튜닝
    - learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    - dropout = hp.Choice('dropout', [0.3, 0.4, 0.5])
    - embed_dim = hp.Choice('embed_dim', [8, 16, 32])
    - hidden_units = hp.Choice('hidden_units', [32, 64, 128])
    - batch_size = hp.Choice('batch_size', [512, 1024, 2048])

    <img width="850" height="510" alt="image" src="https://github.com/user-attachments/assets/40125b9e-5fd7-4f3e-867b-2cdc1d0ebb06" />
