**autointMLP.ipynb**

- Grid search 를 이용한 하이퍼파라미터 튜닝
    - learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    - dropout = hp.Choice('dropout', [0.3, 0.4, 0.5])
    - embed_dim = hp.Choice('embed_dim', [8, 16, 32])
    - hidden_units = hp.Choice('hidden_units', [32, 64, 128])
    - batch_size = hp.Choice('batch_size', [512, 1024, 2048])

**Hyperparameter Tuning TOP 3**<br/>
<img width="249" height="448" alt="image" src="https://github.com/user-attachments/assets/746a7c19-ddeb-4b65-8008-5e96e8e32e28" /> <br/><br/>


**BEST**<br/>
Trial 00 summary<br/>
🎯 Best Hyperparameters:<br/>
Learning rate: 0.0001<br/>
Dropout: 0.5<br/>
Embed dim: 32<br/>
Hidden units: 128<br/>
Batch size: 1024<br/>


|Before Tuning|After Tuning|
|:---|:---|
|<img width="187" height="35" alt="image" src="https://github.com/user-attachments/assets/a4e71679-4679-4d59-9ff0-6b79296f31b5" />|<img width="191" height="37" alt="image" src="https://github.com/user-attachments/assets/657560f7-8ac2-42cf-abed-73f8f3a40f10" />

