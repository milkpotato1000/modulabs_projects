**autointMLP.ipynb**

- Grid search 를 이용한 하이퍼파라미터 튜닝
    - learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    - dropout = hp.Choice('dropout', [0.3, 0.4, 0.5])
    - embed_dim = hp.Choice('embed_dim', [8, 16, 32])
    - hidden_units = hp.Choice('hidden_units', [32, 64, 128])
    - batch_size = hp.Choice('batch_size', [512, 1024, 2048])<br/>

**Hyperparameter Tuning TOP 3**<br/>
<img width="249" height="448" alt="image" src="https://github.com/user-attachments/assets/746a7c19-ddeb-4b65-8008-5e96e8e32e28" /><br/>

**BEST**<br/>
Trial 00 summary<br/>
🎯 Best Hyperparameters:<br/>
Learning rate: 0.0001<br/>
Dropout: 0.5<br/>
Embed dim: 32<br/>
Hidden units: 128<br/>
Batch size: 1024<br/>

**추천 성능 향상**
|    |Before Tuning|After Tuning|
|:---|:---|:---|
|NDCG|0.66155|66317|
|Hit rate|0.6301|0.63107|


---
### 파라미터 최적화 전과 후 비교

**파라미터 최적화 전 추천 결과**
<img width="773" height="1218" alt="image" src="https://github.com/user-attachments/assets/b0a159d7-db88-448b-9db6-1ba58c76f203" /><br/>

**파라미터 최적화 후 추천 결과**


