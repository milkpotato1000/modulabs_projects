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
|    |NDCG|Hit rate|
|:---|:---|:---|
|Before Tuning|0.6616|0.63034|
|After Tuning|0.66317|0.63107|


---
### 파라미터 최적화 전과 후 비교<br/>
**입력**<br/>
<img width="731" height="265" alt="image" src="https://github.com/user-attachments/assets/63e76214-55a1-40c4-a0b6-b45ec1f44b85" /><br/>

**파라미터 최적화 전 추천 결과**<br/>
<img width="744" height="1060" alt="image" src="https://github.com/user-attachments/assets/472f662f-3506-4a69-82e4-afe120a89b82" /><br/>



**파라미터 최적화 후 추천 결과**<br/>
<img width="758" height="1069" alt="image" src="https://github.com/user-attachments/assets/3d213f6f-49ae-496b-bea9-a01659dd48ee" /><br/>

- 유저가 즐겨보는 코메디 장르가 추천 목록에 포함됨
- 전반적으로 장르별 추천 유사도가 더 잘 맞는 경향

