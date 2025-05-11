# Inverse Reinforcement Learning by Logistic Regression

[Inverse Reinforcement Learning by Logistic Regression](https://link.springer.com/article/10.1007/s11063-017-9702-7)  
Eiji Uchibe によって提案された逆強化学習（IRL）の手法の一つで、事前分布などの恣意的な設定を最小限に抑えたのが特徴です。

## 特長
- 事前分布の設定がほぼ不要  
- ロジスティック回帰による簡潔な実装  
- 標準的な IRL 手法と比較してチューニングポイントが少ない  

## アーキテクチャ概要
以下の図は、Log-Reg IRL の主要モジュールとデータフローを示しています。

![Log-Reg IRL アーキテクチャ](https://github.com/user-attachments/assets/d4992ed3-5bcd-4bb1-949b-6fec5c507059)

1. **Expert Trajectory Collector**  
   - 専門家デモンストレーションから状態・行動データを収集  
2. **Feature Extractor**  
   - 各状態–行動対の特徴ベクトルを算出  
3. **Logistic Regression Module**  
   - 専門家の行動とランダム行動を区別するためのロジスティック回帰を学習  
4. **Reward Inference**  
   - 学習済みモデルの重みを報酬関数に変換  
5. **Policy Optimizer**  
   - 推定された報酬関数を使って最適方策（policy）を更新  
