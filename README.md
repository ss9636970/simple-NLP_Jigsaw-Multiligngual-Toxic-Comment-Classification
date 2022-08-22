# NLP for Jigsaw Multiligngual Toxic Comment Classification

# 1. 說明

本篇實作自然語言處理，使用bert將文字轉文向量，再用不同模型訓練，ex: LSTM, Ridge model

bert參考: https://github.com/huggingface/transformers



# 2. 檔案說明

dataEmbedding.ipynb: 使用transformers將文字轉為向量

DataLoader.py: 載入資料

dataProcess.ipynb: 整理資料

model.py: LSTM

train_model.ipynb: 模型訓練

pred_model_RidgeAndBert: 預測資料



