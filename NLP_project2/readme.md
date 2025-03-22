# 2024 NLP Project 2
本次project我们将探索AI对人的情感激发。

请不要在网络上传播、公开本次project的代码与材料。

## 基础说明
main.py 用于生成对话，predict.py 用于预测对话中蕴含的情绪，evaluate.py 用于量化情感激发的效果，report.py 用于报告最后的平均激发成绩。
我们分别都提供相应的shell脚本来一键式运行：

```
bash speaker.sh / listener.sh # 将inducer切换到不同角色上完成情绪激发

bash predict.sh # 完成情绪预测

bash evaluate.sh # 完成情绪激发评测结果的获取

bash report.sh # 生成平均激发结果
```

## Emotion Classifier
在运行代码之前，请下载huggingface上开源的情绪分类模型到 emotion_recognition_model/ 文件夹下面：

https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

## 运行环境
根据你是否需要本地部署模型，选择是否需要安装vllm。
一般来说，按照下面的命令全部安装即可：

```
pip install -r ./requirements.txt
```

## Free Qwen API
https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-metering-and-billing?spm=a2c4g.11186623.0.0.40f0fa70Tfi8lC



