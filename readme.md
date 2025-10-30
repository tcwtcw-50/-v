# Transformer Translation Model

本项目基于 **PyTorch** 实现了一个轻量级的 **Transformer 序列到序列（Seq2Seq）翻译模型**，  
支持自定义数据集训练与验证，可视化损失曲线，并在测试语句上生成预测结果。

---

## 📂 1. 项目结构

TransformerProject/
│
├── tokenizer/
│ └── tokenizer.py # 简易分词器定义
│
├── transformer/
│ ├── transformer_model.py # Transformer 模型定义
│ ├── layers.py # 注意力与前馈模块
│
├── utils/
│ ├── device.py # 设备选择（CPU/GPU）
│ ├── loss.py # Label Smoothing, 损失计算
│ ├── train_utils.py # 训练与验证循环
│
├── data.txt # 简单的中英平行语料
├── train.py # 主训练脚本
├── requirements.txt # 环境依赖文件
└── README.md # 项目说明文档

yaml
复制代码

---

## ⚙️ 2. 环境配置

### 方式一：使用 conda
```bash
conda create -n transformer python=3.10
conda activate transformer
pip install -r requirements.txt
方式二：使用 pip
bash
复制代码
python3 -m venv venv
source venv/bin/activate   # Windows 使用 venv\Scripts\activate
pip install -r requirements.txt
🚀 3. 数据准备
项目示例数据文件为：

kotlin
复制代码
data.txt
每一行包含一对平行句（例如英德翻译）：

kotlin
复制代码
i love you    ich liebe dich
this is a book    das ist ein buch
用户可根据任务替换为自己的语料。

🧠 4. 训练命令
运行主程序 train.py：

bash
复制代码
python train.py
程序将自动：

加载数据与分词；

划分训练与验证集；

构建 Transformer 模型；

进行多轮训练（默认 50 epoch）；

在 ./output/ 目录保存训练曲线与模型权重。

📊 5. 模型超参数
参数名称	默认值	说明
batch_size	32	每批样本数
n_layers	2	Transformer 层数
n_heads	4	注意力头数
d_model	128	词向量维度
d_hid	512	前馈网络维度
lr	3e-4	学习率
epochs	50	训练轮数

📈 6. 可视化
训练完成后，会生成：

bash
复制代码
output/loss_curve.png
output/loss_ppl_curves.png
图中展示了训练与验证集的 损失（Loss） 和 困惑度（PPL） 变化趋势。

🧩 7. 推理示例
在训练结束后，可在 train.py 底部修改如下代码进行测试：

python
复制代码
sentences_to_test = [
    "i love you",
    "this is a book",
    "are you ok"
]
运行脚本后将输出翻译预测结果。

🧱 8. 环境与硬件要求
类型	推荐配置
Python	3.10+
GPU	NVIDIA GTX 1660 / RTX 3060 或更高
运行时间	约 10–20 分钟 (50 epoch，小语料)