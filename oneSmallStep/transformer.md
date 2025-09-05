# transformer理解，里面的架构吗，自相关又是这么一回事
# 地址：https://poloclub.github.io/transformer-explainer/
# 八股地址：https://zhuanlan.zhihu.com/p/689965833

# 为什么需要多头注意力
注意到不同维度的信息，局部和全局，关注的信息更多。

# 为什么需要先升维度后降维
其实就是 Transformer 的 FFN（前馈神经网络）部分为什么要把维度从提升之后下降
主要是提升维度之后方便提取特征。高维一般包含信息更多。