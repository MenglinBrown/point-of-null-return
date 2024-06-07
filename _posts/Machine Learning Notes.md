<a name="wnmjl"></a>
## 步骤 (DHFCOLP)	
| 一级分类 | 二级分类 | 展开 |
| --- | --- | --- |
| 模型Model | **D - 数据 (Data)：得到一个有限的训练数据集合** | 训练集 input space<br />$X = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$<br />特征向量 feature vector （特征空间：所有特征向量集合）<br />$\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)})^T$	 |
|  | **H - 假设空间 (Hypothesis Space)：确定包含所有可能的模型的假设空间，即学习模型的集合** | 模型的假设空间包括所有可能的条件概率分布或决策函数。由输入空间到输出空间的映射的集合<br />$\begin{array}{|c|c|}
    \hline
    & \text{假设空间 } \mathcal{F}  \\
    \hline
    \text{决策函数} & \mathcal{F} = \{ f_\theta \mid Y = f_\theta(x), \theta \in \mathbb{R}^n \} \\
    \hline
    \text{条件概率分布} & \mathcal{F} = \{ P \mid P_\theta (Y \mid X), \theta \in \mathbb{R}^n \}  \\
    \hline
\end{array}$ |
|  | **F - Problem Formulation（把D和H组合到一起）	** | 在监督学习过程中，**模型**就是所要学习的**条件概率分布**或者**决策函数**。<br />概率模型/生成模型，非概率模型()/判别模型<br />$\begin{array}{|c|c|c|c|c|}
    \hline
    & \text{假设空间 } \mathcal{F} & \text{输入空间 } \mathcal{X} & \text{输出空间 } \mathcal{Y} & \text{参数空间} \\
    \hline
    \text{决策函数} & \mathcal{F} = \{ f_\theta \mid Y = f_\theta(x), \theta \in \mathbb{R}^n \} & \text{变量} & \text{变量} & \mathbb{R}^n \\
    \hline
    \text{条件概率分布} & \mathcal{F} = \{ P \mid P_\theta (Y \mid X), \theta \in \mathbb{R}^n \} & \text{随机变量} & \text{随机变量} & \mathbb{R}^n \\
    \hline
\end{array}$<br />notation: 大写的字母$X$表示随机变量，小写字母$x$表示随机变量的取值，大写字母$P$表示概率。<br />- **离散变量**的概率分布可以用**概率质量函数** (probability mass function, PMF) 来描述。<br />   - Bernoulli Distribution的PMF：$p(y;\phi) = \phi^y(1-\phi)^{1-y}$. 其中，ϕ 表示 Y=1 的概率<br />- **连续变量**的概率分布可以用**概率密度函数** (probability density function, PDF) 来描述。<br />   - Gaussian Distribution/Normal Distribution的PDF：$p(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$<br /> |
| 策略Strategy | **C - 准则 (Criteria)确定模型选择的准则，即学习的策略** | <br />1. 损失函数loss function/代价函数cost function e.g. 0-1 loss, least mean squre LMS, abs loss, log loss<br />   1. 损失函数定义：对给定输入X的预测值f（x)和真实值Y之间的Y之间的非负实值函数，L（Y，f（X））<br />      1. least mean square: $J(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$<br />      2. log loss: <br />2. 现在有了损失函数，损失函数数值越小，模型越好，损失函数的期望就是期望损失。学习的目标就是选择期望风险最小的模型。<br />   1. 风险函数risk function和期望损失expected loss<br />      1. 公式：$R_{exp}(f)=E_p[L(Y, f(X))]=\int_{\mathcal X\times\mathcal Y}L(y,f(x))P(x,y)\, {\rm d}x{\rm d}y \tag{1.9}$<br />      2. 如果知道联合分布P（X，Y）就可以直接计算Inference出来，不需要学习Learning，所以学习学的就是这个联合分布（cs221 reflex models：learning）<br />3. 但是现在不知道联合分布P（x，y），所以需要用emperical loss，来近似<br />   1. 经验风险empirical risk和经验损失emprirical loss<br />      1. 公式：$R_{emp}(f)=\frac{1}{N}\sum^{N}_{i=1}L(y_i,f(x_i)) \tag{1.10}$<br />      2. 期望风险时模型关于联合分布的期望损失，经验风险时模型关于训练样本集的平均损失。<br />      3. 根据大数定律，当样本容量N趋于无穷大时，经验风险趋于期望风险。<br />   2. 经验风险最小化(empirical risk minimization, ERM)<br />      1. 公式：$\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) \tag{1.11}$其中F是假设空间<br />      2. 经验风险最小的就是最优的模型，极大似然估计(maximum likelihood estimation, MLE)就是经验风险最小化的一个例子，当模型是条件概率分布conditional distribution，损失函数是对数损失函数log loss的时候，经验风险最小化ERM等价于极大似然估计<br />4. 但是样本容量小的时候，经验风险最小化学习容易出现过拟合overfitting，所以用结构风险最小化SRM防止过拟合<br />   1. 结构风险structural risk<br />      1. 公式：$R_{srm}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f) \tag{1.12}$其中 J(f)为模型复杂度，lambda>=0 是系数，用以权衡经验风险和模型复杂度<br />      2. 结构风险最小化SRM等价于正则化regularization:结构风险在经验风险上加上表示模型复杂度的正则化项regularizer/penalty term<br />   2. 结构风险最小化(structural risk minimization, SRM)<br />      1. 公式：$\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) + \lambda J(f) \tag{1.13}$<br />      2. 贝叶斯估计中的最大后验概率估计(maximum posterior probability estimation, MAP) 就是SRM的一个例子。当模型是条件概率分布conditional distribution、损失函数是对数损失函数log loss、模型复杂度由模型的先验概率表示的时候，结构风险最小化就等价于最大后验概率估计<br />5. 至此，监督学习问题就成了经验风险或结构风险函数的最优化问题(1.11, 1.13)，这时经验或结构风险函数就是最优化的目标函数<br /> |
| Learning | **O - 优化 (Optimization)实现求解最优模型的算法，即学习算法** | <br />1. 朴素贝叶斯法[**CH04**](../CH04/README.md)和隐马尔科夫模型[**CH10**](../CH10/README.md)<br />2. 感知机[**CH02**](../CH02/README.md)，逻辑斯谛回归模型[**CH06**](../CH06/README.md)，最大熵模型[**CH06**](../CH06/README.md)，条件随机场[**CH11**](../CH11/README.md)<br />3. 支持向量机[**CH07**](../CH07/README.md)<br />4. 决策树[**CH05**](../CH05/README.md)<br />5. 提升方法[**CH08**](../CH08/README.md)<br />6. EM算法[**CH09**](../CH09/README.md)<br />7. NB和HMM的监督学习，最优解就是极大似然估计值，可以由概率计算公式直接计算。之前看NB其实就是计数查表，这种要有大的语料库进行统计，所谓学的多，就知道的多。<br /> |
|  | **L - 学习 (Learning)通过学习方法选择最优的模型	** |  |
|  | **P - 预测/分析 (Prediction)利用学习的最优模型对新数据进行预测或分析** |  |

<a name="wLR6L"></a>
## Model By Model (DHFCOLP)	
| <br /> |  |  | **模型Model (**Problem Formulation**)** |  |  | **策略Strategy** | **算法Algorithm** |  | <br /> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 步骤 (DHFCOLP) |  |  | D - 数据 (Data | H - 假设空间 (Hypothesis Space) | 参数parameters | C - 准则 (Criteria) | O - 优化 (Optimization) | L - 学习 (Learning) | P - 预测/分析 (Prediction) |
| 监督学习 | <br /> | linear regression | 结构化数据 vs 非结构化数据 | as 决策函数<br />$h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}=\sum_{i=0}^{d} \theta_{i} x_{i}=\theta^{T} x$<br />as 条件概率分布<br />Construct as GLM: y is continuous and we model the conditional distribution of y given x as a $Gaussian\mathcal{N}\left(\mu, \sigma^{2}\right)$<br />$\begin{aligned}h_{\theta}(x) &=E[y \mid x ; \theta] \\&=\mu \\&=\eta \\&=\theta^{T} x\end{aligned}$ |  | least mean square: $J(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$<br />probalistic view:<br /> |  |  |  |
|  |  | logistic regression |  | as 决策函数：logistic function/sigmoid function<br />逻辑回归模型可以表示为以下决策函数的形式：<br />$$ f(x) = \\begin{cases} 1, & \\theta^Tx \\ge 0 \\ 0, & \\theta^Tx < 0 \\end{cases} $$<br />其中，x 是输入特征，θ 是模型参数。当 θTx≥0 时，模型预测样本属于类别 1，否则预测样本属于类别 0。--> 其实就是perceptron<br /><br />as 条件概率分布：y服从伯努利分布，特征之间没有假设<br />更常见的是，逻辑回归模型表示为条件概率分布的形式：<br />$$ P(y=1&#124;x;\\theta) = h_\\theta(x) = \\frac{1}{1+e^{-\\theta^Tx}} $$<br />$$ P(y=0&#124;x;\\theta) = 1 - h_\\theta(x) $$<br />其中，hθ(x) 是 sigmoid 函数，其输出值在 0 和 1 之间，可以解释为样本属于类别 1 的概率。<br />$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$<br />as GLM：假设空间hypothesis function是期望函数<br />$\begin{aligned}h_{\theta}(x) &=E[y \mid x ; \theta] \\
&=\phi \\
&=\frac{1}{1+e^{-\eta}} \\
&=\frac{1}{1+e^{-\theta^{T} x}}\end{aligned}\\$<br />$P(y=1|x;\theta) = h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$<br />assume<br />$\begin{array}{l}
P(y=1 \mid x ; \theta)=h_{\theta}(x) \\
P(y=0 \mid x ; \theta)=1-h_{\theta}(x)
\end{array}$<br />compactly: $p(y \mid x ; \theta)=\left(h_{\theta}(x)\right)^{y}\left(1-h_{\theta}(x)\right)^{1-y}$<br /><br />广义线性模型（GLM）属于李航所说的**条件概率分布假设空间**。<a name="z6N6O"></a>
### **理由**
<br />- **GLM 的核心**：GLM 的核心在于对条件概率分布 p(y∣x;θ) 进行建模，其中 y 是输出，x 是输入，θ 是模型参数。模型通过假设 y∣x;θ 服从某个指数族分布来构建。<br />- **指数族分布**：指数族分布是一类广泛的概率分布，包括高斯分布、伯努利分布、多项式分布等。<br />- **条件概率建模**：无论是线性回归（高斯分布）、逻辑回归（伯努利分布）还是 softmax 回归（多项式分布），GLM 都是通过对给定输入 x 的条件下，输出 y 的概率分布进行建模来实现的。<br />- **决策函数是推导结果**：虽然 GLM 可以通过条件概率分布推导出决策函数（例如，逻辑回归中的 sigmoid 函数），但其核心仍然是对条件概率分布的建模。<br /> | 逻辑回归是一种判别模型，直接对条件概率 p(y∣x) 进行建模。<br />- 决策边界：线性<br />- 参数：权重向量theta<br /> | Assume i.i.d, likelihood of the parameters:<br />$L(\theta) = p(\mathbf{y} | X; \theta) = \prod_{i=1}^n p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^n \left( h_\theta(x^{(i)}) \right)^{y^{(i)}} \left( 1 - h_\theta(x^{(i)}) \right)^{1 - y^{(i)}}$<br />这里还缺一个用glm的角度看loss function<br />log likelihood:<br />$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \left( y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})) \right)$ |  |  |  |
|  |  | naive bayes |  | 学习联合概率分布. HOW? <br />通过学习 $P(X|Y)$和 $P(Y)$的估计i.e. $P(X^{(i)} = x^{(i)} | Y = c_k) \text{和}  P(Y = c_k)$，得到联合概率分布<br />$P(X,Y) = P(Y)P(X|Y)$<br />$\begin{aligned}
p\left(x^{(i)}, y^{(i)}\right) &=\prod_{j=1}^{d} p\left(x_{j}^{(i)} \mid y^{(i)}\right) p\left(y^{(i)}\right)
\end{aligned}$<br /> | 朴素贝叶斯是一种生成模型，对联合概率分布 p(x,y) 进行建模。<br />- 决策边界：非线性<br />- 参数：先验概率p(y)和条件概率p(x&#124;y)<br /> | $\begin{aligned}
\ell &=\log \prod_{i=1}^{n} p\left(x^{(i)}, y^{(i)}\right)=\log \prod_{i=1}^{n} p\left(y^{(i)}\right) \prod_{j=1}^{d} p\left(x_{j}^{(i)} \mid y^{(i)}\right) \\
&=\sum_{i=1}^{n}\left[\log p\left(y^{(i)}\right)+\sum_{j=1}^{d} \log p\left(x_{j}^{(i)} \mid y^{(i)}\right)\right]
\end{aligned}$ |  |  |  |
|  |  | 感知机Perceptron | 输入空间：<br />$x_i \in \mathcal X\sube \bf R^n$<br />输出空间：<br />$y_i \in \mathcal Y= \{+1,-1\}$ | <br /><br />决策函数：$h_{\theta}(x)=sign (\theta^T x+b)$<a name="zbbxj"></a>
### **与决策函数假设空间的区别**
决策函数假设空间关注的是从输入到输出的直接映射关系，不显式地对条件概率分布进行建模。例如，感知机算法直接学习一个决策函数（分离超平面），将输入空间划分为不同的类别，而不涉及概率。 |  | <a name="ujpQN"></a>
#### 损失函数选择
损失函数的一个自然选择是误分类点的总数，但是，这样的损失函数**不是参数w,b的连续可导函数，不易优化**<br />损失函数的另一个选择是误分类点到超平面S的总距离，这是感知机所采用的<br />感知机学习的经验风险函数(损失函数)<br />$L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)$<br />其中M是误分类点的集合<br />给定训练数据集T，损失函数L(w,b)是w和b的连续可导函数 |  |  |  |
|  |  | softmax | <br /> | $h_{\theta}(x) = 
\begin{bmatrix}
p(y = 1 \mid x; \theta) \\
p(y = 2 \mid x; \theta) \\
\vdots \\
p(y = k \mid x; \theta)
\end{bmatrix}
=
\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x}}
\begin{bmatrix}
e^{\theta_{1}^{T} x} \\
e^{\theta_{2}^{T} x} \\
\vdots \\
e^{\theta_{k}^{T} x}
\end{bmatrix}$ |  |  |  |  |  |
|  | state-based models | decision tree |  |  |  | quick select |  |  |  |
| <br /> | <br /> | augmented bayesian tree |  |  |  |  |  |  |  |
| <br /> | variable-based models | bayesian networkds |  | <a name="yANfN"></a>
### 
 |  |  |  |  |  |
| <br /> | <br /> | <br /> |  |  |  |  |  |  |  |
| 无监督学习 | <br /> | <br /> |  |  |  |  |  |  |  |
| 强化学习 | <br /> | <br /> |  |  |  |  |  |  |  |

| **方法** | **适用问题** | **模型特点** | **模型类型** | **学习策略** | **学习的损失函数** | **学习算法** |
| --- | --- | --- | --- | --- | --- | --- |
| Peceptron | 二类分类 | 分离超平面 | 判别模型 | 极小化误分点到超平面距离 | 误分点到超平面距离 | SGD |
| KNN | 多类分类, 回归 | 特征空间, 样本点 | 判别模型 |  |  |  |
| NB | 多类分类 |  | 生成模型 | MLE, MAP | 对数似然损失 | 概率计算公式, EM算法 |
| DT | 二类分类 |  | 判别模型 | 正则化的极大似然估计 | 对数似然损失 | 特征选择, 生成, 剪枝 |
| LR Maxent | 多类分类 |  | 判别模型 |  |  |  |
| SVM | 二类分类 |  | 判别模型 |  |  |  |
| AdaBoost | 二类分类 |  | 判别模型 |  |  |  |
| EM | 概率模型参数估计 | 含隐变量的概率模型 |  |  |  |  |
| HMM | 标注 | 观测序列与状态序列的联合概率分布模型 | 生成模型 |  |  |  |
| CRF | 标注 |  | 判别模型 |  |  |  |

