## 步骤 (DHFCOLP)	
| 一级分类 | 二级分类 | 展开 |
| --- | --- | --- |
| 模型Model | **D - 数据 (Data)：得到一个有限的训练数据集合** | 训练集 input space
![](https://cdn.nlark.com/yuque/__latex/168d9c00915870c0a65aa68b3ac45147.svg#card=math&code=X%20%3D%20%5C%7B%28%5Cmathbf%7Bx%7D_1%2C%20y_1%29%2C%20%28%5Cmathbf%7Bx%7D_2%2C%20y_2%29%2C%20%5Cdots%2C%20%28%5Cmathbf%7Bx%7D_N%2C%20y_N%29%5C%7D%0A&id=WLojI)
特征向量 feature vector （特征空间：所有特征向量集合）
![](https://cdn.nlark.com/yuque/__latex/ca6770a33bca9e621b3f10c9e8524a16.svg#card=math&code=%0A%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D%20%3D%20%28x_1%5E%7B%28i%29%7D%2C%20x_2%5E%7B%28i%29%7D%2C%20%5Cdots%2C%20x_n%5E%7B%28i%29%7D%29%5ET&id=oFUmX)	 |
|  | **H - 假设空间 (Hypothesis Space)：确定包含所有可能的模型的假设空间，即学习模型的集合** | 模型的假设空间包括所有可能的条件概率分布或决策函数。由输入空间到输出空间的映射的集合
![](https://cdn.nlark.com/yuque/__latex/3c6d989a62dd61f25aa13be2221bf948.svg#card=math&code=%5Cbegin%7Barray%7D%7B%7Cc%7Cc%7C%7D%0A%20%20%20%20%5Chline%0A%20%20%20%20%26%20%5Ctext%7B%E5%81%87%E8%AE%BE%E7%A9%BA%E9%97%B4%20%7D%20%5Cmathcal%7BF%7D%20%20%5C%5C%0A%20%20%20%20%5Chline%0A%20%20%20%20%5Ctext%7B%E5%86%B3%E7%AD%96%E5%87%BD%E6%95%B0%7D%20%26%20%5Cmathcal%7BF%7D%20%3D%20%5C%7B%20f_%5Ctheta%20%5Cmid%20Y%20%3D%20f_%5Ctheta%28x%29%2C%20%5Ctheta%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%5C%7D%20%5C%5C%0A%20%20%20%20%5Chline%0A%20%20%20%20%5Ctext%7B%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83%7D%20%26%20%5Cmathcal%7BF%7D%20%3D%20%5C%7B%20P%20%5Cmid%20P_%5Ctheta%20%28Y%20%5Cmid%20X%29%2C%20%5Ctheta%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%5C%7D%20%20%5C%5C%0A%20%20%20%20%5Chline%0A%5Cend%7Barray%7D&id=PHNBo) |
|  | **F - Problem Formulation（把D和H组合到一起）	** | 在监督学习过程中，**模型**就是所要学习的**条件概率分布**或者**决策函数**。
概率模型/生成模型，非概率模型()/判别模型
![](https://cdn.nlark.com/yuque/__latex/50c65c6153b651e39cc438fa2dc11811.svg#card=math&code=%5Cbegin%7Barray%7D%7B%7Cc%7Cc%7Cc%7Cc%7Cc%7C%7D%0A%20%20%20%20%5Chline%0A%20%20%20%20%26%20%5Ctext%7B%E5%81%87%E8%AE%BE%E7%A9%BA%E9%97%B4%20%7D%20%5Cmathcal%7BF%7D%20%26%20%5Ctext%7B%E8%BE%93%E5%85%A5%E7%A9%BA%E9%97%B4%20%7D%20%5Cmathcal%7BX%7D%20%26%20%5Ctext%7B%E8%BE%93%E5%87%BA%E7%A9%BA%E9%97%B4%20%7D%20%5Cmathcal%7BY%7D%20%26%20%5Ctext%7B%E5%8F%82%E6%95%B0%E7%A9%BA%E9%97%B4%7D%20%5C%5C%0A%20%20%20%20%5Chline%0A%20%20%20%20%5Ctext%7B%E5%86%B3%E7%AD%96%E5%87%BD%E6%95%B0%7D%20%26%20%5Cmathcal%7BF%7D%20%3D%20%5C%7B%20f_%5Ctheta%20%5Cmid%20Y%20%3D%20f_%5Ctheta%28x%29%2C%20%5Ctheta%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%5C%7D%20%26%20%5Ctext%7B%E5%8F%98%E9%87%8F%7D%20%26%20%5Ctext%7B%E5%8F%98%E9%87%8F%7D%20%26%20%5Cmathbb%7BR%7D%5En%20%5C%5C%0A%20%20%20%20%5Chline%0A%20%20%20%20%5Ctext%7B%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83%7D%20%26%20%5Cmathcal%7BF%7D%20%3D%20%5C%7B%20P%20%5Cmid%20P_%5Ctheta%20%28Y%20%5Cmid%20X%29%2C%20%5Ctheta%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%5C%7D%20%26%20%5Ctext%7B%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%7D%20%26%20%5Ctext%7B%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%7D%20%26%20%5Cmathbb%7BR%7D%5En%20%5C%5C%0A%20%20%20%20%5Chline%0A%5Cend%7Barray%7D&id=hM1d5)
notation: 大写的字母![](https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg#card=math&code=X&id=jnJpp)表示随机变量，小写字母![](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg#card=math&code=x&id=yWJKe)表示随机变量的取值，大写字母![](https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg#card=math&code=P&id=nDFu8)表示概率。
- **离散变量**的概率分布可以用**概率质量函数** (probability mass function, PMF) 来描述。
   - Bernoulli Distribution的PMF：![](https://cdn.nlark.com/yuque/__latex/e7e4216fc107748d9780d6b0c487712b.svg#card=math&code=p%28y%3B%5Cphi%29%20%3D%20%5Cphi%5Ey%281-%5Cphi%29%5E%7B1-y%7D%0A&id=BrhYC). 其中，ϕ 表示 Y=1 的概率
- **连续变量**的概率分布可以用**概率密度函数** (probability density function, PDF) 来描述。
   - Gaussian Distribution/Normal Distribution的PDF：![](https://cdn.nlark.com/yuque/__latex/7f5f1e633fe2fe3511906474fc3ac0d3.svg#card=math&code=p%28x%3B%5Cmu%2C%5Csigma%5E2%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%5Csigma%5E2%7D%7Dexp%28-%5Cfrac%7B%28x-%5Cmu%29%5E2%7D%7B2%5Csigma%5E2%7D%29%0A&id=gx4jz)
 |
| 策略Strategy | **C - 准则 (Criteria)确定模型选择的准则，即学习的策略** | 
1. 损失函数loss function/代价函数cost function e.g. 0-1 loss, least mean squre LMS, abs loss, log loss
   1. 损失函数定义：对给定输入X的预测值f（x)和真实值Y之间的Y之间的非负实值函数，L（Y，f（X））
      1. least mean square: ![](https://cdn.nlark.com/yuque/__latex/290bcfb43203f4524e511797f3dcb03a.svg#card=math&code=J%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D&id=Dq70y)
      2. log loss: 
2. 现在有了损失函数，损失函数数值越小，模型越好，损失函数的期望就是期望损失。学习的目标就是选择期望风险最小的模型。
   1. 风险函数risk function和期望损失expected loss
      1. 公式：![](https://cdn.nlark.com/yuque/__latex/bb71264f3876adca1853ed417f5a5a51.svg#card=math&code=R_%7Bexp%7D%28f%29%3DE_p%5BL%28Y%2C%20f%28X%29%29%5D%3D%5Cint_%7B%5Cmathcal%20X%5Ctimes%5Cmathcal%20Y%7DL%28y%2Cf%28x%29%29P%28x%2Cy%29%5C%2C%20%7B%5Crm%20d%7Dx%7B%5Crm%20d%7Dy%20%5Ctag%7B1.9%7D&id=SZvdF)
      2. 如果知道联合分布P（X，Y）就可以直接计算Inference出来，不需要学习Learning，所以学习学的就是这个联合分布（cs221 reflex models：learning）
3. 但是现在不知道联合分布P（x，y），所以需要用emperical loss，来近似
   1. 经验风险empirical risk和经验损失emprirical loss
      1. 公式：![](https://cdn.nlark.com/yuque/__latex/e09c95f089fc14c08f2c622cba7434e3.svg#card=math&code=R_%7Bemp%7D%28f%29%3D%5Cfrac%7B1%7D%7BN%7D%5Csum%5E%7BN%7D_%7Bi%3D1%7DL%28y_i%2Cf%28x_i%29%29%20%5Ctag%7B1.10%7D&id=Ae9x3)
      2. 期望风险时模型关于联合分布的期望损失，经验风险时模型关于训练样本集的平均损失。
      3. 根据大数定律，当样本容量N趋于无穷大时，经验风险趋于期望风险。
   2. 经验风险最小化(empirical risk minimization, ERM)
      1. 公式：![](https://cdn.nlark.com/yuque/__latex/d1d54408e4d69ef3f594a131bd254fe5.svg#card=math&code=%5Cmin_%7Bf%20%5Cin%20%5Cmathcal%7BF%7D%7D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20L%28y_i%2C%20f%28x_i%29%29%20%5Ctag%7B1.11%7D&id=vKPPo)其中F是假设空间
      2. 经验风险最小的就是最优的模型，极大似然估计(maximum likelihood estimation, MLE)就是经验风险最小化的一个例子，当模型是条件概率分布conditional distribution，损失函数是对数损失函数log loss的时候，经验风险最小化ERM等价于极大似然估计
4. 但是样本容量小的时候，经验风险最小化学习容易出现过拟合overfitting，所以用结构风险最小化SRM防止过拟合
   1. 结构风险structural risk
      1. 公式：![](https://cdn.nlark.com/yuque/__latex/ef03b6668922621a470d2c577a7810f8.svg#card=math&code=R_%7Bsrm%7D%28f%29%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL%28y_i%2Cf%28x_i%29%29%2B%5Clambda%20J%28f%29%20%5Ctag%7B1.12%7D&id=PludN)其中 J(f)为模型复杂度，lambda>=0 是系数，用以权衡经验风险和模型复杂度
      2. 结构风险最小化SRM等价于正则化regularization:结构风险在经验风险上加上表示模型复杂度的正则化项regularizer/penalty term
   2. 结构风险最小化(structural risk minimization, SRM)
      1. 公式：![](https://cdn.nlark.com/yuque/__latex/8664dbf95755143e74002e853ac4a471.svg#card=math&code=%5Cmin_%7Bf%20%5Cin%20%5Cmathcal%7BF%7D%7D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20L%28y_i%2C%20f%28x_i%29%29%20%2B%20%5Clambda%20J%28f%29%20%5Ctag%7B1.13%7D&id=Ox5ve)
      2. 贝叶斯估计中的最大后验概率估计(maximum posterior probability estimation, MAP) 就是SRM的一个例子。当模型是条件概率分布conditional distribution、损失函数是对数损失函数log loss、模型复杂度由模型的先验概率表示的时候，结构风险最小化就等价于最大后验概率估计
5. 至此，监督学习问题就成了经验风险或结构风险函数的最优化问题(1.11, 1.13)，这时经验或结构风险函数就是最优化的目标函数
 |
| Learning | **O - 优化 (Optimization)实现求解最优模型的算法，即学习算法** | 
1. 朴素贝叶斯法[**CH04**](../CH04/README.md)和隐马尔科夫模型[**CH10**](../CH10/README.md)
2. 感知机[**CH02**](../CH02/README.md)，逻辑斯谛回归模型[**CH06**](../CH06/README.md)，最大熵模型[**CH06**](../CH06/README.md)，条件随机场[**CH11**](../CH11/README.md)
3. 支持向量机[**CH07**](../CH07/README.md)
4. 决策树[**CH05**](../CH05/README.md)
5. 提升方法[**CH08**](../CH08/README.md)
6. EM算法[**CH09**](../CH09/README.md)
7. NB和HMM的监督学习，最优解就是极大似然估计值，可以由概率计算公式直接计算。之前看NB其实就是计数查表，这种要有大的语料库进行统计，所谓学的多，就知道的多。
 |
|  | **L - 学习 (Learning)通过学习方法选择最优的模型	** |  |
|  | **P - 预测/分析 (Prediction)利用学习的最优模型对新数据进行预测或分析** |  |

## Model By Model (DHFCOLP)	
| 
 |  |  | **模型Model (**Problem Formulation**)** |  |  | **策略Strategy** | **算法Algorithm** |  | 
 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 步骤 (DHFCOLP) |  |  | D - 数据 (Data | H - 假设空间 (Hypothesis Space) | 参数parameters | C - 准则 (Criteria) | O - 优化 (Optimization) | L - 学习 (Learning) | P - 预测/分析 (Prediction) |
| 监督学习 | 
 | linear regression | 结构化数据 vs 非结构化数据 | as 决策函数
![](https://cdn.nlark.com/yuque/__latex/fc3b60e43a208e32f09bdf3a27e36ab4.svg#card=math&code=h_%7B%5Ctheta%7D%28x%29%3D%5Ctheta_%7B0%7D%2B%5Ctheta_%7B1%7D%20x_%7B1%7D%2B%5Ctheta_%7B2%7D%20x_%7B2%7D%3D%5Csum_%7Bi%3D0%7D%5E%7Bd%7D%20%5Ctheta_%7Bi%7D%20x_%7Bi%7D%3D%5Ctheta%5E%7BT%7D%20x&id=dvic0)
as 条件概率分布
Construct as GLM: y is continuous and we model the conditional distribution of y given x as a ![](https://cdn.nlark.com/yuque/__latex/05ba9d1f5915fbda79178e3a5263c24b.svg#card=math&code=Gaussian%5Cmathcal%7BN%7D%5Cleft%28%5Cmu%2C%20%5Csigma%5E%7B2%7D%5Cright%29&id=TqcJn)
![](https://cdn.nlark.com/yuque/__latex/245133bb6fd370c2b38cbb4cebb3e491.svg#card=math&code=%5Cbegin%7Baligned%7Dh_%7B%5Ctheta%7D%28x%29%20%26%3DE%5By%20%5Cmid%20x%20%3B%20%5Ctheta%5D%20%5C%5C%26%3D%5Cmu%20%5C%5C%26%3D%5Ceta%20%5C%5C%26%3D%5Ctheta%5E%7BT%7D%20x%5Cend%7Baligned%7D%0A&id=nIruW) |  | least mean square: ![](https://cdn.nlark.com/yuque/__latex/290bcfb43203f4524e511797f3dcb03a.svg#card=math&code=J%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D&id=ju3Lv)
probalistic view:
 |  |  |  |
|  |  | logistic regression |  | as 决策函数：logistic function/sigmoid function
逻辑回归模型可以表示为以下决策函数的形式：
$$ f(x) = \\begin{cases} 1, & \\theta^Tx \\ge 0 \\ 0, & \\theta^Tx < 0 \\end{cases} $$
其中，x 是输入特征，θ 是模型参数。当 θTx≥0 时，模型预测样本属于类别 1，否则预测样本属于类别 0。--> 其实就是perceptron

as 条件概率分布：y服从伯努利分布，特征之间没有假设
更常见的是，逻辑回归模型表示为条件概率分布的形式：
$$ P(y=1&#124;x;\\theta) = h_\\theta(x) = \\frac{1}{1+e^{-\\theta^Tx}} $$
$$ P(y=0&#124;x;\\theta) = 1 - h_\\theta(x) $$
其中，hθ(x) 是 sigmoid 函数，其输出值在 0 和 1 之间，可以解释为样本属于类别 1 的概率。
![](https://cdn.nlark.com/yuque/__latex/f5d92f69405262e5fa261c66adbf5132.svg#card=math&code=h_%5Ctheta%28x%29%20%3D%20g%28%5Ctheta%5ET%20x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%5Ctheta%5ET%20x%7D%7D&id=D6z9c)
as GLM：假设空间hypothesis function是期望函数
![](https://cdn.nlark.com/yuque/__latex/7a71d708cd8c0adb15723fbd23fcd1d0.svg#card=math&code=%5Cbegin%7Baligned%7Dh_%7B%5Ctheta%7D%28x%29%20%26%3DE%5By%20%5Cmid%20x%20%3B%20%5Ctheta%5D%20%5C%5C%0A%26%3D%5Cphi%20%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ceta%7D%7D%20%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5E%7BT%7D%20x%7D%7D%5Cend%7Baligned%7D%5C%5C%0A%0A&id=p9pYC)
![](https://cdn.nlark.com/yuque/__latex/950bf5174fcd2bb311638031a84631f8.svg#card=math&code=P%28y%3D1%7Cx%3B%5Ctheta%29%20%3D%20h_%5Ctheta%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5ETx%7D%7D%0A&id=JFzyK)
assume
![](https://cdn.nlark.com/yuque/__latex/38500b474e7ef5fae40fd2adce68e3f1.svg#card=math&code=%5Cbegin%7Barray%7D%7Bl%7D%0AP%28y%3D1%20%5Cmid%20x%20%3B%20%5Ctheta%29%3Dh_%7B%5Ctheta%7D%28x%29%20%5C%5C%0AP%28y%3D0%20%5Cmid%20x%20%3B%20%5Ctheta%29%3D1-h_%7B%5Ctheta%7D%28x%29%0A%5Cend%7Barray%7D&id=rxsJc)
compactly: ![](https://cdn.nlark.com/yuque/__latex/76c0b95585424d803b73ecfdf1540de9.svg#card=math&code=p%28y%20%5Cmid%20x%20%3B%20%5Ctheta%29%3D%5Cleft%28h_%7B%5Ctheta%7D%28x%29%5Cright%29%5E%7By%7D%5Cleft%281-h_%7B%5Ctheta%7D%28x%29%5Cright%29%5E%7B1-y%7D&id=kSeHo)

广义线性模型（GLM）属于李航所说的**条件概率分布假设空间**。### **理由**

- **GLM 的核心**：GLM 的核心在于对条件概率分布 p(y∣x;θ) 进行建模，其中 y 是输出，x 是输入，θ 是模型参数。模型通过假设 y∣x;θ 服从某个指数族分布来构建。
- **指数族分布**：指数族分布是一类广泛的概率分布，包括高斯分布、伯努利分布、多项式分布等。
- **条件概率建模**：无论是线性回归（高斯分布）、逻辑回归（伯努利分布）还是 softmax 回归（多项式分布），GLM 都是通过对给定输入 x 的条件下，输出 y 的概率分布进行建模来实现的。
- **决策函数是推导结果**：虽然 GLM 可以通过条件概率分布推导出决策函数（例如，逻辑回归中的 sigmoid 函数），但其核心仍然是对条件概率分布的建模。
 | 逻辑回归是一种判别模型，直接对条件概率 p(y∣x) 进行建模。
- 决策边界：线性
- 参数：权重向量theta
 | Assume i.i.d, likelihood of the parameters:
![](https://cdn.nlark.com/yuque/__latex/cc9caafbaf5617f10ad31c633169ab94.svg#card=math&code=L%28%5Ctheta%29%20%3D%20p%28%5Cmathbf%7By%7D%20%7C%20X%3B%20%5Ctheta%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5En%20p%28y%5E%7B%28i%29%7D%20%7C%20x%5E%7B%28i%29%7D%3B%20%5Ctheta%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5En%20%5Cleft%28%20h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20%5Cright%29%5E%7By%5E%7B%28i%29%7D%7D%20%5Cleft%28%201%20-%20h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20%5Cright%29%5E%7B1%20-%20y%5E%7B%28i%29%7D%7D&id=qlLvw)
这里还缺一个用glm的角度看loss function
log likelihood:
![](https://cdn.nlark.com/yuque/__latex/7cdd1dcc10eab5a8874a0986859fbcd7.svg#card=math&code=%5Cell%28%5Ctheta%29%20%3D%20%5Clog%20L%28%5Ctheta%29%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20%5Cleft%28%20y%5E%7B%28i%29%7D%20%5Clog%20h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20%2B%20%281%20-%20y%5E%7B%28i%29%7D%29%20%5Clog%20%281%20-%20h_%5Ctheta%28x%5E%7B%28i%29%7D%29%29%20%5Cright%29&id=ua7K6) |  |  |  |
|  |  | naive bayes |  | 学习联合概率分布. HOW? 
通过学习 ![](https://cdn.nlark.com/yuque/__latex/77bbc7a633995548490b0fa0d89575b9.svg#card=math&code=P%28X%7CY%29&id=Nj0as)和 ![](https://cdn.nlark.com/yuque/__latex/766b113289a916ca44480a32955dd773.svg#card=math&code=P%28Y%29&id=Qttdk)的估计i.e. ![](https://cdn.nlark.com/yuque/__latex/48a4785f6c0fcbbc9db250bb20f268fe.svg#card=math&code=P%28X%5E%7B%28i%29%7D%20%3D%20x%5E%7B%28i%29%7D%20%7C%20Y%20%3D%20c_k%29%20%5Ctext%7B%E5%92%8C%7D%20%20P%28Y%20%3D%20c_k%29&id=uuYJK)，得到联合概率分布
![](https://cdn.nlark.com/yuque/__latex/dd16187b320660e764f1ffab63f8d26c.svg#card=math&code=P%28X%2CY%29%20%3D%20P%28Y%29P%28X%7CY%29&id=etVfx)
![](https://cdn.nlark.com/yuque/__latex/373fd20fec4ca62cf622d664420dba74.svg#card=math&code=%5Cbegin%7Baligned%7D%0Ap%5Cleft%28x%5E%7B%28i%29%7D%2C%20y%5E%7B%28i%29%7D%5Cright%29%20%26%3D%5Cprod_%7Bj%3D1%7D%5E%7Bd%7D%20p%5Cleft%28x_%7Bj%7D%5E%7B%28i%29%7D%20%5Cmid%20y%5E%7B%28i%29%7D%5Cright%29%20p%5Cleft%28y%5E%7B%28i%29%7D%5Cright%29%0A%5Cend%7Baligned%7D&id=N3kG2)
 | 朴素贝叶斯是一种生成模型，对联合概率分布 p(x,y) 进行建模。
- 决策边界：非线性
- 参数：先验概率p(y)和条件概率p(x&#124;y)
 | ![](https://cdn.nlark.com/yuque/__latex/f8be3f8ccba0c67365f369fc1c51b74e.svg#card=math&code=%5Cbegin%7Baligned%7D%0A%5Cell%20%26%3D%5Clog%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20p%5Cleft%28x%5E%7B%28i%29%7D%2C%20y%5E%7B%28i%29%7D%5Cright%29%3D%5Clog%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20p%5Cleft%28y%5E%7B%28i%29%7D%5Cright%29%20%5Cprod_%7Bj%3D1%7D%5E%7Bd%7D%20p%5Cleft%28x_%7Bj%7D%5E%7B%28i%29%7D%20%5Cmid%20y%5E%7B%28i%29%7D%5Cright%29%20%5C%5C%0A%26%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%5B%5Clog%20p%5Cleft%28y%5E%7B%28i%29%7D%5Cright%29%2B%5Csum_%7Bj%3D1%7D%5E%7Bd%7D%20%5Clog%20p%5Cleft%28x_%7Bj%7D%5E%7B%28i%29%7D%20%5Cmid%20y%5E%7B%28i%29%7D%5Cright%29%5Cright%5D%0A%5Cend%7Baligned%7D&id=KYT8E) |  |  |  |
|  |  | 感知机Perceptron | 输入空间：
![](https://cdn.nlark.com/yuque/__latex/dbe9dc4886f91f16e4bd8f068ea9781f.svg#card=math&code=x_i%20%5Cin%20%5Cmathcal%20X%5Csube%20%5Cbf%20R%5En&id=X1TJy)
输出空间：
![](https://cdn.nlark.com/yuque/__latex/a166889214ef9429bb466835693f1f98.svg#card=math&code=y_i%20%5Cin%20%5Cmathcal%20Y%3D%20%5C%7B%2B1%2C-1%5C%7D&id=QYjwp) | 

决策函数：![](https://cdn.nlark.com/yuque/__latex/e1c7a729d49b3d31fc97d7e8e5574f52.svg#card=math&code=h_%7B%5Ctheta%7D%28x%29%3Dsign%20%28%5Ctheta%5ET%20x%2Bb%29&id=jb43b)### **与决策函数假设空间的区别**
决策函数假设空间关注的是从输入到输出的直接映射关系，不显式地对条件概率分布进行建模。例如，感知机算法直接学习一个决策函数（分离超平面），将输入空间划分为不同的类别，而不涉及概率。 |  | #### 损失函数选择
损失函数的一个自然选择是误分类点的总数，但是，这样的损失函数**不是参数w,b的连续可导函数，不易优化**
损失函数的另一个选择是误分类点到超平面S的总距离，这是感知机所采用的
感知机学习的经验风险函数(损失函数)
![](https://cdn.nlark.com/yuque/__latex/73a46ff85119304288d8d584e83789b7.svg#card=math&code=L%28w%2Cb%29%3D-%5Csum_%7Bx_i%5Cin%20M%7Dy_i%28w%5Ccdot%20x_i%2Bb%29%0A&id=Pj4Hr)
其中M是误分类点的集合
给定训练数据集T，损失函数L(w,b)是w和b的连续可导函数 |  |  |  |
|  |  | softmax | 
 | ![](https://cdn.nlark.com/yuque/__latex/ed32a2427b63f429611d2b5b1bb51a9c.svg#card=math&code=%0Ah_%7B%5Ctheta%7D%28x%29%20%3D%20%0A%5Cbegin%7Bbmatrix%7D%0Ap%28y%20%3D%201%20%5Cmid%20x%3B%20%5Ctheta%29%20%5C%5C%0Ap%28y%20%3D%202%20%5Cmid%20x%3B%20%5Ctheta%29%20%5C%5C%0A%5Cvdots%20%5C%5C%0Ap%28y%20%3D%20k%20%5Cmid%20x%3B%20%5Ctheta%29%0A%5Cend%7Bbmatrix%7D%0A%3D%0A%5Cfrac%7B1%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bk%7D%20e%5E%7B%5Ctheta_%7Bj%7D%5E%7BT%7D%20x%7D%7D%0A%5Cbegin%7Bbmatrix%7D%0Ae%5E%7B%5Ctheta_%7B1%7D%5E%7BT%7D%20x%7D%20%5C%5C%0Ae%5E%7B%5Ctheta_%7B2%7D%5E%7BT%7D%20x%7D%20%5C%5C%0A%5Cvdots%20%5C%5C%0Ae%5E%7B%5Ctheta_%7Bk%7D%5E%7BT%7D%20x%7D%0A%5Cend%7Bbmatrix%7D&id=gZ3Ox) |  |  |  |  |  |
|  | state-based models | decision tree |  |  |  | quick select |  |  |  |
| 
 | 
 | augmented bayesian tree |  |  |  |  |  |  |  |
| 
 | variable-based models | bayesian networkds |  | ### 
 |  |  |  |  |  |
| 
 | 
 | 
 |  |  |  |  |  |  |  |
| 无监督学习 | 
 | 
 |  |  |  |  |  |  |  |
| 强化学习 | 
 | 
 |  |  |  |  |  |  |  |

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

