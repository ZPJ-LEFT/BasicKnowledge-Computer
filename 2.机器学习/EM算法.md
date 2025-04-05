# 期望最大化算法

期望最大化算法（EM算法，Expectation-Maximization Algorithm） 是一种常用于隐变量模型参数估计的方法。

具体而言，当我们不能直接用 MLE 解模型参数时（因为有些变量是“看不见”的），我们用 EM 来迭代逼近。

## 数学定义与推导

假设有观测数据$X=\{x_1,x_2,...,x_n\}$，隐变量$Z=\{z_1,z_2,...,z_n\}$，参数$\theta$。

似然函数为:

$$
l(\theta)=\sum_{i=1}^nlogp(x_i,z_i|\theta)
$$

不同于MLE的是，除了参数$\theta$，我们也不知道隐变量$z$，因此需要枚举每种可能的z，此时似然函数实际上为：

$$
\begin{aligned}
l(\theta)=\sum_{i=1}^nlog\sum_{z_i}p(x_i,z_i|\theta)
\end{aligned}
$$

### 引入新的分布$q(z)$

假设$z_i\sim q(z_i)$，则：

$$
\begin{aligned}
l(\theta)&=\sum_{i=1}^nlog\sum_{z_i}q(z_i)\frac{p(x_i,z_i|\theta)}{q(z_i)}\\
&= \sum_{i=1}^nlogE_{q(z_i)}\frac{p(x_i,z_i|\theta)}{q(z_i)}\\
&\geq \sum_{i=1}^nE_{q(z_i)}log\frac{p(x_i,z_i|\theta)}{q(z_i)} (Jensen不等式)\\
\end{aligned}
$$

如果要使该下界逼近$l(\theta)$，需要调整$q(z)$和$p(x,z|\theta)$，根据Jensen不等式取等号的条件有以下关系：

$$
\frac{p(x,z|\theta)}{q(z)}=c，c为常数
$$

而$\sum_z q(z)=1$，所以可得$\sum_z p(x,z|\theta) = c$。

因此：

$$
\begin{aligned}
q(z_i) &= \frac{p(x_i,z_i|\theta)}{c} \\
&= \frac{p(x_i,z_i|\theta)}{\sum_{z_i} p(x_i,z_i|\theta)}\\
&= \frac{p(x_i,z_i|\theta)}{p(x_i|\theta)}\\
&= p(z_i|x_i, \theta)\\
\end{aligned}
$$

由此，通过求解$l(\theta)$的下界，我们得出一个隐变量的后验分布$q(z)$。

因此，我们可以将$\theta$的求解过程分为两步，先通过期望最大化求隐变量分布，再通过似然函数最大化求解结果。

## 算法流程

**输入**：观测数据$X={x_1,x_2,...,x_n}$，联合概率分布$p(x,z|\theta)$，条件分布$p(z|x,\theta)$，最大迭代次数N。

**算法步骤**：
1. 随机初始化参数$\theta^0$
2. t=0,...,N-1, 开始EM算法迭代：
    - **E步（Expectation）**

        在当前参数$\theta^{t}$下，估计隐变量的后验分布：

        $$
        q(z_i) = p(z_i|x_i, \theta^{t}) = \frac{p(z_i,x_i|\theta^t)}{\sum_{z_i} p(z_i,x_i|\theta^t)}
        $$

    - **M步（Maximization）**

        基于隐变量的后验分布，利用似然函数求解参数$\theta^{t+1}$：

        $$
        \begin{aligned}
        L(\theta^{t+1})&= \sum_{i=1}^nE_{q(z_i)}log\frac{p(x_i,z_i|\theta^{t})}{q(z_i)}\\
        &= \sum_{i=1}^n \sum_{z_i} q(z_i)log\frac{p(x_i,z_i|\theta^{t})}{q(z_i)}\\
        &= \sum_{i=1}^n \sum_{z_i} q(z_i)logp(x_i|z_i,\theta^t)\\
        \end{aligned}
        $$

## 关于算法的收敛性

简单而言，EM算法收敛的数学本质是：利用 Jensen 不等式构造的变分下界，每次迭代都在提升这个下界，从而单调地提升目标函数 log-likelihood。

## 算法实例

假设我们有A，B两枚硬币，其中正面朝上的概率分别为$\theta_{A},\theta_{B}$ ，这两个参数即为需要估计的参数。

我们设计5组实验，每次实验投掷5次硬币，但是每次实验都不知道用哪一枚硬币进行的本次实验。

投掷结束后，会得到一个数组$X={x1,x_2,...,x_5}$，表示每组实验有几次硬币正面朝上。

假设实验数据为：

$$
X=\{3,2,1,3,2\}
$$

请你估计$\theta_A$和$\theta_B$的值。

### 求解过程

该问题的隐变量是每次实验选取A硬币或B硬币，引入随机变量z表示选取A硬币(z=0)或B硬币(z=1)的概率，并假设其分布为q(z)。

首先随机选取参数的值，例如$\theta_A=0.2$和$\theta_B=0.7$。

#### E步
对于E步，需要求解在当前参数$\theta$和当前输入$x$下，隐变量z的概率：

$$
q(z_i|x_i,\theta) = \frac{q(x_i,z_i|\theta)}{\sum_{z_i} q(x_i,z_i|\theta)}
$$

对$x_1$：

- $p(z_1=0,x_1|\theta)=0.2^3\times0.8^2=0.00512$
- $p(z_1=1,x_1|\theta)=0.7^3\times0.3^2=0.3087$

则对于第一组实验，利用A硬币的概率为：

$$
q(z_1=0|x_1,\theta) = \frac{q(z=0,x_1|\theta)}{\sum_{z_1} q(z_1,x_1|\theta)} = \frac{0.00512}{0.00512+0.3087}=0.14
$$

利用B硬币的概率为：

$$
q(z_1=1|x_1,\theta) = \frac{q(z=1,x_1|\theta)}{\sum_{z_1} q(z_1,x_1|\theta)} = \frac{0.03087}{0.00512+0.3087}=0.86
$$

以此类推，可以得到结果如下：

|轮数|A硬币|B硬币|
|----|---|---|
|1|0.14|0.86|
|2|0.61|0.39|
|3|0.94|0.06|
|4|0.14|0.86|
|5|0.61|0.39|

#### M步

我们的目标是最小化似然函数，可以求得更新公式如下：

$$
\theta_A = \sum_{i=1}^n \frac{q(z_i=0)\times x_i}{5n}
$$

$$
\theta_B = \sum_{i=1}^n \frac{q(z_i=1)\times x_i}{5n}
$$

|轮数|A正|A负|B正|B负|
|----|---|---|---|---|
|1|0.42|0.28|2.58|1.72|
|2|1.22|1.83|0.78|1.17|
|3|0.94|3.76|0.06|0.24|
|4|0.42|0.28|2.58|1.72|
|5|1.22|1.83|0.78|1.17|
|Total|4.22|7.98|6.78|6.02|

由上表的数据，可得：

$$
\theta_A = \frac{4.22}{4.22+7.98}=0.35
$$

$$
\theta_B = \frac{6.78}{6.78+6.02}=0.5296875
$$

最后，一直循环迭代EM步，直至$\theta_A$和$\theta_A$不再变化。

# 参考文献
- [EM算法详解](https://zhuanlan.zhihu.com/p/40991784)
- [机器学习-隐变量模型和期望最大算法](https://zhuanlan.zhihu.com/p/136169435)
- [【大道至简】机器学习算法之EM算法(Expectation Maximization Algorithm)详解(附代码)---通俗理解EM算法](https://blog.csdn.net/qq_36583400/article/details/127047093)