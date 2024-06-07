## 竞价策略 Bidding Strategies (p.46)

### D - 数据 Data (p.46)

- **竞价请求 Bid request:**  包含有关用户、广告、页面和上下文的高维特征向量。
- **用户反馈 User feedback:** 点击或转化等用户对展示的广告做出的反应。
- **市场价格 Market price:** 赢得特定广告展示的最低出价（第二高出价）。
- **历史竞价数据 Historical bidding data:** 包括竞价、赢得/输掉拍卖、成本等信息。

### H - 假设空间 Hypothesis Space (p.50)

- **竞价函数 Bidding function:** 决定给定广告展示机会时出价的函数, $$b(r)$$。
- **获胜概率函数 Winning probability function:**  估计赢得给定出价的拍卖的概率, $$w(b)$$。
- **效用函数 Utility function:** 量化广告展示的效用（例如每次点击的价值或利润）, $$u(r)$$。
- **成本函数 Cost function:** 估计赢得广告展示的成本, $$c(b)$$。

### F - 问题建模 Problem Formulation (p.46)

- **单一广告系列竞价优化 Single Campaign Bid Optimization:** 在给定预算和竞价数量的限制下，确定最大化广告系列KPI（点击次数、转化次数或利润）的最优竞价函数。
- **多广告系列统计套利挖掘 Multi-Campaign Statistical Arbitrage Mining:** 作为多个广告系列的元竞价者，通过概率抽样选择一个广告系列并计算其广告出价，从而优化跨多个广告系列的利润，同时管理风险。
- **预算控制 Budget Pacing:** 优化预算分配，以确保在整个广告系列生命周期内平稳地消耗预算。 (p.57)

### C - 准则 Criteria (p.48)

- **最大化广告系列KPI Maximizing Campaign KPIs:**  目标是最大化点击次数、转化次数或利润等指标。
- **预算约束 Budget Constraints:** 竞价策略必须遵守预先设定的预算限制。
- **风险管理 Risk Management:** 在多广告系列竞价中，需要平衡利润最大化和风险最小化。(p.56)

### O - 优化 Optimization (p.51)

- **Truth-telling bidding:** 在没有预算限制的情况下，最优竞价策略是真实出价，即出价等于展示的真实价值：btrue(r)=vr，其中 v 是每次点击的价值。

- **Linear bidding:** 在考虑预算和竞价量约束的情况下，线性竞价函数是真实出价策略的扩展：blin(r)=ϕvr，其中 ϕ 是调整参数。

- **Budget constrained clicks and conversions maximisation:**

  $$max_{b(\cdot)} T \int_r u(r)w(b(r))p_r(r)dr
  subject to  T\int_r c(b(r))w(b(r))p_r(r)dr = B$$​ (p.52)

  $$

  \max_{b(\cdot)} T \int_r u(r)w(b(r))p_r(r)dr \\

  \text{subject to } T\int_r c(b(r))w(b(r))p_r(r)dr = B

  $$

- **Multi-campaign statistical arbitrage mining:**$$max_{b(\cdot),s} E[R]
  subject to  E[C] = B
  Var[R] = 1
  s^T1 = 1
  0 \le s \le 1$$(p.56)

- **Budget pacing:** 使用节流或竞价修改等方法来优化预算分配。(p.57)

### L - 学习 Learning (p.21)

- 使用历史竞价数据来估计获胜概率、效用和成本函数。
- 使用机器学习算法（如逻辑回归、生存模型）来学习和优化竞价函数。

### P - 预测/分析 Prediction (p.57)

- **实时竞价 Real-Time Bidding (RTB):** 根据学习到的竞价函数对每个广告展示进行实时竞价。
- **预算控制 Budget Pacing:** 监测广告系列的支出，并根据需要调整竞价策略。
- **效果评估 Performance Evaluation:** 分析广告系列结果，评估竞价策略的有效性。 



# Chapter 5: Bidding Strategies

## 1. Quantitative Bidding in RTB

- The bidding function depends only on the estimated utility and the cost of the ad display opportunity.

## 2. Notations and Preliminaries

- Bidding function notation: \( b(r) \) returns the bid price given the estimated click-through rate (CTR) \( r \).

## 3. Winning Probability

- Probability of winning the auction as a function of bid price \( b \):
  $$
  w(b) = \int_0^b p_z(z) \, dz \quad (5.1)
  $$
  where \( p_z(z) \) is the probability density function of the market price.

## 4. Utility and Cost Functions

- Utility function for click number:
  $$
  u_{	ext{clk}}(r) = r \quad (5.2)
  $$
- Utility for campaign's profit:
  $$
  u_{	ext{profit}}(r, z) = vr - z \quad (5.3)
  $$
- Expected cost in a first price auction:
  $$
  c_1(b) = b \quad (5.4)
  $$
- Expected cost in a second price auction:
  $$
  c_2(b) = rac{\int_0^b zp_z(z) \, dz}{\int_0^b p_z(z) \, dz} \quad (5.5)
  $$
  These define how the bid price is adjusted based on the cost and utility of ad impressions.

## 5. Optimal Bidding Strategy

- General form for the optimal bid price \( b(r) \):
  $$
  b(r) = \phi vr \quad (5.10)
  $$
  where \( \phi \) is a factor adjusted to fit market competitiveness and volume.

## 6. Budget-Constrained Maximization

- Framework for maximizing clicks and conversions under budget constraints, using utility and cost functions integrated over all ad requests:
  $$
  \max_b \int u(r)w(b(r))p_r(r) \, dr \quad 	ext{subject to} \quad \int c(b(r))w(b(r))p_r(r) \, dr = B
  $$
  Here, \( B \) is the budget, \( u(r) \) and \( c(b(r)) \) represent the utility and cost functions, respectively.

