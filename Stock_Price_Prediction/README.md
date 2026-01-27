# Multi-Stock Trading with Reinforcement Learning (PPO)

This project implements a custom **Multi-Stock Trading Environment** to train a Reinforcement Learning agent to manage a portfolio of Nifty 50 stocks. Using the **Proximal Policy Optimization (PPO)** algorithm, the agent learns to make buy, sell, or hold decisions based on technical indicators and market momentum.

## Overview
- **Environment**: Custom Gymnasium-based `StockTradingEnv`.
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable Baselines3.
- **Data**: Historical Nifty 50 stock data (2010–2019) fetched via `yfinance`.
- **Goal**: Maximize total portfolio returns while accounting for transaction costs and outperforming a "Buy & Hold" benchmark.

## Environment Architecture (`StockTradingEnv`)
The project utilizes a custom **Gymnasium** environment designed for multi-asset portfolio management. It simulates the constraints of a real-world trading desk:

* **State Space (Observation)**: The agent receives a concatenated vector containing:
    * **Market Features**: Normalized technical indicators (MACD, RSI, CCI, ADX) for all stocks.
    * **Portfolio Status**: The current cash balance and the number of shares held in each stock (scaled by `H_MAX`).
* **Action Space**: A continuous `Box(-1, 1, (n_stocks,))` space. Each value represents a trade decision for a specific stock:
    * **Positive (0 to 1)**: Buy shares (scaled to a maximum of `H_MAX` shares per step).
    * **Negative (-1 to 0)**: Sell existing shares.
* **Reward Function**: The reward is defined as the daily change in the **Net Asset Value (NAV)**.
    > $$Reward_t = PortfolioValue_t - PortfolioValue_{t-1}$$
    By focusing on the variance in total wealth rather than just price changes, the agent learns to account for the impact of its holdings and transaction costs simultaneously.
* **Constraints**: Includes a **0.1% transaction fee** on all trades to prevent high-frequency "churning" that would be unprofitable in real markets.



## Data Preprocessing & Normalization
Financial time-series data is non-stationary. To ensure the PPO model learns effectively, a **Rolling Window Normalization** strategy was implemented:

* **The Problem**: Absolute stock prices and indicator ranges (e.g., RSI vs. MACD) vary drastically across different stocks and time periods, leading to unstable training.
* **The Solution (`rolling_norm`)**: Instead of global scaling, a **50-day lookback window** is used to calculate local means and standard deviations.
* **Transformation**: Each feature $x$ is transformed into a Z-score:
    $$z = \frac{x - \mu_{window}}{\sigma_{window} + \epsilon}$$
    *where $\epsilon = 10^{-10}$ to prevent division by zero.*
* **Benefits**: 
    * **Stationarity**: Centers the data around zero, allowing the neural network to identify relative patterns regardless of the absolute price.
    * **Adaptability**: The model learns to react to recent volatility rather than being biased by price levels from years prior.

## Technical Walkthrough

### Feature Engineering
The model utilizes several technical indicators to capture market momentum:
- **MACD & ADX**: Used to identify the direction and strength of market trends.
- **RSI & CCI**: Used to identify overbought or oversold conditions.

### Agent Training
- **Policy**: `MlpPolicy` (Actor-Critic framework).
- **Training Duration**: 100,000 timesteps.
- **Benchmark**: A "Buy & Hold" strategy is used as a baseline, where the initial capital is distributed equally across all stocks at the start of the test period.

## Performance & Results
During the final evaluation on unseen data (2018–2019):
- **PPO Agent**: Successfully identified trend reversals and maintained a positive return.
- **Comparison**: The agent significantly outperformed the Buy & Hold benchmark during market downturns, showing a higher Sharpe Ratio and better risk management.

## Comments
- **Algorithm Choice**: PPO was selected over A2C and DDPG for its stability and robustness in continuous action spaces.
- **Realistic Simulation**: The inclusion of transaction costs was critical; without them, the agent developed "noisy" trading habits that failed in realistic backtesting.
