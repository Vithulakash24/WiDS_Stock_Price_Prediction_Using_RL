# Multi-Stock Trading with Reinforcement Learning (PPO)

This project implements a custom **Multi-Stock Trading Environment** to train a Reinforcement Learning agent to manage a portfolio of Nifty 50 stocks. Using the **Proximal Policy Optimization (PPO)** algorithm, the agent learns to make buy, sell, or hold decisions based on technical indicators and market momentum.

## Directory Structure

| File/Folder | Description |
| :--- | :--- |
| `nifty_data_2010_2019/` | Raw historical CSV files for Nifty 50 constituent stocks. |
| `RL_proj_final.ipynb` | Main project notebook: contains environment logic, PPO training, and backtesting. |
| `processed_data.npy` | Feature-engineered dataset (includes Technical Indicators like MACD, RSI). Un-normalized |
| `normed_data.npy` | Feature-engineered dataset (includes Technical Indicators like MACD, RSI). Normalized |
| `normed_data.npy` | Final input data processed via **50-day rolling normalization**. |
| `ppo_nifty_model_v1.zip` | Trained weights for the Stable-Baselines3 PPO agent trained using raw data. |
| `ppo_model`|  Trained weights for the Stable-Baselines3 PPO agent trained using normed data. |
| `README.md` | Comprehensive project documentation. |

---

## Environment Architecture (`StockTradingEnv`)

The project utilizes a custom **Gymnasium** environment designed for multi-asset portfolio management. It simulates the constraints of a real-world trading desk:



* **State Space (Observation)**: The agent receives a concatenated vector containing:
    * **Market Features**: Normalized technical indicators (MACD, RSI, CCI, ADX) for all stocks.
    * **Portfolio Status**: The current cash balance and the number of shares held in each stock (scaled by `H_MAX`).
* **Action Space**: A continuous `Box(-1, 1, (n_stocks,))` space. Each value represents a trade decision for a specific stock:
    * **Positive (0 to 1)**: Buy shares (scaled to a maximum of `H_MAX` shares per step).
    * **Negative (-1 to 0)**: Sell existing shares.
* **Reward Function**: The reward is defined as the daily change in the **Net Asset Value (NAV)**:
    > $$Reward_t = PortfolioValue_t - PortfolioValue_{t-1}$$
    By focusing on the variance in total wealth rather than just price changes, the agent learns to account for the impact of its holdings and transaction costs simultaneously.
* **Constraints**: Includes a **0.1% transaction fee** on all trades to prevent high-frequency "churning" that would be unprofitable in real markets.

---

## Data Preprocessing & Normalization

Financial time-series data is non-stationary. To ensure the PPO model learns effectively, a **Rolling Window Normalization** strategy was implemented:

* **The Problem**: Absolute stock prices and indicator ranges (e.g., RSI vs. MACD) vary drastically across stocks and time, leading to unstable training gradients.
* **The Solution (`rolling_norm`)**: Instead of global scaling, a **50-day lookback window** calculates local means and standard deviations. This prevents data leakage (using future info to scale past data).
* **Transformation**: Each feature $x$ is transformed into a Z-score:
    $$z = \frac{x - \mu_{window}}{\sigma_{window} + \epsilon}$$
    *where $\epsilon = 10^{-10}$ to prevent division by zero.*

---

## Technical Walkthrough

### Feature Engineering
The model utilizes several technical indicators to capture market momentum:
- **Trend/Direction**: MACD (Moving Average Convergence Divergence) and ADX (Average Directional Index).
- **Overbought/Oversold**: RSI (Relative Strength Index) and CCI (Commodity Channel Index).

### Agent Training
- **Policy**: `MlpPolicy` (Actor-Critic framework) via `Stable Baselines3`.
- **Optimization**: PPO was finalized after testing various policy gradient methods, as it provided the most stable convergence in the volatile Nifty 50 environment.
- **Hyperparameters**: Tuned to balance exploration and exploitation, trained for 100,000 timesteps.

## Performance & Results
During evaluation on unseen data (2018–2019):
- **PPO Agent**: Demonstrated the ability to identify trend reversals and manage drawdowns effectively.
- **Benchmark Comparison**: The agent outperformed a standard **Buy & Hold** strategy (equal weight distribution) in terms of the Sharpe Ratio and risk-adjusted returns during market downturns.

## Comments
- I tried using various policy gradient methods ,PPO gave the best possible results so I finalized with PPO
- The hyperparameters were chosen after testing with different parameters
