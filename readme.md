# One Zone Can Be Charged! CEEMDAN+ADGAT-GGRU+BRR Component-Specific Modeling, New Framework for Lithium Battery Capacity Degradation RUL Prediction

\tPreface
Based on the experimental data of lithium battery from NASA, this paper proposes an innovative model CEEMDAN + ADGAT-GGRU + BRR for predicting the RUL of lithium ion batteries.
\t  
The innovative model first decomposes the original capacity degradation sequence into different scale IMFs through \tCEEMDAN\t, achieving explicit decoupling of complex fluctuations and long-term trends. Subsequently, an adaptive dynamic graph attention network is constructed for the \tIMF1–IMF3\t complex components, treating different \tIMF\t as cross-scale graph nodes and dynamically learning node interdependencies based on each cyclic stage; on this basis, \tGraph-Gated GRU\t is further designed to directly inject graph context into the reset gate, update gate, and candidate state of the \tGRU\t, enabling cross-scale correlations to participate in temporal memory updating. For the low-frequency smooth trend IMF4, Bayesian Ridge Regression (BRR) is adopted, obtaining stable long-term trend predictions under small sample conditions through Bayesian regularization. Finally, the prediction results from each component are directly reconstructed into a capacity curve, which can be combined with \tEOL\t threshold calculations for \tRUL\t.
\t  
We also provide detailed materials and video explanations, including tutorials on parameter adjustments, replacement of prediction tasks, and detailed code comments with parameter introductions.
● Dataset: NASA Lithium-Ion Battery Life Test Public Dataset
● Environment Framework: Python 3.11, PyTorch 2.1 and above can be run.
● Model score: 0.99 on test set
● Usage: Paper requirements, graduation design requirements.
● Code Guarantee: The code comments are detailed and ready to run.
1. Introduction to the innovative model
1.1 Model Architecture
This proposal introduces a hybrid prediction framework based on CEEMDAN + ADGAT-GGRU + BRR. The core idea is to perform component-specific modeling (component specificity modeling) according to the statistical characteristics of different components and the difficulty of prediction:
1. CEEMDAN: This method decomposes the original capacity sequence into different time scales, reducing the difficulty of modeling non-stationary degraded signals.
2. ADGAT-GGRU: It focuses on learning the dynamic dependencies between different IMFs (Intrinsic Mode Functions) as they change with cyclical stages, as well as the short/long-term temporal correlation of complex fluctuations.
3. BRR (Bayesian Ridge Regression): It is responsible for smoothing the trend component of IMF4, achieving robust trend prediction under small sample conditions with lower model complexity.
4. Component reconstruction: By adding the predictions of each component, the final capacity prediction curve is restored, which can be further used to calculate the battery RUL.
1.2 Model innovation points
Innovation point 1: Constructing IMFs as dynamic cross-scale graphs
Traditional GRU/LSTM usually directly treats IMF1–IMF3 as three ordinary input features, assuming that the relationship between different features is uniformly encoded by fixed network weights. However, in the degradation process of lithium batteries, the correlation degree among fluctuations at different scales may change with cyclical stages. This proposal treats each complex IMF as a graph node, and recalculates attention between nodes based on the current node state at each time step, thus forming Adaptive Dynamic Graph Attention (ADAGAT).
2. Innovation Point: Dynamic graph attention is not only used for feature extraction but also participates in GRU gate control.
This solution employs Graph-Gated GRU (Graph-Gated Recurrent Unit): first, multiple IMF nodes are aggregated into a graph context; then, the graph context is directly injected into the reset gate, update gate, and candidate state of the GRU. This design forms a complete modeling chain: IMF → state change → dynamic graph relationship changes → graph context changes → GRU → gating changes → complex component prediction changes. Therefore, ADGAT and GGRU are not independent modules but structurally coupled at the gating level.
\t  
Innovation Point 3: Heterogeneous Prediction and Explanable Reconstruction
ADGAT-GGRU outputs the prediction results of three complex IMFs, while BRR outputs the trend prediction results of IMF4. The final capacity is directly reconstructed through the additive property of CEEMDAN, which does not introduce a black-box fusion network or require human-set complex weight coefficients. Consequently, each part of error can be traced back to the corresponding IMF, offering good module interpretability and ease of debugging. Therefore, the core output of this model is firstly the high-precision capacity degradation trajectory, and RUL is further determined by the prediction trajectory and failure threshold.
2 Data Preprocessing
2.1 Dataset Introduction
\t  
2.2 Characteristics of CEEMDAN Decomposition

This paper obtained the original dataset for lithium battery aging from NASA's Ames Research Center. Four batteries, labeled B0005, B0006, B0007, and B0018, were selected to validate the effectiveness of this method. All battery tests were conducted under a constant environmental temperature of 24°C. The specific charging and discharging schemes are as follows: during the charging phase, a constant current (CC) of 1.5A was initially applied until the battery voltage reached 4.2V, then switched to constant voltage (CV) mode until the charging current decreased to 20 mA, indicating the end of charging. The discharge process included a constant current (CC) of 2 A until the battery voltage dropped to each battery's specific cut-off voltage: B0005 at 2.7V, B0006 and B0018 at 2.5V, B0007 at 2.2V. Table 1 also lists other key parameters, including discharge current (DC), discharge voltage (DV), and battery failure threshold (TS). This study utilized the capacity data from the original dataset for experimental verification. The termination of battery life (EOL) is defined as when the battery capacity drops to 70% of its rated capacity (1.4ah). However, for battery B0007, since it did not reach 1.4ah, EOL was set at 72% of the rated capacity (1.45ah). As shown in Figure 2, the capacity degradation trend of NASA lithium batteries is displayed.
This paper directly applies the CEEMDAN algorithm to decompose lithium-ion battery capacity data, obtaining multiple IMFs. Compared with the decomposition of lithium-ion battery current and voltage data, this method has faster processing speed, higher accuracy, and lower computational complexity. The selection of mode number k significantly affects the decomposition performance. Figures show the IMFs obtained by decomposing NASA lithium-ion battery capacity data using the CEEMDAN algorithm. In the figure, each IMF displays different characteristics of lithium-ion battery data. High-frequency IMFs (IMF1 to IMF3) mainly capture short-term changes and rapid fluctuations, although similar to random noise, they can still provide insights into subtle changes in battery performance. IMF4 represents the main trend, reflecting the overall capacity decline over time. This decomposition allows for clear analysis, distinguishing instantaneous fluctuations, periodic characteristics, and major degradation trends.
2.3 Data segmentation and preprocessing
Detailed explanations are provided in the document provided!
3 Prediction Model Based on CEEMDAN+ADGAT-GGRU+BRR
3.1 Importing data
Taking the B0005 dataset as an example, data preprocessing is carried out:
3.2 Setting parameters and training the model
3.3 Model evaluation
(1) Model indicator evaluation
(2) Prediction effect demonstration
The prediction effect of IMF4 component using BBR:
Overall combined prediction effect:
In the current B0005 experimental results, the combination model reconstructs capacity prediction achieves R2=0.990604, RMSE = 0.007072 A·h, MAE = 0.005260 A·h. The entire scheme integrates signal decomposition, dynamic graph modeling, gate-controlled timing learning, lightweight trend prediction, and interpretable reconstruction, suitable for lithium-ion battery SOH/RUL prediction, PHM method research, graduation design, and related scientific research code secondary development.
Click to download: Complete original data, Python codes
Previously excellent content:
Time series forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed prediction (I) Data introduction and preprocessing
Wind speed prediction (II) EMD-LSTM model based on Pytorch
Wind speed prediction (III) EMD-LSTM-Attention model
Wind speed prediction (IV) EMD-Transformer model based on Pytorch
Wind speed prediction (V) EMD-CNN-LSTM model based on Pytorch
Wind speed prediction (VI) EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + combined forecasting model (BiLSTM-Attention + ARIMA)
CEEMDAN + combined forecasting model (CNN-LSTM + ARIMA)
CEEMDAN + combined forecasting model (Transformer - BiLSTM + ARIMA)
CEEMDAN + combined forecasting model (CNN-Transformer + ARIMA)
Multi-feature variable sequence prediction (I) - CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (II) - CNN-LSTM-Attention wind speed prediction model
Multi-feature variable sequence prediction (III) - CNN-Transformer wind speed prediction model
Multi-feature variable sequence prediction (IV) Transformer-BiLSTM wind speed prediction model
Multi-feature variable sequence prediction (V) CEEMDAN+CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (VI) CEEMDAN+CNN-Transformer wind speed prediction model
Prediction model based on SSA optimized by Sparrow optimization algorithm for CEEMDAN-BiLSTM-Attention
Based on the Particle Swarm Optimization Algorithm (SSA) for CEEMDAN-Transformer-BiGRU Prediction Model
Superior Prediction Model: Secondary Decomposition-Combination Prediction
Multi-Feature Variable Sequence Prediction (Part 7): CEEMDAN+Transformer-BiLSTM Prediction Model
Multi-Feature Variable Sequence Prediction (Part 8): CEEMDAN-SSA-BiLSTM Prediction Model Based on SSA
Multi-Feature Variable Sequence Prediction (Part 11): TCN-GRU Prediction Model Based on Pytorch
VMD + CEEMDAN: Secondary Decomposition, CNN-LSTM Prediction Model
VMD + CEEMDAN: Secondary Decomposition, CNN-Transformer Prediction Model
VMD + CEEMDAN: Secondary Decomposition, Transformer-BiGRU Prediction Model
Prediction Model Based on SSA - Code All-in-One Kit
| Superior Combination Prediction Model!
High Innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention Prediction Model
| Superior Combination Prediction Model!
Lowest Price Online | Continuous Updates for the All-in-One Kit!
| Based on TCN-SENet + BiGRU-GlobalAttention Parallel Prediction Model
Secondary Decomposition - Innovative Model Prediction All-in-One Kit
| BiTCN-BiGRU-CrossAttention Fusion Spatiotemporal Features High Innovation Prediction Model
Update to Combination Prediction | Superior Combination Prediction Model!
Spatiotemporal Feature Fusion BiTCN-Transformer Parallel Prediction Model
| Based on Multi-Level Attention Mechanism Parallel Prediction Model
| CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Prediction
Multi-step Prediction Series | Collection of LSTM, CNN, Transformer, TCN, Serial, Parallel Models
END
2. Friends who like this can click to follow, give the article a "like", share on WeChat Moments or discuss in groups.

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
