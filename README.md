This is my code for a Data Science hackathon challenge that I participated for a one week in September 2025.
Link to the challenge: https://www.aicrowd.com/challenges/flextrack-challenge-2025/

Short description:
- We're tasked to build a multi-classes classification model to predict demand response flags at different time point: -1, 0, +1.
- Then based on demand response flag feature, create a regression model to predict demand response capacity during an event compared to baseline.

Modeling Approach:
- Preprocessing for both clf and regression models:
  - perform feature engineerings by creating more features from raw data: seasonality and lags features
  - perform NaN check, data transformation by stacking Yeo-Johnson and standardization due to zero-heavy in some of the features
- For classification task:
  - Built a 3-classes neural network. NN has about 3-4 hidden layers
  - NN architecture includes batchnorm and dropout for regularization
  - Parameters such as: learning rate, batch size, early stopping, learning scheduling, etc are carefully tune
  - Use focal loss (instead of cross-entropy loss) to pay more attention to data belonging to minority classes.
- For regression task:
  - Built xgboost models to learn demand response capacity.
  - Challenges:
    - There are many zeros in the target column. Therefore, we build a quick xgboost classifier model to classify zero vs. non-zero targets.
    - Then regression model to predict the real target, demand response capacity.
    - Our prediction: $y_{hat} = P_{nonzero} \times y_{pred}$ where $P_{nonzero}$ can be obtained from clf and $y_{pred}$ can be obtained from regression models.
- Overall prediction pipeline for scoring data:
  - Use NN to predict demand response flag.
  - Use demand response flag as a feature for second stage prediction
  - Use XGBClassifier of second stage to predict $P_{nonzero}$
  - Use XGBRegressor of second stage to predict $y_{pred}$
  - Overall prediction: $y_{hat} = P_{nonzero} \times y_{pred}$
 
Concluding remarks:
- Our multi-classes NN performance is very well on scoring data. We achieved high geometric mean score.
- Our regression modeling was not very good, compared to other participants. Our highest MAE score was 1.7, compared to 1.1x which was the best score at the time we stopped.
