If you read the code — especially Open.ipynb — you’ll notice that it’s a rather monolithic attempt to produce models and results in order to meet a tight deadline.
It needs some decomposition and a more structured, diagrammatic design.

To reduce the risk of data leakage, the datasets are separated.
The code still needs further investigation to determine whether there is any overfitting or underfitting.

There is plenty of code available on the internet, but many examples suffer from data leakage and treat prices as parabolic data.
Even when such models appear to perform well, they aren’t truly learning — they’re merely keeping up with the data.

The essence of this work is to capture the frequency of the data.
To achieve that, I created a Direction column to help guide the model’s directional predictions.

In the results, you may notice some irregular patterns, especially in the Direction Accuracy and Direction Prediction Loss metrics — this is due to the nature of the data itself.
The most important aspect when dealing with frequently fluctuating data is to predict the vectoral, not the scalar value.

If you look closely at stock prices from 10 or 20 years ago, you’ll notice that many stocks move upward or downward over time due to various factors.
Models trained on this type of data tend to develop a bias toward upward predictions, which becomes even worse in multi-step predictions
because of the cumulative effect and the inherent nature of the data.

Because of these reasons, while Open.ipynb works well for OpenChange predictions, it becomes almost useless when reconfigured for CloseChange.
This is mainly because Open prices are generally more stable and less volatile than CloseChange values.

To improve CloseChange predictions, I developed several smaller models — and interestingly, they performed better than the original OpenChange model.

The yDatas module follows a classic structure, but it also needs some adjustments to support daily data updates more effectively.

Good Luck.
