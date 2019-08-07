# DecisionTreeRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# reading of data

data = pd.read_csv("pos_sal.csv")
# print(data)

real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values
# print("it is train", real_x)
# print(real_y, "it is y ")

# for trainig
reg = DecisionTreeRegressor(random_state=0)
reg.fit(real_x, real_y)


# predictig

y_prd = reg.predict([[11]])
print( "your prediction is :-", y_prd)

# for ploting

x_grid = np.arange(min(real_x),max(real_x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(real_x,real_y, color = "green")
plt.plot(x_grid, reg.predict(x_grid), color = "blue")
plt.title("DecisionTreeRegressor")
plt.xlabel("position")
plt.ylabel("salaries")
plt.show()