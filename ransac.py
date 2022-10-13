print("importing")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.indexes.base import InvalidIndexError
from sklearn import linear_model
from skimage.measure import LineModelND, ransac

print('reading')

df = pd.read_csv('ransac.csv')
df.to_csv('C:\Users\Varsha\Desktop\ransacc1.csv',index=False)
print('readfinish')

framno=np.array(df['frameno'])
conf=np.array(df['confidence'])
xy=np.column_stack((framno,conf))
ransacmodel,inliers=ransac(xy,LineModelND,min_samples=2,residual_threshold=0.05, max_trials=1000)
outliers = inliers == False
plt.xlabel('Frame no.')
plt.ylabel('Confidence Score')
line_x = np.arange(0, 250)
line_y_robust = ransacmodel.predict_y(line_x)


plt.plot(1, 2.5, '.w', alpha=0.6)
plt.plot(xy[inliers, 0], xy[inliers, 1], '.b', alpha=0.6,label='Inlier data')
plt.plot(xy[outliers, 0], xy[outliers, 1], '.r', alpha=0.6,label='Outlier data')
df1=pd.DataFrame(outliers)
with pd.ExcelWriter('xlout.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='Sheet1')
    
plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
plt.legend(loc='upper right')

print(xy[inliers])

plt.show()
