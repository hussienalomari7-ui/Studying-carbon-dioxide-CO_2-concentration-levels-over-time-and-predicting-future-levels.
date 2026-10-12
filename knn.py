# بسم الله

import numpy as np 
import pandas as pd 

# step one 

data = pd.read_csv("data_co2.csv")  # read data file

print(data.head()) # to ensure the file is opened and read 
print("shape of data :",data.shape) # 813 row , 8 colum 
print("name of data : ",data.columns)
print("number of missing values in each column : ", data.isna().sum()) # no missing data , you dont't need to clean it 

# step tow 
print("_" * 50)
import matplotlib.pyplot as plt 

plt.plot(data["year"] , data["monthly_average"]) # titancart -> X=year , y=monthly_avarge , show relashimship between them 
plt.xlabel("Year") # name of horizontal axis 
plt.ylabel("C02 concentration (ppm)") # name of vertical axis 
plt.title("Change in C02 concentration over time ") # title of shape 
# plt.show() # show result ( in month )


annual_co2 =data.groupby("year")["monthly_average"].mean() # calculate the annual average
# print(annual_co2.head()) # to verify

plt.plot(annual_co2.index , annual_co2.values) # index = number of year , values = avareg of monthes in this year
plt.xlabel("Year ") # name x 
plt.ylabel("Avverage CO2 concentration (ppm) ") # name y
plt.title("Annual Average CO2 Concentration Over Time ") # title the print
#رplt.show() # show the result 

print("_" * 50)

from sklearn.linear_model import LinearRegression 

annual_co2 =data.groupby("year")["monthly_average"].mean() # calculate the annual average , convert frome month data to year data

# groupby("year") : by grouping together all the classes of the same year 
# ["monthly_average"].mean(): calculate mean onle for monthly_average colum 

X = annual_co2.index.values.reshape(-1,1) # year=[1958,1959,...2023,2024]
y = annual_co2.values # value of co2 in year 

model = LinearRegression() # creat a model ready for training 

model.fit(X,y) # the best possible straight line is calculated to approximate the relationship between the year and co2 

future_years =np.arange(annual_co2.index.max()+1 , annual_co2.index.max() + 11).reshape(-1,1)
# annual_co2.index.max()+1 :the last year in the data returned and one is added to until it equals the cuuenr year . -> START
# annual_co2.index.max() + 11 : ten years more than the current year . -> END 
# reshape(-1,1) : form requirements 

future_co2 = model.predict(future_years)
# he takes the new years and applies the equation he learned to them . -> co2 = a.(new_year)+ b 


plt.plot(X , y , label =" Actual data ") # it plots the actual data (solid line )
plt.plot(future_years , future_co2 , "--"  , label="Prediction" ) # he draws the predictions with a dashed line (--)
plt.xlabel("Year") # name axis x
plt.ylabel("CO2 conaentration (ppm)" ) # name axis y 
plt.title("CO2 Prediction Using Linear Regresion ") # name title of curve
plt.legend() # definition box ( actual vs prediction )
# plt.show() 

print("_" * 50)
# prediction using an KNN algorithm :
from sklearn.neighbors import KNeighborsRegressor

annual_co2 =data.groupby("year")["monthly_average"].mean() # calculate the annual average , convert frome month data to year data
X = annual_co2.index.values.reshape(-1,1) # year=[1958,1959,...2023,2024] -> (fetuures)
y = annual_co2.values # value of co2 in year ->(target)

knn_model = KNeighborsRegressor(n_neighbors=3) # building prediction model that fpllows the knn algorithem
# abd has 3 neighbors 

knn_model.fit(X, y ) # save a data to work on it 

future_year26 = np.array([[2026]]) # the model is accepted in 24 
predected_co2 = knn_model.predict(future_year26) # knn rules apply 
print("Predicted CO@ concentration for 2026 : " , predected_co2[0] ," pmm") # print result 

plt.plot( X , y ,label="Actual data" ) # 
plt.scatter(future_year26 , predected_co2 , color="red" , label=" KNN Prediction (2026) ")  # draw a red dot to represent the score location 
plt.xlabel("Year")
plt.ylabel("CO2 Prediction (ppm) ")
plt.title("CO2 Prediction for 2026 Using KNN ")
plt.legend() # information box 
#plt.show()

print("_" * 50)
annual_co2 =data.groupby("year")["monthly_average"].mean() # calculate the annual average , convert frome month data to year data

co2_before_2000 = annual_co2[annual_co2.index<2000] # years before 2000
co2_after_2000 = annual_co2[annual_co2.index>=2000] # years after 2000 

meaan_before_2000 = co2_before_2000.mean()
mean_after_2000 = co2_after_2000.mean()

print( "Avarege CO2 before 2000 :" ,round(meaan_before_2000 ,2), "ppm" )
print( "Avarege CO2 after 2000 : " , round(mean_after_2000,2) , "ppm" )

periods = [ "Before 2000 " , "After 2000" ]
means = [ meaan_before_2000 , mean_after_2000]

plt.figure() # new drwe 
plt.bar(periods , means)
plt.xlabel("Time Period ")
plt.ylabel( "Average CO2 concentration (ppm)" )
plt.title("Comparison of Average CO2 Levels Before and After 2000 ")
plt.show()