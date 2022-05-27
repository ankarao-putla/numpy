import numpy as np

x=np.array([[1,2,3,4],[5,6,7,8]])

'''
for i in np.nditer(x):
    print(i)
'''
#Initialization and indexing:
'''
#y=np.array([1,2,3,4]);  #y=np.arange(0,4);  #y=np.array([np.arange(4),np.arange[5,9]])
z=np.array([[0,1,2,3,4],[5,6,7,8,9]])
print(z[1,3]); print(z[1][3]) # Both indexings are same
#Fancy indexing..
newarray=np.array([[0,2],[1,3]]);
print(x[newarray])
print(x.ndim,x.shape,x.dtype, x.size, x.itemsize)
#x3=np.ones([2,2]); print(x3) # np.zeros([3,3])
'''

# Matrix slicing
'''
x=np.arange(21,40)
print(x[0:10:2])   #or newslice=slice(0:10:2); print(x[slice])
print(x[:])
y=np.array([np.arange(0,9),np.arange(10,19)])
print(y)
print(y[0:2,2:9])
'''



#Airthmetics on array
'''
print("each array element is added by 2 : ", x+2) #x-2,x*2,x/2,x%2,x//2, x**y can also be done on array
print("used both x and y array : ",x+y)           #x-y, x*y, x/y, x%y, x//y, x**y
np.add(x,y), np.subtract(x,y),np.multiply(x,y),np.divide(x,y),np.sqrt(x),np.log(x),np.pow(x,y)
np.sin(x), np.cos(y), np.tan(x),
 np.log(x) # np.e==1, log(-1)==nan, log(0)==-inf
# np.hypot(base,height)
#np.bitwise_and(x,y) # bitwise_or, bitwise_xor
#prod=np.dot(x,y);
'''

#Comparison
'''
#z=np.greater(x,y),np.greater_equal(x,y),np.less(x,y),np.less_equal(x,y),equal, not_equal
#print(np.mod(5,2));   print(np.mod(x,4))
#z=np.isnan(x)  #output will be list of true or false based on each element of x
#z=np.all(x), any(x)
#np.iscomplex(x), np.isreal(x)
'''

#y1=np.linspace(0,20,2); print(y1)  #2 means, makes 1 bar.. 3 means, makes 2 equal pars as 0,10,20




#z=x.copy() ;  z[1]=100;  print(z)



#Aggregate functions : sum(x),mean,max,min,std

#x=np.array([[True, False],[True,True]]) ; print(np.all(x,axis=0)); print(np.all(x, axis=1)) #on same array row for 1
#    -infinite +infinite and NaN evaluate to True as they are different from Zero


#Reshaping an array
'''
#z=np.arange(9).reshape(3,3) ; print(z)

x=np.array([[1,2,3,4],[5,6,7,8]])
newarray=np.reshape(x,(4,2))
print(newarray)

print(x.ravel()) # converting nDimensional array into 1D array
print(x.transpose())
'''
#splitting an array
'''
x=np.array([1,2,3,4,5,6,7,8])
#print(np.split(x,2))   # split works to make equal parts. Doesnt work to make unequal parts.. It is possible with np.array_split
#print(np.split(x,[1,3,5]))
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=0) # ==     newarr=np.array.vsplit(arr,3)
newarr2=np.array_split(arr,2,axis=1) # ==         newarr=np.array_split.hsplit(arr,2)
#print(newarr)
#print(newarr2)


x=np.array([0,1,2,3,4,5,6,7,8,9])
#print(np.array_split(x,3))   # array_split works even when it is not possible to make equal parts
'''

#Concatenation (works on same size matrices)
'''
arr1=np.array([[1,2,3],[4,5,6]]);  arr2=np.array([[7,8,9],[11,22,33]])
print(np.concatenate((arr1,arr2)))          # concatenates 2nd array at the bottom
print(np.concatenate((arr1,arr2), axis=0))  # concatenates 2nd array at the bottom
print(np.concatenate((arr1,arr2), axis=1))  # concatenates 2nd array at right side of 1st array
print(np.concatenate((arr1,arr2), axis=None)) # concatenates both the arrays in 1-D

print(np.concatenate((arr1.T, arr2.T),axis=1))
'''

#stack
'''
arr1=np.array([1,2,3]); arr2=np.array([4,5,6])
print(np.stack((arr1,arr2)))  # like normal matrix
print(np.stack((arr1,arr2),axis=0)) # like normal matrix
print(np.stack((arr1,arr2), axis=1)) #  like in shape of transpose to normal matrix

print(np.column_stack((arr1,arr2)))  #  same as above
print(np.vstack((arr1,arr2))) # like normal matrix
print(np.hstack((arr1,arr2)))  # 1D list
'''

#fileoperations in numpy
'''fp=open("numpyfile.npy",'wb+')
arr=np.array([1,2,3,4])
np.save(fp,arr)
fp.seek(0)
newarr=np.load(fp)
print(newarr)
'''
import os
arr=np.array([1,2,3,4,5])
np.savetxt("newtxt.txt",arr,delimiter=',',fmt='%d')
newarr=np.genfromtxt("newtxt.txt",delimiter=',' )
print(newarr)
print(os.getcwd())
#'''