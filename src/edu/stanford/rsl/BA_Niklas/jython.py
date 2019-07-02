import java.util.ArrayList as ArrayList
import math
import pdb
import numpy
import matplotlib.pyplot as plt


arr = ArrayList()
arr.add(10)
arr.add(20)
print("ArrayList:", arr)
arr.remove(10) #remove 10 from arraylist
arr.add(0,5) #add 5 at 0th index
print("ArrayList:", arr)
print("element at index 1:",arr.get(1)) #retrieve item at index 1
arr.set(0,100) #set item at 0th index to 100
print("ArrayList:", arr)
print(math.pow(5, 4))

plt.plot([3, 4, 5], [1, 2, 3], 'b-', label="points")
plt.show()