import numpy as np


n = 0

# arr = np.zeros(shape=({},{}.format(n,n)))
print("1.Create a function which creates an n×n array with (i,j)-entry equal to i+j.")
n = input("Enter n >> ")
n = int(n)
a = []   
def create(n):
    for i in range(0,n):
        for j in range(0,n):
            a.append(i+j)
    
create(n) 
a = np.array(a)
a = a.reshape(n,n)
print(a)           
print("2.Create a numpy array which contains odd numbers below 20. Arrange it to a 2x5 matrix. Compute the log of each element.")
odd = np.arange(1,20,2)
odd = odd.reshape(2,5)
print(odd)
print("Log of each element")
print(np.log(odd))
print("3.Create a function which creates an n×n random array. Subtract the average of each row of the matrix ")
def avg_each_row():
    m = input("Enter n =")
    m = int(m)
    lim = m*m
    ar = np.random.randint(0,100,lim)
    ar = ar.reshape(m,m)
    print("before")
    print(ar)
    np.set_printoptions(precision=6, suppress=True)
    row_means = np.mean(ar, axis=1)
    for i in range(0,m):
        ar[i]=ar[i]-row_means[i]
    np.set_printoptions(precision=6, suppress=True)
    print("After subtraction")
    print(ar)

avg_each_row()

print("4.Create a function which creates an n×n random array. Write a program to find the nearest value from a given value in the array.")
m = input("Enter n =")
m = int(m)
lim = m*m
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

array = np.random.random(lim)
arr = array.reshape(m,m)
print(arr)


value = input("Enter value = ")
value = float(value)
print(find_nearest(array, value))

print("5.Write a function to check if two random arrays are equal or not. ")
flag = 0
p = int(input("Enter the size of 1st array = "))
q = int(input("Enter the size of 2nd array = "))

def diff(p_a,q_a):
    if np.array_equal(p_a,q_a):
        print("Arrays are equal")
    else:
        print("Arrays are not equal")

p_a = np.random.random(p)
q_a = np.random.random(q)
diff(p_a,q_a)

print("6.Create a function to get the n largest values of an array.")

m = input("Enter size of array =")
m = int(m)

a = np.random.randint(1,100,m)
print(a)
def large(a,k):
    a[::-1].sort()
    for i in range(0,k):
        print(a[i])
k = int(input("Enter the number of lar_val >>"))
large(a,k)