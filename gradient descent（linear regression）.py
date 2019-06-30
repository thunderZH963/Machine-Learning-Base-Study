import numpy as np
import matplotlib.pyplot as plt
#y=2 * (x1) + (x2) + 3 
rate = 0.001#rate of gradient descent

#train sample
x_train = np.array([[1, 2],[2, 1],[2, 3],[3,5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7],[9,10],[1,9],[2,3],[10,4]])#train set
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26,25,23,45,33])

#test sample
x_test  = np.array([[1, 4], [2, 2],[2, 5],[5, 3],[1, 5],[4, 1]])#test set

#a b c （predicted parameter） 
a = np.random.normal()
b = np.random.normal()
c = np.random.normal()

#def linear formula
def h(x):
    return a*x[0]+b*x[1]+c

#main gradient descent
for i in range(10000):
    sum_a=0
    sum_b=0
    sum_c=0
    for x, y in zip(x_train, y_train):
        sum_a = sum_a + rate*(y-h(x))*x[0]
        sum_b = sum_b + rate*(y-h(x))*x[1]
        sum_c = sum_c + rate*(y-h(x))
    a = a + sum_a
    b = b + sum_b
    c = c + sum_c
    plt.plot([h(xi) for xi in x_test])

#print predicted paremeter

print("predicted parameter a:")
print(a)
print()
print("predicted parameter b:")
print(b)
print()
print("predicted parameter c:")
print(c)
print()
i=0
for xi in x_train:
    result=[h(xi)]
    i=i+1
    print("sample(train)"+str(i)+":")
    print(xi)
    print("result"+str(i)+";")
    print(result)

i=0
for xi in x_test:
    result=[h(xi)]
    i=i+1
    print("sample(test)"+str(i)+":")
    print(xi)
    print("result"+str(i)+";")
    print(result)

plt.show()

