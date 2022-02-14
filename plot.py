import matplotlib.pylab as plt

max = list()
min = list()
avg = list()

with open("result.txt", "r") as file:
    lines = file.readlines()

for i in lines:
    array = i.split(" ")
    max.append(float(array[0]))
    avg.append(float(array[1]))
    min.append(float(array[2]))


plt.plot(max, color='b', label="Maximum fitness value")
plt.plot(avg, color='r', label="Average fitness value")
plt.plot(min, color='g', label="Minimum fitness value")

plt.xlabel("Number of Generation")
plt.ylabel("Fitness Value")
plt.legend()
plt.show()
