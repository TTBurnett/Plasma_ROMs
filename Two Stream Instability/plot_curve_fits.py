import re
import numpy as np
import matplotlib.pyplot as plt

class Line:
    def __init__(self, a, b, name):
        self.a = abs(float(a))
        self.b = float(b)
        self.name = name

    def xy(self, x_min, x_max):
        x = np.array([x_min, x_max])
        y = self.a*x + self.b
        return x, y

with open('ROM_results.txt') as f:
    text = f.read()
coefficients = re.findall(r'\[.*\]', text)
names = re.findall(r'---.*---', text)

errors = []
hamiltonians = []
for i, c in enumerate(coefficients):
    s = re.sub(r'\[|\]', '', coefficients[i])
    name = re.sub(r'-', '', names[i//2])
    if re.search(r'POD Stacked', name) and int(re.findall(r'\d+', name)[0]) > 2:
        continue
    a, b = s.split()
    if i & 1:
        hamiltonians.append(Line(a, b, name))
    else:
        errors.append(Line(a, b, name))

for e in errors:
    x, y = e.xy(0, 1000)
    plt.plot(x, y, label=e.name)
plt.xlabel('Time (s)')
plt.ylabel('Relative Error')
plt.legend()
plt.show()

for h in hamiltonians:
    x, y = h.xy(0, 1000)
    plt.plot(x, y, label=h.name)
plt.xlabel('Time (s)')
plt.ylabel('|Relative Hamiltonian|')
plt.legend()
plt.show()