import numpy as np
import matplotlib.pyplot as plt
# Known data points
X = np.array([-2, -1, 1, 2], dtype=np.float32)
Y = np.array([6, 4, 0, -2], dtype=np.float32)
# Graph of the known data points
fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.axhline(color='lightgray')
ax.axvline(color='lightgray')

# model prediction
def forward(w, x):
    return w * x
# Start with a random weight.
w=0.5

# Calculate predicted y values
y_predicted = forward(w, X)
print(y_predicted)

# Graph actual vs prediction
plt.figure()
ax = plt.subplot()
ax.scatter(X, Y)
ax.plot(X, y_predicted, 'green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.axhline(color='lightgray')
ax.axvline(color='lightgray')

# MSE
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

# Pick a bunch of random weights
W=[-0.5, 0, 0.5, 1.5, 2, 2.5]

# List store the calculated losses
L=[]

# Calculate loss for each w
for w in W:
    y_predicted = forward(w, X)
    L.append(loss(Y, y_predicted))

print(W)
print(L)

# Graph loss with respect to weight
plt.figure()
ax = plt.subplot()
ax.set_xlabel('weight')
ax.set_ylabel('loss')
ax.set_ylim(-1, 8)
ax.scatter(W, L)

# gradient of loss wrt weight
def gradient_dl_dw(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
# Pick a bunch of random weights
W=[-0.5, 0, 0.5, 1.5, 2, 2.5]

# List store the calculated losses
L=[]

# List store the calculated gradients
G=[]

# Calculate loss and gradient for each w
for w in W:
    y_predicted = forward(w, X)
    L.append(loss(Y, y_predicted))
    G.append(gradient_dl_dw(X, Y, y_predicted))

print(G)

plt.figure()
ax = plt.subplot()
ax.set_xlabel('weight')
ax.set_ylabel('loss')
ax.set_ylim(-1, 8)
ax.scatter(W, L)

# Add gradient labels next to each point
for i, g in enumerate(G):
    plt.text(W[i]+.05, L[i]+.05, g, fontsize=10)

def gradient_dl_db(x, y, y_predicted):
    return np.dot(2, y_predicted-y).mean()
# Training
learning_rate = 0.01
epochs = 500
def forward(w, x, b):
    return w * x+b
w = 10
b = 10
for epoch in range(epochs):
    # forward pass
    # calculate predictions
    y_predicted = forward(w, X, b)

    # calculate losses
    l = loss(Y, y_predicted)

    # backpropagation
    # calculate gradients
    dw = gradient_dl_dw(X,Y, y_predicted)

    db = gradient_dl_db(X, Y, y_predicted)

    # gradient descent
    # update weights
    w -= learning_rate * dw

    b -= learning_rate * db

    # print info
    if(epoch % 1==0):
        print(f'epoch {epoch+1}: w={w:.3f}, b={b:.3f}, loss={l:0.8f}, dw={dw:.3f}, forward(10)={forward(w,10,b):0.3f}')


plt.show()