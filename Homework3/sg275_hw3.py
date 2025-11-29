import numpy as np
import math

# -------------------------------------------------
# McCormick function
# -------------------------------------------------
def f(x, y):
    return math.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

# -------------------------------------------------
# Gradient of f
# -------------------------------------------------
def grad_f(x, y):
    df_dx = math.cos(x + y) + 2*(x - y) - 1.5
    df_dy = math.cos(x + y) - 2*(x - y) + 2.5
    return np.array([df_dx, df_dy])

# -------------------------------------------------
# Hessian of f (for Newton’s method)
# -------------------------------------------------
def hessian_f(x, y):
    d2f_dx2 = -math.sin(x + y) + 2
    d2f_dy2 = -math.sin(x + y) + 2
    d2f_dxdy = -math.sin(x + y) - 2
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# -------------------------------------------------
# Gradient Descent with bold driver heuristic
# -------------------------------------------------
def gd_optimize(a):
    lr = 1.0
    prev_val = f(a[0], a[1])
    print(prev_val)
    
    while True:
        grad = grad_f(a[0], a[1])
        new_a = a - lr * grad
        new_val = f(new_a[0], new_a[1])
        
        print(new_val)
        
        if abs(new_val - prev_val) < 1e-20:
            print(new_a)
            break
        
        # Bold driver heuristic
        if new_val > prev_val:
            lr *= 0.5
        else:
            lr *= 1.1
            a = new_a
            prev_val = new_val

# -------------------------------------------------
# Newton’s Method
# -------------------------------------------------
def nm_optimize(a):
    prev_val = f(a[0], a[1])
    print(prev_val)
    
    while True:
        grad = grad_f(a[0], a[1])
        H = hessian_f(a[0], a[1])
        H_inv = np.linalg.inv(H)
        new_a = a - np.dot(H_inv, grad)
        new_val = f(new_a[0], new_a[1])
        
        print(new_val)
        
        if abs(new_val - prev_val) < 1e-20:
            print(new_a)
            break
        
        a = new_a
        prev_val = new_val

# -------------------------------------------------
# Run both methods as required
# -------------------------------------------------
if __name__ == "__main__":
    print(">>> gd_optimize(np.array([-0.2, -1.0]))")
    gd_optimize(np.array([-0.2, -1.0]))

    print("\n>>> gd_optimize(np.array([-0.5, -1.5]))")
    gd_optimize(np.array([-0.5, -1.5]))

    print("\n>>> nm_optimize(np.array([-0.2, -1.0]))")
    nm_optimize(np.array([-0.2, -1.0]))

    print("\n>>> nm_optimize(np.array([-0.5, -1.5]))")
    nm_optimize(np.array([-0.5, -1.5]))
