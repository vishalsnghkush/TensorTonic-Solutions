def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0

    for i in range(steps):
        slope = 2 * a * x + b   
        x = x - lr * slope      

    return x