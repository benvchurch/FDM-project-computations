import numpy as np

def trapz2d(z, x = None,y = None):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule. 
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    
    sum = np.sum
    dx = (x[-1]-x[0])/(np.shape(x)[0]-1)
    dy = (y[-1]-y[0])/(np.shape(y)[0]-1)    
    
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = sum(z[1:-1,0]) + sum(z[1:-1,-1]) + sum(z[0,1:-1]) + sum(z[-1,1:-1])
    s3 = sum(z[1:-1,1:-1])
    
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)

