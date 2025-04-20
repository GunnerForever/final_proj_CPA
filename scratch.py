# I make a scratch file to test any small thing, like if an argument does what I think it does
import numpy as np
#import tensorflow as tf

a = np.array([[1,1,1], [2, 3, 4]])
print(np.sum(a, 0))
print(np.sum(a, 1))