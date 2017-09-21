import random
import numpy as np

# x = [[1,1],[2,2],[3,3],[4,4],[5,5]]
# y = random.sample([i for i in range(5)], 3)
# t = set([i for i in range(5)])
# print(y)
# x = np.array( x )
# print(x[y])
# t1 = set(y)
# t = t - t1
# t = list(t)
# print(t)
# print( y)
# print(x[t])

# BatchSize = 30
# SampleLen = 632
# TrainLen = 316
# TestLen = 316
# ImageHeight = 128
# ImageWidth = 48
# ImageChannel = 3

# train_index = random.sample([ i for i in range(SampleLen)],TrainLen)
# test_index = list( set( [ i for i in range(SampleLen) ] ) - set( train_index ) )

# print( [i for i in range( SampleLen )] == ( list( set(train_index) | set(test_index) )) )

# a = np.array( [ [1,2,3],[1,2,3] ] )
# print( sum( a ) )
a = [ i for i in range(20)]
print(a)
print( [ a[i] for i in range( 0, 20, 5 )] )