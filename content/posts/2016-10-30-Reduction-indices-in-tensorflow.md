---
title: Reduction indices in tensorflow 
date: 2016-10-30
tags: ["tensorflow"]
---

TensorFlow provides many handy functions for performing reductions on tensors.
I was trying to compute max-pooling over some tensor. I did not understand
the idea of reduction indices correctly, and hence ended up writing for loops 
within the tensorflow computation graph. As expected the code ran poorly.

It was clear that using `tf.reduce_max` appropriately would be the key
to rectify the issue. Note that similar ideas apply to numpy arrays. I hope that
this write-up will be useful for future reference.

The reduction index is essentially that index on which the operation
runs. Also, note that the dimensions are indexed starting from 0.
Therefore, for a 2-D tensor, the rows are indexed as the 0 dimension
and columns are indexed as dimension 1.

Say we are given a 2-D tensor (`a`) and we want a tensor which contains the
sum of entries along the column. (rows are indexed by `i`
and columns are indexed by `j`). If we were to write this down as
a summation operation we would have a summation that runs over `i`.
Therefore, our sum would be given by

```
tf.reduce_sum(a, reduction_indices=[0])
```

Let us try to write up some code to understand this.

```python
import tensorflow as tf
import numpy as np

t1 = tf.Variable(np.reshape(np.arange(1,13), (4,3)))
# t1 is a tensor [[1,  2,  3],
#		  [4,  5,  6],
#		  [7,  8,  9],
#		  [10, 11, 12]]
t2 = tf.reduce_max(t1, reduction_indices=[0])
# t2 is the tensor [10, 11, 12]
t3 = tf.reduce_max(t1, reduction_indices=[1])
# t3 is the tensor [3, 6, 9, 12]

with tf.Session() as sess:
  tf.initialize_all_variables().run()
  print(sess.run(t2))
  # [10 11 12]
  print(sess.run(t3))
  # [3 6 9 12]
```

These ideas generalize to higher order tensors. It is worthwhile to
think about the right reshaping of the tensor and reduction along 
appropriate dimension(s).
