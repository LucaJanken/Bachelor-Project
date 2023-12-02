import tensorflow as tf

class Spline_tri():
    def __init__(self,x,x_target):

        self.h = tf.experimental.numpy.diff(x)
        self.n = x.shape[0]
        self.a = tf.stack([self.h[i]/(self.h[i]+self.h[i+1]) for i in range(self.n-2)] + [0], axis=0)
        self.b = tf.scalar_mul(2,tf.ones(x.shape[0]))
        self.c = tf.stack([0]+[self.h[i+1]/(self.h[i]+self.h[i+1]) for i in range(self.n-2)], axis=0)
        self.y_res = tf.Variable(initial_value=tf.zeros((x_target.shape[0],)),dtype=tf.float32)
        self.coef = tf.Variable(initial_value=tf.zeros((x.shape[0]-1,4)),dtype=tf.float32)
        self.diagonals = (self.c,self.b,self.a)
        self.x = x
        self.x_target = x_target

        self.interval=tf.Variable(initial_value=tf.zeros((self.n-1,2),dtype=tf.int32))
        for i in tf.range(self.n-1):
            if tf.equal(i,0):
                q = tf.where(x_target < x[i+1])
            elif tf.equal(i,self.n-2):
                q = tf.where(x_target >= x[i])
            else:
                boolean_array = tf.logical_and(x_target >= x[i], x_target < x[i+1])
                q = tf.where(boolean_array)
            if q.shape[0] < 1:
                q0 = 0
                q1 = 0
            else:
                q0 = q[0,0]
                q1 = q[-1,0]+1
            self.interval[i,:].assign([q0,q1])

    @tf.function
    def do_spline(self, y):
        y_res = []
        yp = tf.pad(y[:, 1:  ], tf.constant([[0,0],[0,1]]), constant_values=1)
        ym = tf.pad(y[:,  :-1], tf.constant([[0,0],[1,0]]), constant_values=1)
        hp = tf.pad(self.h,  tf.constant([[0,1]]), constant_values=1)
        hm = tf.pad(self.h,  tf.constant([[1,0]]), constant_values=1)

        rhs = (yp-y)/hp - (y-ym)/hm
        rhs /= hp+hm
        rhs *= 6

        rhs = tf.pad(rhs[:, 1:-1],tf.constant([[0,0],[1,1]]))
        X = tf.transpose(tf.linalg.tridiagonal_solve(self.diagonals, tf.transpose(rhs), diagonals_format='sequence'))
        for i in range(self.n-1):
            C = tf.concat([[(X[:,i+1]-X[:,i])*self.h[i]*self.h[i]/6],
                            [X[:,i]*self.h[i]*self.h[i]/2],
                            [(y[:,i+1] - y[:,i] - (X[:,i+1]+2*X[:,i])*self.h[i]*self.h[i]/6)],
                            [y[:,i]]],0)
            z = tf.divide(tf.subtract(self.x_target[self.interval[i,0]:self.interval[i,1]],self.x[i]),self.h[i])
            Z = tf.concat([[tf.pow(z,3)],[tf.pow(z,2)],[tf.pow(z,1)],[tf.pow(z,0)]],0)
            y_res.append(tf.linalg.matmul(C,Z,transpose_a=True))
        return tf.concat(y_res,1)