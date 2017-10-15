import numpy as np
import tensorflow as tf
from rl_algs.a2c.utils import conv, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from rl_algs.common.distributions import make_pdtype

def normc_init(std=1.0, axis=0):
    """
    Initialize with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def fc(x, scope, nh, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope): #pylint: disable=E1129
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=normc_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
    def __init__(self, num_units, name, nin, rec_gate_init=0.):
        tf.nn.rnn_cell.RNNCell.__init__(self)
        self._num_units = num_units
        self.rec_gate_init = rec_gate_init
        self.name = name
        self.w1 = tf.get_variable(name + "w1", [nin+num_units, 2*num_units], initializer=normc_init(1.))
        self.b1 = tf.get_variable(name + "b1", [2*num_units], initializer=tf.constant_initializer(rec_gate_init))
        self.w2 = tf.get_variable(name + "w2", [nin+num_units, num_units], initializer=normc_init(1.))
        self.b2 = tf.get_variable(name + "b2", [num_units], initializer=tf.constant_initializer(0.))

    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        x, new = inputs
        while len(state.get_shape().as_list()) > len(new.get_shape().as_list()):
            new = tf.expand_dims(new,len(new.get_shape().as_list()))
        h = state * (1.0 - new)
        hx = tf.concat([h, x], axis=1)
        mr = tf.sigmoid(tf.matmul(hx, self.w1) + self.b1)
        # r: read strength. m: 'member strength
        m, r = tf.split(mr, 2, axis=1)
        rh_x = tf.concat([r * h, x], axis=1)
        htil = tf.tanh(tf.matmul(rh_x, self.w2) + self.b2)
        h = m * h + (1.0 - m) * htil
        return h, h

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, init_scale=0.01)
            vf = fc(h4, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class GRUPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps=1, memsize=1024, reuse=False, deterministic=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        if nsteps==1:
            ob_shape = (nbatch, nh, nw, nc)
            M = tf.placeholder(tf.float32, [nbatch], name='M')  # mask (done t-1)
        else:
            ob_shape = (nenv, nsteps, nh, nw, nc)
            M = tf.placeholder(tf.float32, [nenv, nsteps], name='M')  # mask (done t-1)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape, name='X') #obs
        S = tf.placeholder(tf.float32, [nenv, memsize], name='S') #states

        with tf.variable_scope("model", reuse=reuse):
            X_reshaped = tf.reshape(X, (nbatch, nh, nw, nc))
            h = conv(tf.cast(X_reshaped, tf.float32)/255., 'c1', nf=128, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=256, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=256, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=memsize, init_scale=np.sqrt(2))
            h5 = tf.reshape(h4, [nenv, nsteps, memsize])
            m = tf.reshape(M, [nenv, nsteps, 1])
            cell = GRUCell(memsize, 'gru1', nin=memsize)
            h6, snew = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False, initial_state=S)

            h7 = tf.reshape(h6, [nbatch, memsize])
            pi = fc(h7, 'pi', nact)
            vf = fc(h7, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        if deterministic:
            a0 = tf.argmax(pi, axis=1)
        else:
            a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, memsize), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.a = a0
        self.snew = snew
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.memsize = memsize

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value