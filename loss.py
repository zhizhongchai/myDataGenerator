def tversky_loss():
    def loss(y_true, y_pred):
        alpha = 0.5
        beta = 0.5
        ones = tf.cast(tf.greater(y_true, -1), tf.float32)

        # ones = tf.ones(shape)
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = K.sum(p0 * g0, (0, 1, 2))
        den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

        T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl - T

    return loss
