import math
import numpy as np
import tensorflow as tf

PC_CELL_X_SIZE = 1.0

class Weights():
    def __init__(self, dim_th, dim_xy):
        self.dim_th = dim_th
        self.dim_xy = dim_xy
        self.mid_th = self.dim_th / 2;
        self.mid_xy = self.dim_xy / 2;
        self.shape  = [self.dim_th, self.dim_xy, self.dim_xy]

        self.size_th = (2.0 * math.pi) / self.dim_th

        self.vrot   = tf.placeholder(tf.float32, shape=[], name="vrotInput")
        self.vtrans = tf.placeholder(tf.float32, shape=[], name="vtransInput")

    def norm2d(self, var, x, y, z):
        """
        Generate 2D PDF
        """
        return 1.0 / (var * math.sqrt(2.0 * math.pi)) \
            * math.exp(( \
                - (x - self.mid_xy) * (x - self.mid_xy) \
                - (y - self.mid_xy) * (y - self.mid_xy) \
                - (z - self.mid_th) * (z - self.mid_th) \
            ) / (2.0 * var * var))

    def pdf(self, var):
        """
        Generate and normalise 3D PDF
        """
        total = 0
        res = []

        for k in xrange(self.dim_th):
            for j in xrange(self.dim_xy):
                for i in xrange(self.dim_xy):
                    norm = self.norm2d(var, i, j, k)
                    res.append(norm)
                    total += norm

        for x in xrange(len(res)):
            res[x] /= total

        return res

    def attractor(self, var):
        """
        Weights for each PoseCell based on PDF
        """
        print('Generating PDF weights...')
        pdf = self.pdf(var)
        res = []
        for k in xrange(self.dim_th):
            res.append([])
            for j in xrange(self.dim_xy):
                res[k].append([])
                for i in xrange(self.dim_xy):
                    # TODO: Sparsify, indexes and values, over a min limit?
                    res[k][j].append(self.displace_pdf(pdf, k, j, i))

        return np.asarray(res)

    def displace_pdf(self, pdf, x, y, z):
        """
        Roll the PDF until aligned with this cell
        """
        # Fully connected
        # TODO: Locally connected "Activity Packets"?
        # [[left, most, corner], [smaller, p, d, f]]
        return np.reshape(np.roll(
            pdf,
            ((x-self.mid_xy) * self.dim_xy * self.dim_xy) + ((y-self.mid_xy) * self.dim_xy) + (z-self.mid_th)),
            self.shape
        )

    def translation(self):
        """
        Generating Translation weights
        """
        print('Generating Translation weights...')
        val = []
        idx = []
        cnt = 0

        # TODO: Scale velocity by size of time step (not really applicable to DQN agents which step evenly)
        # `vtrans = vtrans * time_diff_s;`
        # `vrot = vrot * time_diff_s;`

        vtranscell = tf.divide(self.vtrans, PC_CELL_X_SIZE)

        # Z axis
        for k in xrange(self.dim_th):
            trans_dir = k * self.size_th #th index * size of each th in rad + angle_to_add (for reverse)

            # Break down to one single quadrant
            # `math.floor(trans_dir * 2 / math.pi)` gives number of quadrant.

            trans_quad = math.floor(trans_dir * 2 / math.pi)
            trans_dir = trans_dir - trans_quad * math.pi / 2

            # Y axis
            for j in xrange(self.dim_xy):
                # X axis
                for i in xrange(self.dim_xy):
                    weights = []

                    if trans_quad == 3.0:
                        ne = [j, i]
                        nw = [j-1, i]
                        se = [j, i+1]
                        sw = [j-1, i+1]
                    elif trans_quad == 2.0:
                        ne = [j, i]
                        nw = [j+1, i]
                        se = [j+1, i]
                        sw = [j+1, i+1]
                    elif trans_quad == 1.0:
                        ne = [j, i]
                        nw = [j+1, i]
                        se = [j, i-1]
                        sw = [j+1, i-1]
                    else:
                        ne = [j, i]
                        nw = [j, i-1]
                        se = [j-1, i]
                        sw = [j-1, i-1]

                    # Wrap edges
                    for card in xrange(len(ne)):
                        if ne[card] < 0:
                            ne[card] = self.dim_xy-1
                        if ne[card] > self.dim_xy-1:
                            ne[card] = 0
                    for card in xrange(len(nw)):
                        if nw[card] < 0:
                            nw[card] = self.dim_xy-1
                        if nw[card] > self.dim_xy-1:
                            nw[card] = 0
                    for card in xrange(len(se)):
                        if se[card] < 0:
                            se[card] = self.dim_xy-1
                        if se[card] > self.dim_xy-1:
                            se[card] = 0
                    for card in xrange(len(sw)):
                        if sw[card] < 0:
                            sw[card] = self.dim_xy-1
                        if sw[card] > self.dim_xy-1:
                            sw[card] = 0

                    # Note: At 45deg, 0.2 comes from either side and 0.5 from behind.

                    # weight_sw = vtranscell * vtranscell * cos(dir90) * sin(dir90);
                    weights.append(vtranscell * vtranscell * tf.cos(trans_dir) * tf.sin(trans_dir))
                    weight_val = weights[0]
                    weight_idx = [k,j,i,k,sw[0],sw[1]]
                    idx.append(weight_idx)
                    val.append(weight_val)

                    # weight_se = vtranscell * sin(dir90) * (1.0 - vtranscell * cos(dir90));
                    weights.append(vtranscell * tf.sin(trans_dir) * (1.0 - vtranscell * tf.cos(trans_dir)))
                    weight_val = weights[1]
                    weight_idx = [k,j,i,k,se[0],se[1]]
                    idx.append(weight_idx)
                    val.append(weight_val)

                    # weight_nw = vtranscell * cos(dir90) * (1.0 - vtranscell * sin(dir90));
                    weights.append(vtranscell * tf.cos(trans_dir) * (1.0 - vtranscell * tf.sin(trans_dir)))
                    weight_val = weights[2]
                    weight_idx = [k,j,i,k,nw[0],nw[1]]
                    idx.append(weight_idx)
                    val.append(weight_val)

                    # weight_ne = 1.0 - weight_sw - weight_se - weight_nw;
                    weight_val = 1.0 - weights[2] - weights[1] - weights[0]
                    weight_idx = [k,j,i,k,ne[0],ne[1]]
                    idx.append(weight_idx)
                    val.append(weight_val)

        return idx, val

    def rotation(self):
        """
        Generating Rotation weights
        """
        print('Generating Rotation weights...')
        val = []
        idx = []

        # Convert to radians
        rads = tf.multiply(self.vrot, math.pi/180)

        weight = tf.mod(tf.divide(tf.abs(rads), self.size_th), 1)
        weight = tf.where(tf.equal(weight, 0.0), 1.0, weight)

        # TODO: `tf.sign()`
        sign_vrot = tf.cond(rads < 0, lambda: 1.0, lambda: -1.0)

        shifty1 = tf.cast(tf.multiply(sign_vrot, tf.floor(tf.divide(tf.abs(rads), self.size_th))), dtype=tf.int32)
        shifty2 = tf.cast(tf.multiply(sign_vrot, tf.ceil(tf.divide(tf.abs(rads), self.size_th))), dtype=tf.int32)

        cond = lambda i: tf.greater(i, 0)
        body = lambda i: tf.subtract(i, self.dim_th)
        tf.while_loop(cond, body, [shifty1])
        tf.while_loop(cond, body, [shifty2])

        for k in xrange(self.dim_th):
            newk1 = tf.mod(tf.subtract(k, shifty1), self.dim_th)
            newk2 = tf.mod(tf.subtract(k, shifty2), self.dim_th)
            for j in xrange(self.dim_xy):
                for i in xrange(self.dim_xy):
                    #posecells[k][j][i] = pca_new[newk1][j][i] * (1.0 - weight) + pca_new[newk2][j][i] * weight;

                    weight_val = 1.0 - weight
                    weight_idx = [k,j,i,newk1,j,i]
                    idx.append(weight_idx)
                    val.append(weight_val)

                    weight_val = weight
                    weight_idx = [k,j,i,newk2,j,i]
                    idx.append(weight_idx)
                    val.append(weight_val)

        return idx, val
