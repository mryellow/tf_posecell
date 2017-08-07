import math
import tensorflow as tf
import pyglet

from window import Window
from posecells import Posecells

sess = tf.Session()

PC_DIM_TH = 7
PC_DIM_XY = 7

def main(_):
    writer = tf.summary.FileWriter("/tmp/tf_posecell_graph", sess.graph)

    res = None

    window = Window(PC_DIM_TH, PC_DIM_XY)
    posecells = Posecells(sess, PC_DIM_TH, PC_DIM_XY)

    window.timestep = 0

    def update(dt):
        window.timestep += 1

        # Do stuff
        data = None
        tasks = []
        # TODO: Prevent injection into recent view templates
        #if window.pose_last != window.pose:
        #    #print('pose', window.pose)
        #    print('vel', window.vtrans['x'], window.vrot['z'])
        #    tasks.append(on_odo)
        #if window.view_last != window.view:
        #    print('view', window.view)
        #    tasks.append(on_view_template)
        #    tasks.append(decay_restore)

        #if window.view[0] >= 0:
        #    print('view', window.view)
        #    tasks.append(on_view_template)
        #    tasks.append(decay_restore)

        tasks.append(posecells.excite())
        #tasks.append(posecells.path_integration())

        #tasks.append(find_best)

        res = sess.run(tasks, feed_dict={
            #pose_input: window.pose,
            #view_input: window.view,
            posecells.vtrans: window.vtrans['x'],
            posecells.vrot: window.vrot['z']
        })

        #window.pose_last = window.pose[:]
        #window.view_last = window.view[:]

        # Re-scale between [0, 1] for rendering (transparency percentage)
        if res and len(res) > 0:
            data = res[0]
        else:
            return

        max_activation = 0
        min_activation = 1
        for z in xrange(len(data)):
            for y in xrange(len(data[z])):
                for x in xrange(len(data[z][y])):
                    if data[z][y][x] > 0:
                        max_activation = max(max_activation, data[z][y][x])
                        min_activation = min(min_activation, data[z][y][x])
                    else:
                        max_activation = max(max_activation, 0.0)
                        min_activation = min(min_activation, 0.0)

        for z in xrange(len(data)):
            for y in xrange(len(data[z])):
                for x in xrange(len(data[z][y])):
                    if max_activation - min_activation > 0:
                        data[z][y][x] = (data[z][y][x] - min_activation) / (max_activation - min_activation)
                    else:
                        data[z][y][x] = 0.0

        for z in xrange(len(data)):
            for y in xrange(len(data[z])):
                for x in xrange(len(data[z][y])):
                    # TODO: Switch these axis around and draw axis in window
                    window.voxel.set(x, y, z, data[z][y][x])

    pyglet.clock.schedule(update)

    pyglet.app.run()
    writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
