## RatSLAM PoseCell in TensorFlow

`pca_new[zw][yw][xw] += posecells[z][y][x] * PC_W_EXCITE[excite_index++];`

https://github.com/mryellow/ratslam/blob/ratslam_ros/src/ratslam/posecell_network.cpp#L710

That's meant to be adding to surrounding cells. Rather than moving the distribution to those coordinates.
