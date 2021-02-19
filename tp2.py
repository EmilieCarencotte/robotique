import math
import time
import numpy as np
from numpy.linalg import norm,inv,pinv,svd,eig
import pinocchio as pin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
import tp2.load_ur5_parallel as robex2
from tp2.meshcat_viewer_wrapper import *
from pinocchio import SE3, Quaternion


robot = robex2.load()
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
[w, h, d] = [0.5, 0.5, 0.005]
color = [red, green, blue, transparency] = [1, 1, 0.78, .8]
viz.addBox('world/robot0/toolplate', [w, h, d], color)
Mtool = pin.SE3(pin.utils.rotate('z', 1.268), np.array([0, 0, .75]))
viz.applyConfiguration('world/robot0/toolplate', Mtool)
effIdxs = [ robot.model.getFrameId('tool0_#%d' % i) for i in range(4)]

quat = pin.Quaternion(0.7,0.2,0.2,0.6).normalized()
R  = quat.matrix()                # Create a rotation matrix from quaternion
p  = np.array([0.,0.,0.75])     # Translation (R3) vector)
M  = SE3(R,p)
viz.applyConfiguration('world/robot0/toolplate', M)
oMrs = [robot.framePlacement(robot.q0,effIdxs[i]) for i in range(4)]
rsMbox = [oMrs[i].inverse()*Mtool for i in range(4)]
Mtarget = SE3(R,p)

def cost(q):
  '''Compute score from a configuration'''
  cost = 0
  oMrs = [robot.framePlacement(q,effIdxs[i]) for i in range(4)]
  for i in range(4):
    cost += norm(pin.log((oMrs[i].inverse() * Mtarget).inverse() * rsMbox[i]).vector)
  return cost

def callback(q):
  viz.display(q)
  time.sleep(1e-1)


def rotation():
  qopt = fmin_bfgs(cost, robot.q0, callback=callback)

  print('The robot finally reached effector placement at\n',robot.placement(qopt,6))
