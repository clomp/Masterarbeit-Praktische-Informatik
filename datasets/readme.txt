A train-test split of 0.8/0.2 is used for all datasets.

dataset1:
- N = 10,000 datapoints
- simulated data with noise in the input and output
- input (6): six joint angles (rad)
- output (7): absolute pose, i.e. position (m) and quaternions

dataset2:
- N = 10,000 datapoints
- simulated data with noise in the input and output
- input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
- output (7): absolute pose, i.e. position (m) and quaternions

dataset3:
- N = 10,000 datapoints
- simulated data with noise in the input and output
- input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
- output (7): relative pose, i.e. position error (m) and quaternion error

dataset4:
- N = 2,205 datapoints
- real measurement data
- input (6): six joint angles (rad)
- output (7): absolute pose, i.e. position (m) and quaternions

dataset5:
- N = 2,205 datapoints
- real measurement data
- input (6): six joint angles (rad)
- output (6): relative pose, i.e. position error (m) and angular error (rad)

dataset6:
- N = 352 datapoints
- real measurement data
- input (9): six joint angles (rad), force vector (N)
- output (6): absolute pose, i.e. position (m) and angular error (rad)

normalization
absolute pose: (x,y,z) in [-1,3]
quaternions in [-1,1]
relative pose?
lb_theta = [0,-30,-30,-180,-90,-180]*pi/180;    %lower bounds joint angles
ub_theta = [90,60,30,180,90,180]*pi/180;          %upper bounds joint angles

lb_force = [-2000,-2000,-2000,0,0,0];           %lower bounds force
ub_force = [2000,2000,2000,0,0,0];              %upper bounds force


