import mdtraj as md
import numpy as np
import sys
sys.path.append('/home/kevin/git_repos/dummy_particles')
import file_io    # noqa
import force_analysis  # noqa


# basic test of loading trajectories, and reference coordinates, and calculating forces
# on the beads from a spring constant. This system has 800 particles, 400 on top of bilayer
# and 400 on bottom
def basic_force_test():
    spring_constant = 1000  # kj/ mol nm^2

    # load trajectory and reference positions
    prefix = './files/test_basic_force_calculation/'
    ref_pdb  = md.load(prefix + 'dummy_ref.pdb')
    dummy_traj = md.load_xtc(prefix + 'dummy_coords.xtc', top=prefix + 'dummy_firstframe.pdb')

    # get trajectory info
    n_frames = dummy_traj.xyz.shape[0]
    ref_dims = ref_pdb.unitcell_lengths.flatten()
    traj_dims = dummy_traj.unitcell_lengths

    # do frame matching, scaling, and force calculation
    ref_xyz = force_analysis.multiply_coordinate_frame(ref_pdb.xyz, n_frames)     # match size of trajectory
    ref_xyz = force_analysis.scale_box_coordinates(ref_xyz, traj_dims, ref_dims)  # scale reference dimensions
    forces =  force_analysis.calc_posres_forces(dummy_traj.xyz, ref_xyz, spring_constant)
    print("upper force average = {}".format(forces[:, 0:400, :].mean(axis=0).mean(axis=0)))
    print("lower force average = {}".format(forces[:, 400:,  :].mean(axis=0).mean(axis=0)))


def main():
    basic_force_test()


if __name__ == '__main__':
    main()
