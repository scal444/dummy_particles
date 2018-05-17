import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import file_io


def multiply_coordinate_frame(ref_xyz, n_frames):
    ''' Takes references coordinates and repeats them so that they have same dimensions as a trajectory xyz array'''
    return np.repeat(ref_xyz, n_frames, axis=0)


def scale_box_coordinates(traj_xyz, traj_dims, ref_dims):
    '''
        Scales a coordinate set in 3 dimensions according to changing box size. The scaling is by the fractional size
        of the trajectory box compared to a static reference box dimension. The trajectory, box dimensions, and
        reference dimensions all need to be 3D.

        Parameters
            -traj_xyz    - n_frames * n_particles * 3 array of coordinates
            -traj_dims   - n_frames * 3               array of box dimensions
            -ref_dims    - size 3 array of box dimensions

        Returns
            -scaled_xyz  - scaled coordinates same size as traj_xyz
    '''

    # make sure back dims match up
    if traj_xyz.shape[2] != 3 or traj_dims.shape[1] != 3 or ref_dims.shape != (3,):
        raise ValueError("One of the inputs does not have 3 as it's final dimension size")
    if traj_xyz.shape[0] != traj_dims.shape[0]:
        raise ValueError("trajectory coords/dims have different frame counts: {} vs {}".format(traj_xyz.shape[0], traj_dims.shape[0]))  # noqa

    scale_factor = traj_dims / ref_dims
    return traj_xyz * scale_factor[:, np.newaxis, :]


def calc_posres_forces(traj_xyz, ref_xyz, spring_constant):
    '''
        Calculates the average force acting on a dummy particle kept in place using position restraints. This is done
        by calculating the displacement of the bead from its equilibrium value and using the spring constant force
        equation.

        Physics
            F = -kX   (kX to keep a particle at a specific distance from reference)

        Parameters
            -traj_xyz        - n_frames * n_particles * xyz array of coordinates (actual coordinates)
            -ref_xyz         - n_frames * n_particles * xyz array of coordinates (position restraint reference coords)
            -spring constant - force constant that keeps dummy particles in place. Gromacs units are k=kJ/(mol nm^2)

        Returns
            - forces         - n_frames * n_particles * xyz - dimensional components of forces
    '''

    # check dimensions of inputs
    trajshape, refshape = traj_xyz.shape, ref_xyz.shape
    if len(trajshape) < 3 or len(refshape) < 3:
        raise ValueError("Dimension mismatch. traj_xyz and ref_xyz must be mdtraj format xyz arrays, of dimensions" +
                         "n_frames * n_particles * dims, even if there is only one frame/particle/dimension")

    if  trajshape != refshape:
        raise ValueError("Dimension mismatch between traj_xyz and ref_xyz - {} to {}".format(trajshape, refshape ))

    dists = traj_xyz - ref_xyz
    return  spring_constant * dists


if __name__ == '__main__':
    # testing
    prefix = '/home/kevin/hdd/Projects/software_validation/dummy_particles/flat_bilayer/dummy_3.6nm/'
    top_force = file_io.load_xvg(prefix + 'freeze/data/dummy_top_force.xvg', dims=3)

    dummy_ref  = md.load(prefix + 'pos_res/dummy_ref.pdb')
    dummy_traj = md.load_xtc(prefix + 'pos_res/dummy_coords.xtc', top=prefix + 'pos_res/dummy_firstframe.pdb')
    dummy_ref_dims = dummy_ref.unitcell_lengths
    dummy_traj_dims = dummy_traj.unitcell_lengths

    a = calc_posres_forces(dummy_traj.xyz, dummy_ref.xyz, 1000,
                           scaling=True, traj_dims=dummy_traj_dims, ref_dims=dummy_ref_dims)
