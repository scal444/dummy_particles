import mdtraj as md
import numpy as np
import file_io


def calc_vectors(p_origin, p_destination, boxdims):
    """
        MDtraj has functionality for computing distances but it's not always applicable to every dataset, and distances
        contain no directonality. This function will calculate vectors for coordinates, taking into account the box
        dimensions. For simplicity, will only take in mdtraj xyz shaped arrays (and trajectory.unitcell_lengths)

        Note that this will only calculate vectors within 1 periodic image!

        Parameters
            p_origin      - n_frames * n_particles * n_dimensions coordinate array
            p_destination - n_frames * n_particles * n_dimensions coordinate array - same size as p_origin
            boxdims       - n_frames * n_dimensions array of box dimensions

        Returns
            vecs -n_frames * n_particles * n_dimensions array
    """
    if not p_origin.ndim == 3:
        raise ValueError("coordinates should be nframes * nparticles * ndims, p_origin shape = {}".format(p_origin.shape))   # noqa
    if not boxdims.ndim == 2:
        raise ValueError("boxdims should be nframes * nparticles, boxdims shape = {}".format(boxdims.shape))
    if not p_origin.shape == p_destination.shape:
        raise ValueError("input vector dimension mismatch. Origin shape = {}, destination shape =  {}".format(
                         p_origin.shape, p_destination.shape))
    if not p_origin.shape[0] == boxdims.shape[0]:  # mismatch between number of frames in coords and boxdims
        raise ValueError("Mismatch between number of frames in coordinates ({}) and boxdims ({})".format(
                         p_origin.shape[0], boxdims.shape[0]))
    if not p_origin.shape[2] == boxdims.shape[1]:  # mismatch between dimensionality
        raise ValueError("Mismatch between number of dimensions in coordinates ({}) and boxdims ({})".format(
                         p_origin.shape[2], boxdims.shape[1]))

    boxdims_reshaped = boxdims[:, np.newaxis, :]  # allows broadcasting
    boxdims_midpoint = boxdims_reshaped / 2
    vecs = p_destination - p_origin
    veclengths = np.abs(vecs)

    # these are the vectors who's periodic image are closer than the original vecotor
    vecs_gt_boxdims = veclengths >  (boxdims_midpoint)  # these positions will be changed

    # boolean arrays for identifying closest periodic image - based on vector direction instead of
    # place in box, which might not be centered on (0, 0, 0)
    negative_vecs = vecs < 0
    positive_vecs = vecs > 0

    # for positive vectors greater than half the box, use previous periodic image
    vecs[vecs_gt_boxdims & positive_vecs] = -(boxdims_reshaped - veclengths)[vecs_gt_boxdims & positive_vecs]

    # for negative vectors greater than half the box, use next periodic image.
    vecs[vecs_gt_boxdims & negative_vecs] = (boxdims_reshaped - veclengths)[vecs_gt_boxdims & negative_vecs]

    return vecs


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
