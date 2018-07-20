import unittest
import numpy as np
import sys
sys.path.append('..')
import force_analysis  # noqa


# -------------------------------
# force analysis tests
# -------------------------------
class test_scale_box_coordinates(unittest.TestCase):

    def test_dims_mismatch_error(self):
        traj_xyz = np.ones((10, 8, 3))
        traj_dims = 100 + (np.random.rand(10, 3))
        ref_dims = 100 + np.random.rand(2)
        self.assertRaises(ValueError, force_analysis.scale_box_coordinates, traj_xyz, traj_dims, ref_dims)

    def test_all_3D_error(self):
        good_xyz,  bad_xyz  = np.ones((10, 5, 3)), np.ones((10, 5, 2))
        good_dims, bad_dims = np.ones((10, 3)),    np.ones((10, 1))
        good_ref,  bad_ref  = np.ones((3)),        np.ones((2))
        self.assertRaises(ValueError, force_analysis.scale_box_coordinates, good_xyz, good_dims,  bad_ref)
        self.assertRaises(ValueError, force_analysis.scale_box_coordinates, good_xyz,  bad_dims, good_ref)
        self.assertRaises(ValueError, force_analysis.scale_box_coordinates, bad_xyz,  good_dims, good_ref)

    def test_scaling(self):
        traj_xyz  = np.zeros((10, 5, 3)) + (15, 10, 5)
        traj_dims = np.zeros((10, 3)) + 3
        ref_dims  = np.array((1, 6, 3))
        # dims are 3X, 0.5X and 1X reference. So should be 15 * 3, 10 * 0.5, 5 * 1
        scaled_coords = force_analysis.scale_box_coordinates(traj_xyz, traj_dims, ref_dims)
        self.assertTrue(all(scaled_coords[5, 0, :] == (45, 5, 5)))

    def test_coordinate_frame_multiplication(self):
        ref_xyz = np.ones((1, 10, 3))
        mult_xyz = force_analysis.multiply_coordinate_frame(ref_xyz, 15)
        self.assertSequenceEqual(mult_xyz.shape, (15, 10, 3))


class test_calc_vectors(unittest.TestCase):
    def test_coordinate_mismatch_exceptions(self):
        cp = np.ones((10, 20, 3))
        cd1 = np.ones((9, 20, 3))    # bad frame number
        cd2 = np.ones((10, 10, 3))   # bad particle number
        cd3 = np.ones((10, 20, 2))   # bad dimension number
        boxdims = np.ones((10, 3))   # correct
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp, cd1, boxdims)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp, cd2, boxdims)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp, cd3, boxdims)

    def test_coordinate_boxdim_mismatch_exceptions(self):
        cp = np.ones((10, 20, 3))
        boxdims = np.ones((9, 3))    # not enough frames
        boxdims2 = np.ones((10, 2))  # not enough dimensions
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp, cp, boxdims)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp, cp, boxdims2)

    def test_bad_input_dimension_exceptions(self):
        cp_too_few       = np.ones((100, 3))
        cp_too_many      = np.ones((100, 20, 3, 4))
        cp_good          = np.ones((100, 20, 3))
        boxdims_too_few  = np.ones(100)
        boxdims_too_many = np.ones((100, 3, 10))
        boxdims_good     = np.ones((100, 3))
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp_too_few, cp_too_few, boxdims_good)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp_too_many, cp_too_many, boxdims_good)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp_good, cp_good, boxdims_too_few)
        self.assertRaises(ValueError, force_analysis.calc_vectors, cp_good, cp_good, boxdims_too_many)

    def test_for_correct_vectors(self):
        boxdims = np.array([[10, 10, 10], [9.5, 9.5, 9.5]])   # 2 frames, 3D
        p_low  = np.ones((2, 2, 3))          # = 1
        p_high = np.ones((2, 2, 3)) + 8.4    # = 9.4
        p_mid  = np.ones((2, 2, 3)) + 4      # = 5
        vecs_pos_no_pi   = force_analysis.calc_vectors(p_mid, p_high, boxdims)
        vecs_neg_no_pi   = force_analysis.calc_vectors(p_mid,  p_low, boxdims)
        vecs_prev_pi     = force_analysis.calc_vectors(p_low, p_high, boxdims)
        vecs_next_pi     = force_analysis.calc_vectors(p_high, p_low, boxdims)
        self.assertAlmostEqual(vecs_pos_no_pi[0, 0, 2], 4.4)
        self.assertAlmostEqual(vecs_neg_no_pi[1, 1, 2], -4)
        self.assertAlmostEqual(vecs_prev_pi[0, 0, 0], -1.6)
        self.assertAlmostEqual(vecs_next_pi[1, 1, 1], 1.1)   # 2nd frame, 9.5 size


class test_calc_posres_forces(unittest.TestCase):

    def test_shape_error(self):
        bad_coords = np.zeros((1, 15))
        good_coords = np.zeros((1, 15, 1))
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, good_coords, bad_coords, 10)
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, bad_coords, good_coords, 10)

    def test_dimension_mismatch_error(self):
        traj_coords = np.zeros((10, 100, 3))
        ref_coords  = np.zeros((10, 100, 2))
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, traj_coords, ref_coords, 10)

    def test_spring_constant_calculation(self):
        traj_coords     = np.zeros((2, 10, 1)) + 5   # all set to 5
        traj_coords_neg = np.zeros((2, 10, 1)) - 3
        ref_coords =  np.ones((2, 10, 1))         # all set to 1
        forces     = force_analysis.calc_posres_forces(traj_coords,     ref_coords, 10)
        forces_neg = force_analysis.calc_posres_forces(traj_coords_neg, ref_coords, 10)
        self.assertSequenceEqual(forces.shape, (2, 10, 1))
        self.assertEqual(forces[0, 0, 0],      40)
        self.assertEqual(forces_neg[0, 0, 0], -40)


if __name__ == '__main__':
    unittest.main()
