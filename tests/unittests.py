import unittest
import numpy as np
import sys
sys.path.append('/home/kevin/git_repos/dummy_particles')
import file_io    # noqa
import force_analysis  # noqa


# ----------------------------------------------
# io tests
# ----------------------------------------------
class test_load_xvg(unittest.TestCase):

    def test_xvg_1D(self):
        data = file_io.load_xvg('files/testFileIO/data_1D.xvg', dims=1)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 1)

    def test_xvg_2D(self):
        data = file_io.load_xvg('files/testFileIO/data_2D.xvg', dims=2)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 2)

    def test_xvg_3D(self):
        data = file_io.load_xvg('files/testFileIO/data_3D.xvg', dims=3)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 3)

    def test_fakedata_3D(self):
        #  make sure reordering is correct, values actually right
        data = file_io.load_xvg('files/testFileIO/fake_3D_data.xvg', dims=3)
        self.assertEqual(data[1, 1, 0], 10)
        self.assertEqual(data[1, 1, 1], 11)
        self.assertEqual(data[1, 1, 2], 12)

    def test_xvg_return_time(self):
        data, time = file_io.load_xvg('files/testFileIO/data_3D.xvg', dims=3, return_time_data=True)
        self.assertEqual(time.size, 6)

    def test_xvg_comments(self):
        data = file_io.load_xvg('files/testFileIO/data_&comments.xvg', dims=3, comments=('#', '@', '&'))
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 3)

    def test_xvg_column_mismatch_error(self):
        self.assertRaises(ValueError, file_io.load_xvg, 'files/testFileIO/data_1D.xvg', dims=3)


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


class test_calc_posres_forces(unittest.TestCase):

    def test_shape_error(self):
        bad_coords = np.zeros((1, 15))
        good_coords = np.zeros((1, 15, 1))
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, good_coords, bad_coords, 10)
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, bad_coords, good_coords, 10)

    def test_dimension_mismatch_error(self):
        traj_coords = np.zeros((10, 100, 3))
        ref_coords  = np.zeros((1,  98, 3))
        self.assertRaises(ValueError, force_analysis.calc_posres_forces, traj_coords, ref_coords, 10)

    def test_spring_constant_calculation(self):
        traj_coords = np.zeros((2, 10, 1)) + 5   # all set to 5
        ref_coords =  np.ones((1, 10, 1))         # all set to 1
        forces = force_analysis.calc_posres_forces(traj_coords, ref_coords, 10)
        self.assertSequenceEqual(forces.shape, (2, 10, 1))
        self.assertEqual(forces[0, 0, 0], 40)


if __name__ == '__main__':
    unittest.main()
