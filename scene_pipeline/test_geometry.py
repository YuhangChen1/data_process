"""Small synthetic checks for slab clipping and path planning failure modes."""
import unittest
import numpy as np
from .geometry import raster_slab,astar,segment_valid,smooth_route

class GeometryChecks(unittest.TestCase):
    def test_slab_clips_vertical_wall(self):
        v=np.array([[1.,0.,-1.],[1.,2.,-1.],[1.,2.,3.],[1.,0.,3.]])
        f=np.array([[0,1,2],[0,2,3]])
        mask,_=raster_slab(v,f,.1,1.8,np.array([0.,0.]),.1,25,25)
        self.assertTrue(mask[10,9:11].any())
        self.assertFalse(mask[:,5].any())
        outside,_=raster_slab(v,f,4.,5.,np.array([0.,0.]),.1,25,25)
        self.assertFalse(outside.any())

    def test_astar_uses_open_door_and_smoothing_stays_free(self):
        nav=np.ones((30,30),bool);nav[:,15]=False;nav[20:25,15]=True
        path=astar(nav,(5,5),(5,25),np.ones(nav.shape))
        self.assertTrue(any(p[0]>=20 for p in path))
        p=smooth_route(path,nav)
        self.assertTrue(all(segment_valid(a,b,nav) for a,b in zip(p[:-1],p[1:])))

    def test_no_diagonal_corner_cut(self):
        mask=np.array([[1,0],[0,1]],bool)
        self.assertFalse(segment_valid(np.array([0.,0.]),np.array([1.,1.]),mask))
        with self.assertRaises(RuntimeError):astar(mask,(0,0),(1,1),np.ones((2,2)))

if __name__=='__main__':unittest.main()
