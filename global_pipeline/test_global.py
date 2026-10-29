import tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .common import DEFAULTS,read
from .geometry import project_points,fh
from .instances import constrained_merge,refine,stabilize

class GlobalChecks(unittest.TestCase):
    def test_occlusion_and_depth_consistency(self):
        frame={'K':[[1,0,.5],[0,1,.5],[0,0,1]],'camera_to_canonical':np.eye(4).tolist()}
        points=np.array([[0,0,1],[0,0,2],[10,0,1]],np.float32)
        indices,pixels,z=project_points(points,frame,np.array([[1.]],np.float32),DEFAULTS)
        self.assertEqual(indices.tolist(),[0]);self.assertEqual(pixels.tolist(),[0])
        indices,_,_=project_points(points,frame,np.array([[3.]],np.float32),DEFAULTS)
        self.assertEqual(len(indices),0)
        for invisible in [np.array([[0,0,-1]],np.float32),np.array([[10,0,1]],np.float32),np.empty((0,3),np.float32)]:
            indices,pixels,z=project_points(invisible,frame,np.array([[1.]],np.float32),DEFAULTS)
            self.assertEqual((len(indices),len(pixels),len(z)),(0,0,0))

    def test_transitive_cannot_link_survives_bridge(self):
        records=[]
        for i,(frame,lo,hi) in enumerate([('f0',[0,0,0],[1,1,1]),('f1',[0,0,0],[3,1,1]),('f0',[2,0,0],[3,1,1])]):
            records.append({'proposal_id':i,'frame_id':frame,'bbox_low':lo,'bbox_high':hi,'hard_quality':True,'quality':1.,'support_points':60})
        support=csr_matrix(np.array([[1,1,1,0,0,0],[1,1,1,1,1,1],[0,0,0,1,1,1]],np.float32))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);coarse,_,_=constrained_merge(records,support,out,DEFAULTS);summary=read(out/'coarse_summary.json')
            self.assertEqual(summary['hard_conflicts_inside_clusters'],0)
            self.assertGreater(summary['blocked_transitive_merges'],0)
            self.assertEqual(summary['coarse_instances'],2)

    def test_fh_respects_max_size_and_extent(self):
        positions=np.c_[np.arange(20)*.05,np.zeros(20),np.zeros(20)].astype(np.float32)
        edges=np.c_[np.arange(19),np.arange(1,20)].astype(np.int32)
        parents=fh(20,edges,np.full(19,.01,np.float32),positions,.5,2,4,.16)
        for parent in np.unique(parents):
            group=positions[parents==parent];self.assertLessEqual(len(group),4);self.assertLessEqual(np.ptp(group[:,0]),.16001)

    def test_fine_requires_independent_views_and_keeps_topology(self):
        coarse=np.array([1,1,2,2],np.int32)
        cloud={'centroids':np.array([[0,0,0],[-.2,0,0],[.03,0,0],[.06,0,0]],np.float32),
               'adjacency':np.array([[0,1],[0,2],[2,3]],np.int32)}
        evidence=csr_matrix(np.array([[2,0,10,10],[2,0,10,10]],np.float32))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            repeated=[{'quality':1.,'view_group':(0,0,0,0)}]*2
            same=refine(coarse,repeated,evidence,cloud,out,DEFAULTS)
            self.assertTrue(np.array_equal(same,coarse))
            diverse=[{'quality':1.,'view_group':(0,0,0,0)},{'quality':1.,'view_group':(1,0,0,0)}]
            changed=refine(coarse,diverse,evidence,cloud,out,DEFAULTS)
            self.assertEqual(set(changed),{1,2});self.assertEqual(changed[0],2)

    def test_global_ids_survive_label_permutation_and_new_instance(self):
        cloud={'points':np.c_[np.arange(6),np.zeros(6),np.zeros(6)].astype(np.float32),
               'colors':np.zeros((6,3),np.float32),'point_superpoint_ids':np.arange(6),
               'superpoint_sizes':np.ones(6,np.int32)}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);initial=np.array([1,1,2,2,0,0],np.int32)
            _,first=stabilize(out,cloud,initial,initial)
            permuted=np.array([2,2,1,1,3,3],np.int32)
            _,second=stabilize(out,cloud,permuted,permuted)
            self.assertTrue(np.array_equal(first[:4],second[:4]))
            self.assertGreater(second[4],max(first));self.assertEqual(second[4],second[5])

if __name__=='__main__':unittest.main()
