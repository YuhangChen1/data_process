import tempfile,unittest
from pathlib import Path
import numpy as np
from .common import stable_ids,aabb_iou,poly_iou
from .finalize import build_texts

class AnnotationChecks(unittest.TestCase):
    def test_stable_region_assignment_after_reorder(self):
        def square(x):return {'type':'Polygon','coordinates':[[[x,0],[x+1,0],[x+1,1],[x,1],[x,0]]]}
        old=[{'id':'r01','geometry':square(0)},{'id':'r02','geometry':square(3)}]
        new=[{'geometry':square(3.05)},{'geometry':square(.05)},{'geometry':square(8)}];registry={'r':5}
        result=stable_ids(new,old,'r',lambda a,b:poly_iou(a['geometry'],b['geometry']),.5,registry)
        self.assertEqual([x['id'] for x in new],['r02','r01','r06']);self.assertEqual(result['matched'],2)

    def test_object_iou_and_retired_ids(self):
        a={'center':[0,0,0],'size':[2,2,2]};b={'center':[.2,0,0],'size':[2,2,2]}
        self.assertGreater(aabb_iou(a,b),.8)
        new=[{'bbox':a}];old=[{'id':'o003','bbox':b}];registry={'o':20}
        stable_ids(new,old,'o',lambda a,b:aabb_iou(a['bbox'],b['bbox']),.2,registry)
        self.assertEqual(new[0]['id'],'o003')
        nxt=[{'bbox':{'center':[9,9,9],'size':[1,1,1]}}]
        stable_ids(nxt,new,'o',lambda a,b:aabb_iou(a['bbox'],b['bbox']),.2,registry)
        self.assertEqual(nxt[0]['id'],'o021')

    def test_text_policy_and_id_stability(self):
        graph={'scene':{'id':'s0','type_pred':{},'description_pred':{'short':'左侧可见沙发。','long':'右侧有餐桌。','score':.8,'source':'vlm'}},'regions':[],'objects':[],'issues':[]}
        old={'texts':[{'id':'t003','level':'scene','ref':'s0','length':'short'},{'id':'t009','level':'scene','ref':'s0','length':'long'}]}
        build_texts(graph,old,{'t':9});self.assertEqual([x['id'] for x in graph['texts']],['t003','t009'])
        self.assertNotIn('左侧',str(graph['texts']));self.assertNotIn('右侧',str(graph['texts']))
        self.assertTrue(all(x['visibility_default']=='public' and x['review']['status']=='unreviewed' for x in graph['texts']))

if __name__=='__main__':unittest.main()
