from pathlib import Path
import sys,json,hashlib,os,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/python'))
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import shape
from shapely import make_valid

VOCAB=['sofa','bed','table','chair','cabinet','wardrobe','desk','bookshelf','television','door','window','refrigerator','sink','toilet','bathtub','shower','washing machine','air conditioner','lamp','mirror','plant','stove','oven','curtain','railing','stairs']
REGION_TYPES=['living_room','dining_room','bedroom','kitchen','bathroom','balcony','corridor','entryway','study','storage','laundry','stairwell','courtyard','terrace','garage','mixed','unknown']
NAMES={'living_room':'客厅','dining_room':'餐厅','bedroom':'卧室','kitchen':'厨房','bathroom':'卫生间','balcony':'阳台','corridor':'走廊','entryway':'玄关','study':'书房','storage':'储藏区','laundry':'洗衣区','stairwell':'楼梯区','courtyard':'庭院','terrace':'露台','garage':'车库','mixed':'混合功能区','unknown':'待确认区域'}
OBJECT_ZH={'sofa':'沙发','bed':'床','table':'桌子','chair':'椅子','cabinet':'柜子','wardrobe':'衣柜','desk':'书桌','bookshelf':'书架','television':'电视','door':'门','window':'窗户','refrigerator':'冰箱','sink':'水槽','toilet':'马桶','bathtub':'浴缸','shower':'淋浴设施','washing machine':'洗衣机','air conditioner':'空调','lamp':'灯具','mirror':'镜子','plant':'植物','stove':'灶台','oven':'烤箱','curtain':'窗帘','railing':'栏杆','stairs':'楼梯'}

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def score(x):return float(np.clip(float(x or 0),0,1))
def base(source,confidence):return {'source':source,'score':score(confidence),'review':{'status':'unreviewed'}}
def rel(path,root):return os.path.relpath(Path(path).resolve(),Path(root).resolve())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def aabb_iou(a,b):
    a0=np.array(a['center'])-np.array(a['size'])/2;a1=np.array(a['center'])+np.array(a['size'])/2
    b0=np.array(b['center'])-np.array(b['size'])/2;b1=np.array(b['center'])+np.array(b['size'])/2
    intersection=float(np.maximum(0,np.minimum(a1,b1)-np.maximum(a0,b0)).prod());union=float((a1-a0).prod()+(b1-b0).prod()-intersection)
    return intersection/max(union,1e-10)
def poly_iou(a,b):
    x=make_valid(shape(a));y=make_valid(shape(b));return x.intersection(y).area/max(x.union(y).area,1e-10)
def stable_ids(new,old,prefix,metric,threshold,registry):
    old=list(old);next_id=max([int(x['id'][len(prefix):]) for x in old if x['id'].startswith(prefix)]+[registry.get(prefix,0)])+1
    matched={}
    if new and old:
        sim=np.array([[metric(a,b) for b in old] for a in new])
        rows,cols=linear_sum_assignment(-sim)
        matched={i:old[j]['id'] for i,j in zip(rows,cols) if sim[i,j]>=threshold}
    for i,record in enumerate(new):
        if i in matched:record['id']=matched[i]
        else:record['id']=prefix+str(next_id).zfill(3 if prefix in ['o','t'] else 2);next_id+=1
    registry[prefix]=max(registry.get(prefix,0),next_id-1)
    return {'matched':len(matched),'new':len(new)-len(matched),'previous_unmatched':len(old)-len(matched),'threshold':threshold}
