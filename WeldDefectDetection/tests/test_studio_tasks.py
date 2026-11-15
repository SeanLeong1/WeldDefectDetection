from pathlib import Path
import numpy as np
import pytest
from inspection import studio_tasks as tasks


def test_discovery_excludes_masks_and_exports(tmp_path):
    for name in ('topo.npy','topo_mask.npy','stages.npz','predictions.npz','inspection.png','x.PNG','other.TIFF'):
        (tmp_path/name).touch()
    sub=tmp_path/'sub';sub.mkdir();(sub/'image.jpeg').touch()
    output=tmp_path/'batch_results';output.mkdir();(output/'image.png').touch()
    assert {p.name for p in tasks.discover(tmp_path)}=={'topo.npy','x.PNG','other.TIFF','image.jpeg'}
    assert len(tasks.discover(tmp_path,False))==3


def test_truth_halo_and_periodic_matching(tmp_path):
    mask=np.zeros((160,5024),'u1');mask[50:60,:5]=1;mask[50:60,-5:]=1
    np.save(tmp_path/'image_mask.npy',mask)
    truth=tasks.read_truth(tmp_path/'image.npy',tmp_path)
    assert truth.shape==(128,5024)
    boxes=tasks.engine.metrics.regions(truth)[1]
    assert len(boxes)==1
    score=tasks.evaluate_result({'boxes':{'cnn':boxes,'median':[]}},truth)
    assert score['cnn']['tp']==1 and score['cnn']['fp']==0
    assert score['median']['fn']==1


def test_summary_denominators_and_unlabelled_images():
    rows=[dict(status='ok',seconds=2,cnn_candidates=2,median_candidates=0,
               evaluation={'cnn':dict(tp=1,fp=1,fn=2),'median':dict(tp=0,fp=0,fn=3)}),
          dict(status='ok',seconds=3,cnn_candidates=5,median_candidates=1,evaluation={}),
          dict(status='error')]
    result=tasks.summarise(rows)
    assert result['completed']==2 and result['errors']==1
    cnn=result['methods']['cnn']
    assert cnn['candidates']==7 and cnn['mean_candidates']==3.5
    assert cnn['labelled_images']==1 and cnn['precision']==.5 and cnn['f1']==.4
    assert result['methods']['median']['precision'] is None


def test_bad_mask_is_rejected(tmp_path):
    np.save(tmp_path/'image_mask.npy',np.zeros((42,42)))
    with pytest.raises(ValueError,match='binary'):tasks.read_truth(tmp_path/'image.npy',None)


def test_folder_continues_after_error_and_evaluates_after_inference(tmp_path,monkeypatch):
    source=tmp_path/'inputs';source.mkdir()
    output=tmp_path/'output';output.mkdir()
    for name in ('a.npy','bad.npy'):(source/name).touch()
    np.save(source/'a_mask.npy',np.zeros((160,5024),'u1'))
    order=[]
    class Inspector:
        def infer(self,path):
            order.append('infer')
            if path.name=='bad.npy':raise ValueError('invalid input')
            return dict(boxes={'cnn':[],'median':[]},seconds=.1)
    def export(result,folder):folder.mkdir();return folder
    monkeypatch.setattr(tasks.engine,'export_result',export)
    original=tasks.read_truth
    def read(*args):order.append('truth');return original(*args)
    monkeypatch.setattr(tasks,'read_truth',read)
    summary,rows=tasks.inspect_folder([source/'a.npy',source/'bad.npy'],source,Inspector(),output,
        lambda *a:None,lambda:False,use_masks=True)
    assert order==['infer','truth','infer']
    assert summary['completed']==1 and summary['errors']==1
    assert summary['methods']['cnn']['f1'] is None
    assert (output/'images.csv').exists() and (output/'summary.json').exists()


def test_cancelled_folder_keeps_requested_count(tmp_path):
    summary,rows=tasks.inspect_folder([Path('unused.npy')],None,None,tmp_path,lambda *a:None,lambda:True)
    assert rows==[] and summary['remaining']==1 and summary['cancelled']


@pytest.mark.parametrize('settings',[tasks.GenerationSettings(number=0),tasks.GenerationSettings(crown_width=20),
                                    tasks.GenerationSettings(seed=3000000000,number=2)])
def test_generation_validation(settings):
    with pytest.raises(ValueError):settings.validate()
