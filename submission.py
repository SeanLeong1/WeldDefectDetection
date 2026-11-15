"""Portable entry point for the frozen thesis methods and existing studio GUI."""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install_portable_inspector():
    from inspection import final_methods as engine
    import torch
    class PortableInspector(engine.FinalInspector):
        def __init__(self):
            torch.set_num_threads(4)
            assets=ROOT/'assets'; spec=read(assets/'final_model.json')
            expected=read(assets/'SHA256.json')
            for name,digest in expected.items():
                if sha(assets/name)!=digest:raise ValueError(f'Frozen asset hash mismatch: {name}')
            self.cnn={**spec['cnn'],'checkpoint':str(assets/'cnn_selected.pt'),
                      'selection':{'best':spec['cnn_operating_point']}}
            self.median=spec['median'];self.norm=read(assets/'normalization.json')
            self.calibration=read(assets/'median_calibration.json');self.model=None
            self.identity={'cnn_arm':self.cnn['arm'],'cnn_seed':self.cnn['seed'],
                'cnn_epoch':self.cnn['epoch'],'checkpoint_sha256':self.cnn['checkpoint_sha256'],
                'frozen_selection_sha256':spec['source_frozen_sha256'],
                'normalization_sha256':expected['normalization.json'],
                'calibration_sha256':expected['median_calibration.json'],
                'cnn_operating_point':spec['cnn_operating_point'],
                'median_operating_point':self.median['base'],
                'median_box_correction':self.median['correction']}
    engine.FinalInspector=PortableInspector
    engine.example_path=lambda:ROOT/'examples/open_intrinsic.npy'
    return engine

def demo(engine,args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from circular_study import metrics
    inspector=engine.FinalInspector(); args.output.mkdir(parents=True,exist_ok=True)
    result=inspector.infer(args.input)
    report={'input':str(args.input),'identity':result['identity'],'seconds':result['seconds'],
            'boxes':result['boxes'],'labels_used_for_inference':False}
    if args.mask:
        mask=np.load(args.mask,allow_pickle=False)
        if mask.shape==(160,5024):mask=mask[16:144]
        if mask.shape!=(128,5024) or not np.isin(mask,[0,1]).all():raise ValueError('Expected an aligned binary 160x5024 or 128x5024 reference mask')
        truth=metrics.regions(mask)[1]
        report['evaluation']={m:metrics.match(boxes,truth,5024) for m,boxes in result['boxes'].items()}
    fig=Figure(figsize=(12,6));engine.draw_result(fig,result)
    fig.savefig(args.output/'detections.png',dpi=150)
    np.savez_compressed(args.output/'predictions.npz',**{f'{m}_score':s for m,s in result['scores'].items()})
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'output':str(args.output),'boxes':{m:len(b) for m,b in result['boxes'].items()},'evaluation':report.get('evaluation')}))

def main():
    parser=argparse.ArgumentParser(description='Thesis-selected CNN and corrected-median demonstration')
    parser.add_argument('mode',choices=['demo','gui'],nargs='?',default='demo')
    parser.add_argument('--input',type=Path,default=ROOT/'examples/open_intrinsic.npy')
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'demo_output')
    args=parser.parse_args();engine=install_portable_inspector()
    if args.mode=='gui':
        from inspection.studio_gui import create_app
        create_app().run()
    else:demo(engine,args)

if __name__=='__main__':main()
