"""Acquisition-aware inspection; historical workbenches remain separate."""
from dataclasses import replace,asdict
from pathlib import Path
import queue,threading,uuid
import tkinter as tk
from tkinter import ttk,messagebox
import numpy as np
from circular_study import metrics
from raised_crown import rendering as r, learning as base, campaign as prior
from . import campaign as c, learning as learn


def infer(path,arm=None):
    import torch
    torch.set_num_threads(4); path=Path(path); meta=c.read(path.with_suffix('.json'))
    rendering=meta.get('metadata',meta); geometry=rendering.get('geometry',{})
    expected=prior.geo.specification()
    if geometry!=expected: raise ValueError('Geometry differs from the frozen raised-crown contract')
    if (rendering.get('profile')!='raised' or rendering.get('visibility_resolution')!=dict(radial_mm=.1,angular_stride=4,ray_step_mm=.1)
        or rendering.get('renderer_metadata',{}).get('emission_b',{}).get('mode')!='lambertian'
        or rendering.get('acquisition',{}).get('version')!='raster_v2'):
        raise ValueError('Image does not match the frozen renderer contract')
    if meta.get('npz_sha256') and c.sha(path)!=meta['npz_sha256']: raise ValueError('Image checksum mismatch')
    frozen=learn.verify_freeze(); arm=arm or frozen['selected_arm']
    if arm not in c.ARMS: raise ValueError('Unknown trained arm')
    with np.load(path) as a: raw=a['raw'][0].copy(); height=a['height'].copy()
    if raw.shape!=(160,5024) or not np.isfinite(raw).all(): raise ValueError('Expected finite 160x5024 raw TOPO')
    image=base.normalize(raw,c.read(c.OUT/'normalization.json'))
    scores=dict(cnn=base.predict_image(image,base.model_for(frozen['arms'][arm]['cnn'])),
        median=base.median_image(image,frozen['median']['spec']))
    masks={}; boxes={}
    for method,score in scores.items():
        spec=frozen['arms'][arm]['cnn'] if method=='cnn' else frozen['median']; op=spec['selection']['best']
        masks[method],boxes[method]=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
    return dict(raw=raw,height=height,scores=scores,masks=masks,boxes=boxes,metadata=meta,arm=arm,
        manifest_sha256=c.sha(c.OUT/'frozen_models.json'))


def generate(seed,kind,dwell_us,index=0):
    if not .1<=dwell_us<=20: raise ValueError('Use the studied dwell range, 0.1 to 20 microseconds')
    if kind not in set(prior.KINDS+prior.STRESS): raise ValueError('Unknown pore condition')
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config'); sample,_=r.oldg.baseline.c.sampled(cfg,seed,index)
    sample=replace(sample,beam=replace(sample.beam,dwell_time_us=float(dwell_us)),defects=prior.plans(seed,index)[kind])
    engine=r.Engine(seed,(0.,.05,.1)[(index//3+index)%3]); arrays,metadata=r.render(engine,sample)
    folder=c.ROOT/'data/acquisition_examples'/f'{seed}_{kind}_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True); path=folder/'scene.npz'; np.savez_compressed(path,**arrays)
    c.write(path.with_suffix('.json'),dict(metadata=metadata,parameters=asdict(sample),research_split=False,
        index=index,npz_sha256=c.sha(path),dwell_time_us=float(dwell_us)))
    return path


def export(path,result,figure=None):
    from matplotlib.figure import Figure
    figure=figure or Figure(figsize=(12,8),dpi=100); figure.clear(); axes=figure.subplots(4,1,sharex=True)
    axes[0].imshow(result['height'][r.SCORE],origin='lower',aspect='auto',cmap='terrain',extent=(0,360,-3.2,3.2))
    axes[0].set_ylabel('Height\nradial mm')
    axes[1].imshow(result['raw'][r.SCORE],origin='lower',aspect='auto',cmap='gray',extent=(0,360,-3.2,3.2))
    axes[1].set_ylabel('TOPO\nradial mm')
    for ax,method in zip(axes[2:],('cnn','median')):
        ax.imshow(result['masks'][method],origin='lower',aspect='auto',cmap='gray',vmin=0,vmax=1,extent=(0,360,-3.2,3.2))
        ax.set_ylabel(method+'\nradial mm')
    axes[-1].set_xlabel('Around circular weld (degrees)')
    dwell=result['metadata'].get('dwell_time_us',result['metadata'].get('parameters',{}).get('beam',{}).get('dwell_time_us'))
    figure.suptitle(f'6 Oct acquisition study | {result["arm"]} | dwell {dwell:.3g} us\nFrozen operating points; complete ring predictions; synthetic evidence')
    figure.tight_layout(); output=Path(path).parent/'acquisition_inspection'/Path(path).stem/result['arm']; output.mkdir(parents=True,exist_ok=True)
    figure.savefig(output/'complete_ring.png',dpi=150)
    # Show a useful local view in the GUI while retaining the complete-ring export.
    # The crop is selected from predictions, never from the truth mask.
    column=int(np.argmax(result['scores']['cnn'].max(axis=0))); columns=(np.arange(128)+column-64)%5024
    figure.clear(); axes=figure.subplots(1,3)
    height=axes[0].imshow(result['height'][:,columns],origin='lower',aspect='equal',cmap='terrain',extent=(-3.2,3.2,-4,4))
    axes[0].set(title='Actual crown height',xlabel='Arc offset (mm)',ylabel='Radial offset (mm)')
    figure.colorbar(height,ax=axes[0],shrink=.7,label='Height above datum (mm)')
    topo=result['raw'][r.SCORE][:,columns]
    for ax,method in zip(axes[1:],('cnn','median')):
        ax.imshow(topo,origin='lower',aspect='equal',cmap='gray',extent=(-3.2,3.2,-3.2,3.2))
        mask=result['masks'][method][:,columns]
        if mask.any(): ax.contour(np.linspace(-3.175,3.175,128),np.linspace(-3.175,3.175,128),mask,levels=[.5],colors=['cyan' if method=='cnn' else 'orange'],linewidths=1.)
        ax.set(title=f'TOPO + {method} predictions',xlabel='Arc offset (mm)',ylabel='Radial offset (mm)')
    figure.suptitle(f'{result["arm"]} | dwell {dwell:.3g} us | strongest CNN-score neighbourhood\n'
        f'Full-ring regions: CNN {len(result["boxes"]["cnn"])}, median {len(result["boxes"]["median"])}. Full-ring export saved separately; synthetic evidence')
    figure.tight_layout(); figure.savefig(output/'preview.png',dpi=150)
    np.savez_compressed(output/'predictions.npz',**{k+'_score':v for k,v in result['scores'].items()},**{k+'_mask':v for k,v in result['masks'].items()})
    c.write(output/'prediction.json',dict(source=str(path),source_sha256=c.sha(path),boxes=result['boxes'],arm=result['arm'],
        model_manifest_sha256=result['manifest_sha256'],labels_used=False,physical_validation=False))
    return output


class AcquisitionPanel(ttk.Frame):
    def __init__(self,app):
        super().__init__(app.notebook,padding=12); self.app=app; self.busy=False; self.events=queue.Queue()
        frozen=learn.verify_freeze(); self.arm=tk.StringVar(value=frozen['selected_arm']); self.kind=tk.StringVar(value='open_intrinsic')
        self.dwell=tk.StringVar(value='0.2'); self.seed=tk.StringVar(value='2450600000'); self.index=tk.StringVar(value='0')
        ttk.Label(self,text='Circular crown | acquisition coverage and clean-negative training | 6 Oct',style='Heading.TLabel').pack(anchor='w')
        ttk.Label(self,text='Same 3 mm crown and surface renderer. Dwell affects the existing raster acquisition model.\n'
            'Choose either frozen CNN arm; the median comparator stays fixed. Generated examples stay outside research splits.',wraplength=1050).pack(anchor='w',pady=6)
        result=c.read(c.OUT/'results.json'); parts=[]
        for arm in c.ARMS:
            m=result['results'][result['selected'][arm]]['metrics']
            parts.append(f'{arm}: F1 {m["f1"]:.3f}, clean alarms {m["clean_rings_flagged"]}/{m["clean_rings"]}')
        ttk.Label(self,text='Fresh balanced-dwell confirmation: '+'; '.join(parts),wraplength=1050).pack(anchor='w',pady=5)
        robustness=c.read(c.OUT/'robustness_results.json')['groups']; details=[]
        for arm in c.ARMS:
            group=robustness[arm]
            details.append(f'{arm}: known coordinates {group["nominal"]["metrics"]["f1"]:.3f}, image-estimated {group["image_estimated"]["metrics"]["f1"]:.3f}')
        ttk.Label(self,text='Localisation remains a limitation. F1 on the same 84 stress scenes: '+'; '.join(details)+
            '. Generated examples here use declared ring coordinates.',wraplength=1050).pack(anchor='w',pady=4)
        bar=ttk.Frame(self); bar.pack(fill='x')
        ttk.Combobox(bar,textvariable=self.arm,values=c.ARMS,state='readonly',width=17).pack(side='left')
        ttk.Combobox(bar,textvariable=self.kind,values=tuple(dict.fromkeys(prior.KINDS+prior.STRESS)),state='readonly',width=17).pack(side='left',padx=4)
        for label,var,width in [('Dwell us',self.dwell,7),('Seed',self.seed,12),('Variation 0-23',self.index,4)]:
            ttk.Label(bar,text=label).pack(side='left',padx=3); ttk.Entry(bar,textvariable=var,width=width).pack(side='left')
        actions=ttk.Frame(self); actions.pack(fill='x',pady=5)
        ttk.Button(actions,text='Generate and inspect',command=lambda:self.start(True)).pack(side='left')
        ttk.Button(actions,text='Load evaluated example',command=lambda:self.start(False)).pack(side='left',padx=5)
        def open_report():
            import os
            os.startfile(c.ROOT/'docs/RAISED_ACQUISITION_RESULTS_20261006.md')
        ttk.Button(actions,text='Results and method changes',command=open_report).pack(side='left')
        self.status=tk.StringVar(value='Ready. Default arm was selected on validation before confirmation.')
        ttk.Label(self,textvariable=self.status,wraplength=1050).pack(anchor='w',pady=5)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.figure=Figure(figsize=(12,6),dpi=90); self.canvas=FigureCanvasTkAgg(self.figure,master=self)
        self.canvas.get_tk_widget().pack(fill='both',expand=True); self.after(100,self.poll)

    def start(self,new):
        if self.busy or self.app.raised_panel.busy or self.app.current_panel.busy or getattr(self.app,'_batch_running',False):
            self.status.set('Wait for active work to finish.'); return
        try:
            dwell=float(self.dwell.get()); seed=int(self.seed.get()); index=int(self.index.get())
            if not (.1<=dwell<=20 and 0<=seed<=3_000_000_000 and 0<=index<24): raise ValueError()
        except ValueError:
            messagebox.showerror('Parameters','Use dwell 0.1-20 us, an integer seed 0-3000000000, and layout 0-23.',parent=self); return
        kind=self.kind.get(); arm=self.arm.get(); self.busy=True; self.status.set('Rendering or loading, then applying frozen models...')
        self.tab_states={tab:self.app.notebook.tab(tab,'state') for tab in self.app.notebook.tabs() if tab!=str(self)}
        for tab in self.tab_states: self.app.notebook.tab(tab,state='disabled')
        def work():
            try:
                if new: path=generate(seed,kind,dwell,index)
                else:
                    split='test' if kind in (*prior.KINDS,'hidden') else 'stress'
                    rows=[x for x in c.read(c.OUT/f'datasets/{split}/index.json') if x['source_class']==kind]
                    path=Path(min(rows,key=lambda x:abs(np.log(x['dwell_time_us']/dwell)))['npz_path'])
                    if not path.is_absolute(): path=c.ROOT/path
                self.events.put(('done',(path,infer(path,arm))))
            except Exception as exc: self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            kind,value=self.events.get_nowait(); self.busy=False
            for tab,state in self.tab_states.items(): self.app.notebook.tab(tab,state=state)
            if kind=='error': self.status.set(value)
            else:
                path,result=value; output=export(path,result,self.figure); self.canvas.draw()
                self.status.set(f'Saved {output}. CNN regions {len(result["boxes"]["cnn"])}, median {len(result["boxes"]["median"])}.')
        except queue.Empty: pass
        self.after(100,self.poll)


def create_app(config):
    from raised_crown.gui import create_app as previous_app
    app=previous_app(config); panel=AcquisitionPanel(app); app.acquisition_panel=panel
    app.notebook.add(panel,text='Acquisition study | 6 Oct'); app.notebook.insert(0,panel); app.notebook.select(panel)
    app.notebook.tab(app.raised_panel,text='Historical crown | 5 Oct')
    app.root.title('Weld Vision | circular acquisition study and preserved history')
    old_batch=app._generate_batch
    def batch_guard():
        if panel.busy: app.status_var.set('Wait for acquisition-study work to finish.'); return
        old_batch()
    app._generate_batch=batch_guard; app.generate_button.configure(command=batch_guard)
    return app
