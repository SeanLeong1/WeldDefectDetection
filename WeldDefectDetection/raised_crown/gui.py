"""Raised-crown workflow alongside the preserved recessed/historical workbench."""
import json,queue,threading,time,uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox
import numpy as np
from . import campaign as c,learning as learn,rendering as r

def infer(path):
    from circular_study import metrics
    import torch
    torch.set_num_threads(4)
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    rendering=meta.get('metadata',meta);geometry=rendering.get('geometry',{})
    version=geometry.get('version')
    if version!='raised-crown-v1':raise ValueError('This model requires raised-crown geometry metadata; use historical inspection for other images.')
    expected=dict(radius_mm=40.,crown_height_mm=3.,crown_width_mm=2.8,penetration_depth_mm=7.,
        fusion_surface_width_mm=2.8,fusion_width_at_5mm=1.2,pixel_size_mm=.05)
    if any(geometry.get(k)!=v for k,v in expected.items()):raise ValueError('Geometry or physical sampling does not match the trained crown contract.')
    if (rendering.get('profile')!='raised' or
        rendering.get('visibility_resolution')!=dict(radial_mm=.1,angular_stride=4,ray_step_mm=.1) or
        rendering.get('renderer_metadata',{}).get('emission_b',{}).get('mode')!='lambertian' or
        rendering.get('acquisition',{}).get('version')!='raster_v2'):
        raise ValueError('Surface profile, visibility, emission or acquisition differs from the trained renderer contract.')
    if meta.get('npz_sha256') and c.sha(path)!=meta['npz_sha256']:raise ValueError('Image checksum does not match its metadata.')
    frozen=learn.verify_freeze();norm=c.read(c.OUT/'normalization.json')
    with np.load(path) as a:raw=a['raw'][0].copy();height=a['height'].copy()
    if raw.shape!=(160,5024):raise ValueError('Expected 160x5024 raw TOPO at 0.05 mm radial pitch.')
    # No labels are read or passed into inference.
    image=learn.normalize(raw,norm);model=learn.model_for(frozen['cnn'])
    scores=dict(cnn=learn.predict_image(image,model),median=learn.median_image(image,frozen['median']['spec']))
    masks={};boxes={}
    for method,score in scores.items():
        op=frozen[method]['selection']['best'];masks[method],boxes[method]=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
    return dict(raw=raw,height=height,scores=scores,masks=masks,boxes=boxes,metadata=meta)

def generate(seed,kind):
    from dataclasses import replace,asdict
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config');sample,_=r.oldg.baseline.c.sampled(cfg,seed,0)
    sample=replace(sample,defects=c.plans(seed,0)[kind]);engine=r.Engine(seed,.05)
    arrays,meta=r.render(engine,sample)
    folder=c.ROOT/'data/raised_crown_examples'/f'{seed}_{kind}_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True);path=folder/'scene.npz';np.savez_compressed(path,**arrays)
    c.write(path.with_suffix('.json'),dict(metadata=meta,parameters=asdict(sample),research_split=False,npz_sha256=c.sha(path)))
    return path

class RaisedPanel(ttk.Frame):
    def __init__(self,app):
        super().__init__(app.notebook,padding=15);self.app=app;self.events=queue.Queue();self.busy=False
        ttk.Label(self,text='Raised circular crown | 3 mm height | 7 mm fusion depth',style='Heading.TLabel').pack(anchor='w')
        ttk.Label(self,text='Synthetic research model: geometric shadows + uncalibrated Lambertian response. No new electron-depth cutoff.\n'
            'Open pore, assumed volume-displacement bulge, and hidden-pore negative control are distinct cases.',wraplength=1000).pack(anchor='w',pady=8)
        result_path=c.OUT/'results.json'
        if result_path.exists():
            result=c.read(result_path);cnn=result['results'][result['selected']]['metrics'];median=result['results']['median']['metrics']
            ttk.Label(self,text=f'Frozen confirmation: CNN F1 {cnn["f1"]:.3f}, clean rings flagged {cnn["clean_rings_flagged"]}/{cnn["clean_rings"]}; '
                f'median F1 {median["f1"]:.3f}, clean rings flagged {median["clean_rings_flagged"]}/{median["clean_rings"]}.',wraplength=1000).pack(anchor='w',pady=4)
        bar=ttk.Frame(self);bar.pack(fill='x');self.kind=tk.StringVar(value='open_keyhole');self.seed=tk.StringVar(value='2360500000')
        ttk.Combobox(bar,textvariable=self.kind,values=tuple(dict.fromkeys(c.KINDS+c.STRESS)),state='readonly',width=20).pack(side='left')
        ttk.Entry(bar,textvariable=self.seed,width=15).pack(side='left',padx=5)
        ttk.Button(bar,text='Generate and inspect',command=lambda:self.start(True)).pack(side='left',padx=5)
        ttk.Button(bar,text='Load evaluated example',command=lambda:self.start(False)).pack(side='left',padx=5)
        def open_report(name):
            import os
            os.startfile(c.ROOT/'docs'/name)
        ttk.Button(bar,text='Results',command=lambda:open_report('RAISED_CROWN_RESULTS_20261005.md')).pack(side='left',padx=5)
        ttk.Button(bar,text='How the methods work',command=lambda:open_report('RAISED_CROWN_METHODS_20261005.md')).pack(side='left',padx=5)
        self.status=tk.StringVar(value='Ready. Historical datasets and models remain on their own tabs.')
        ttk.Label(self,textvariable=self.status,wraplength=1000).pack(anchor='w',pady=8)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.figure=Figure(figsize=(11,6),dpi=90);self.canvas=FigureCanvasTkAgg(self.figure,master=self)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.after(100,self.poll)

    def start(self,new):
        if self.busy:return
        if getattr(self.app.current_panel,'busy',False) or getattr(self.app,'_batch_running',False):
            self.status.set('Wait for the other renderer to finish.');return
        try:seed=int(self.seed.get());assert 0<=seed<=3_000_000_000
        except (ValueError,AssertionError):messagebox.showerror('Seed','Use an integer from 0 to 3000000000.',parent=self);return
        kind=self.kind.get();self.busy=True;self.status.set('Rendering and checking the versioned model. This may take several minutes.')
        self.previous_tab_states={tab:self.app.notebook.tab(tab,'state') for tab in self.app.notebook.tabs() if tab!=str(self)}
        for tab in self.previous_tab_states:self.app.notebook.tab(tab,state='disabled')
        def work():
            try:
                path=generate(seed,kind) if new else c.OUT/'datasets/test/test-000-open_keyhole.npz'
                self.events.put(('done',(path,infer(path))))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def display(self,value):
        path,result=value;self.figure.clear();axes=self.figure.subplots(1,3)
        from .geometry import height,fusion_half_width
        u=np.linspace(-2,2,801);z=height(u,0,0,0);d=np.linspace(0,7,300)
        axes[0].fill_between(u,0,z,color='#df9c45');axes[0].fill_betweenx(-d,-fusion_half_width(d),fusion_half_width(d),color='#df9c45')
        axes[0].axhline(0,color='k',lw=.5);axes[0].set(title='Nominal solid fusion + crown',xlabel='Across weld (mm)',ylabel='Height (mm)',aspect='equal')
        score=result['scores']['cnn'];column=int(np.argmax(score.max(axis=0)));cols=(np.arange(128)+column-64)%5024
        hmap=axes[1].imshow(result['height'][:,cols],origin='lower',aspect='equal',cmap='terrain',extent=(-3.2,3.2,-4,4))
        axes[1].set(title='Actual surface height',xlabel='Arc offset (mm)',ylabel='Radial offset (mm)')
        self.figure.colorbar(hmap,ax=axes[1],shrink=.6,label='Height (mm)')
        axes[2].imshow(result['raw'][r.SCORE][:,cols],origin='lower',cmap='gray',aspect='equal',extent=(-3.2,3.2,-3.2,3.2))
        for method,color in [('cnn','cyan'),('median','orange')]:
            mask=result['masks'][method][:,cols]
            if mask.any():axes[2].contour(np.linspace(-3.175,3.175,128),np.linspace(-3.175,3.175,128),mask,levels=[.5],colors=[color],linewidths=.8)
        axes[2].set_title('TOPO: CNN cyan / median orange')
        axes[2].set(xlabel='Arc offset (mm)',ylabel='Radial offset (mm)')
        self.figure.suptitle('Image neighbourhood centred on strongest CNN score; predictions cover the full ring',fontsize=10)
        self.figure.tight_layout();self.canvas.draw()
        output=path.parent/'inspection';output.mkdir(exist_ok=True)
        self.figure.savefig(output/'preview.png',dpi=150)
        from matplotlib.figure import Figure
        full=Figure(figsize=(15,4),dpi=100);fax=full.subplots(3,1,sharex=True)
        fax[0].imshow(result['raw'][r.SCORE],origin='lower',aspect='auto',cmap='gray',extent=(0,360,-3.2,3.2));fax[0].set_ylabel('TOPO')
        for ax,method in zip(fax[1:],('cnn','median')):
            ax.imshow(result['masks'][method],origin='lower',aspect='auto',cmap='gray',vmin=0,vmax=1,extent=(0,360,-3.2,3.2));ax.set_ylabel(method)
        fax[-1].set_xlabel('Angle (degrees)');full.suptitle('Complete ring: raw TOPO and predicted regions, radial extent +/-3.2 mm');full.tight_layout()
        full.savefig(output/'complete_ring.png',dpi=150)
        np.savez_compressed(output/'predictions.npz',**{k+'_score':v for k,v in result['scores'].items()},**{k+'_mask':v for k,v in result['masks'].items()})
        c.write(output/'prediction.json',dict(source=str(path),source_sha256=c.sha(path),boxes=result['boxes'],model_manifest_sha256=c.sha(c.OUT/'frozen_models.json'),labels_used=False,physical_validation=False))
        self.status.set(f'Saved {output}. CNN regions: {len(result["boxes"]["cnn"])}; median: {len(result["boxes"]["median"])}. Synthetic evidence only.')

    def poll(self):
        try:
            kind,value=self.events.get_nowait();self.busy=False
            for tab,state in self.previous_tab_states.items():self.app.notebook.tab(tab,state=state)
            if kind=='error':self.status.set(value)
            else:self.display(value)
        except queue.Empty:pass
        self.after(100,self.poll)

def create_app(config):
    from inspection.workbench import create_app as historical_app
    app=historical_app(config);panel=RaisedPanel(app);app.raised_panel=panel
    app.notebook.add(panel,text='Raised crown | 5 Oct');app.notebook.insert(0,panel);app.notebook.select(panel)
    app.notebook.tab(app.current_panel,text='Historical recessed | 4 Oct')
    for tab in app.notebook.tabs():
        if app.notebook.tab(tab,'text')=='Inspect images':app.notebook.tab(tab,text='Historical inspection')
    historical=app.inspection_panel
    selected=historical.campaigns[historical.campaign.get()]
    historical.campaigns={label.replace('Current |','Historical |'):path for label,path in historical.campaigns.items()}
    historical.campaign_box.configure(values=list(historical.campaigns))
    historical.campaign.set(next(label for label,path in historical.campaigns.items() if path==selected))
    historical._campaign_changed()
    app.root.title('Weld Vision | Raised crown synthetic study and history')
    old_generate=app.current_panel.generate
    def guarded_generate():
        if panel.busy:app.current_panel.status.set('Wait for raised-crown work to finish.');return
        old_generate()
    app.current_panel.generate_button.configure(command=guarded_generate)
    old_batch=app._generate_batch
    def guarded_batch():
        if panel.busy:app.status_var.set('Wait for raised-crown work to finish.');return
        old_batch()
    app._generate_batch=guarded_batch
    app.generate_button.configure(command=guarded_batch)
    return app
