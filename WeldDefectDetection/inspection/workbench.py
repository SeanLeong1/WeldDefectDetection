"""Current circular workflow layered over the preserved historical GUI source."""
import json,os,queue,threading,time,uuid
import tkinter as tk
from tkinter import ttk,messagebox
from pathlib import Path
from .registry import ROOT,CURRENT,RASTER

def generate_example(seed,variant):
    from dataclasses import replace,asdict
    import numpy as np
    from circular_robustness import campaign as c
    cfg=c.g.old.load_config_bundle(ROOT/'DataSimulation/config')
    sample,_=c.g.baseline.c.sampled(cfg,seed,0);plan=c.plans(seed,seed%24)[variant]
    sample=replace(sample,defects=plan,randomize_defect_location=True)
    raw,quiet,mask,instances,metadata=c.render(c.Engine(seed,.05),sample)
    folder=ROOT/'data/current_circular_examples'/f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True);path=folder/f'{variant}.npz'
    np.savez_compressed(path,raw=raw,quiet=quiet,mask=mask,instances=instances,pixel_size_mm=.05)
    c.write(path.with_name(path.name+'.json'),dict(schema='circular-annulus-preview/v1',channel='topo_magnitude',pixel_size_mm=.05,
        radius_mm=40,seed=seed,variant=variant,parameters=asdict(sample),renderer_metadata=metadata,
        generation_source_sha256={str(p):c.sha(p) for p in (Path(__file__),Path(c.__file__),Path(c.geo.__file__))},
        source_kind='synthetic_preview',independent_test_evidence=False,physical_validation=False))
    return path

class CurrentPanel(ttk.Frame):
    def __init__(self,app):
        super().__init__(app.notebook,padding=25);self.app=app;self.events=queue.Queue();self.busy=False
        ttk.Label(self,text='Circular weld study | Current workflow',style='Heading.TLabel').pack(anchor='w')
        ttk.Label(self,text='80 mm circular weld • raw TOPO • whole-ring CNN and neighbouring median',style='Muted.TLabel').pack(anchor='w',pady=8)
        ttk.Label(self,text='The current trained model is circular study v4 (4 October). The robustness extension tests that same frozen model.\n'
            'Historical crop models and the older interactive generator remain available with explicit labels.\n'
            'Unfamiliar weld profiles are paused. Images and results are synthetic; no measured-weld validation is claimed.',wraplength=1000).pack(anchor='w',pady=8)
        links=ttk.Frame(self);links.pack(anchor='w',pady=10)
        for name,file in [('Study results','CIRCULAR_STUDY_RESULTS_20261004.md'),('Robustness results','CIRCULAR_ROBUSTNESS_RESULTS_20261004.md'),('Methods and layers','CIRCULAR_STUDY_METHODS_20261004.md'),('Version history','PROJECT_CURRENT.md')]:
            ttk.Button(links,text=name,command=lambda f=file:self.open_file(ROOT/'docs'/f)).pack(side='left',padx=4)
        box=ttk.LabelFrame(self,text='Generate a current circular example',padding=18);box.pack(fill='x',pady=18)
        self.seed=tk.StringVar(value='2260400000');self.variant=tk.StringVar(value='pair_near')
        ttk.Label(box,text='Scene').grid(row=0,column=0,sticky='w')
        self.variant_box=ttk.Combobox(box,textvariable=self.variant,state='readonly',values=('clean','single_keyhole','single_intrinsic','pair_far','pair_near','pair_overlap','triple'),width=25)
        self.variant_box.grid(row=0,column=1,padx=12,sticky='w')
        ttk.Label(box,text='Seed').grid(row=1,column=0,sticky='w',pady=12)
        ttk.Entry(box,textvariable=self.seed,width=25).grid(row=1,column=1,padx=12,sticky='w')
        ttk.Label(box,text='Uses the current recessed circular surface, assumed raster acquisition and fixed 0.05 mm sampling.\n'
            'Saves all raw channels and labels; inference only reads TOPO. A new folder preserves every generated example.',wraplength=850).grid(row=2,column=0,columnspan=3,sticky='w',pady=8)
        self.generate_button=ttk.Button(box,text='Generate and load for inspection',command=self.generate);self.generate_button.grid(row=3,column=0,columnspan=2,sticky='w',pady=8)
        ttk.Button(self,text='Load the frozen study example',command=self.load_reference).pack(anchor='w')
        self.status=tk.StringVar(value='Ready. Generated examples are exploratory; they are not added to a training or test split.')
        ttk.Label(self,textvariable=self.status,wraplength=1000,style='Muted.TLabel').pack(anchor='w',pady=18)
        self.after(120,self.poll)

    def open_file(self,path):
        if path.exists():os.startfile(str(path))
        else:messagebox.showinfo('Report unavailable','This report has not been generated in this workspace yet.',parent=self)

    def select_current(self):
        panel=self.app.inspection_panel
        label=next(k for k,v in panel.campaigns.items() if v==CURRENT)
        panel.campaign.set(label);panel._campaign_changed();return panel

    def load_reference(self):
        panel=self.select_current();panel.example();self.app.notebook.select(panel)

    def generate(self):
        if self.busy:return
        if getattr(self.app,'_batch_running',False):
            self.status.set('Wait for historical generation to finish before starting another render.');return
        try:
            seed=int(self.seed.get())
            if not 0<=seed<=3_000_000_000:raise ValueError('Use an integer seed from 0 to 3000000000.')
        except ValueError as exc:messagebox.showerror('Seed',str(exc),parent=self);return
        variant=self.variant.get();self.busy=True;self.generate_button.configure(state='disabled')
        for tab in self.app.notebook.tabs():
            if self.app.notebook.tab(tab,'text')=='Historical generator':self.app.notebook.tab(tab,state='disabled')
        self.status.set('Rendering the full circular annulus locally. This normally takes tens of seconds.')
        def work():
            try:self.events.put(('done',generate_example(seed,variant)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            kind,value=self.events.get_nowait();self.busy=False;self.generate_button.configure(state='normal')
            for tab in self.app.notebook.tabs():
                if self.app.notebook.tab(tab,'text')=='Historical generator':self.app.notebook.tab(tab,state='normal')
            if kind=='error':self.status.set(value)
            else:
                panel=self.select_current();panel.source.set(str(value));panel.mode.set('Raw TOPO (training units)')
                panel.spacing.set('.05');panel.method.set('Compare CNN and median');self.app.notebook.select(panel)
                self.status.set(f'Saved {value}');panel.status.set('New circular scene loaded. Run detection to inspect the frozen CNN and median.')
        except queue.Empty:pass
        self.after(120,self.poll)

def create_app(config):
    from bse_renderer.gui import App
    class CurrentApp(App):
        def _poll_batch(self):
            # Adapt historical generator hand-off without changing its frozen source.
            previous=self.inspection_panel.source.get()
            super()._poll_batch()
            if self.inspection_panel.source.get()!=previous:
                panel=self.inspection_panel
                label=next((k for k,v in panel.campaigns.items() if v==RASTER),None)
                if label:panel.campaign.set(label);panel._campaign_changed()
        def _generate_batch(self):
            if getattr(self,'current_panel',None) is not None and self.current_panel.busy:
                self.status_var.set('Wait for current circular generation to finish.');return
            super()._generate_batch()
    app=CurrentApp(config);app.root.title('Weld Vision | Circular study, 4 October 2026')
    for tab in app.notebook.tabs():
        label=app.notebook.tab(tab,'text')
        if label=='Overview':app.notebook.tab(tab,text='Historical overview')
        if label=='Generate images':app.notebook.tab(tab,text='Historical generator')
    app.current_panel=CurrentPanel(app);app.notebook.add(app.current_panel,text='Current circular workflow')
    app.notebook.insert(0,app.current_panel);app.notebook.select(app.current_panel)
    return app
