"""Current thesis methods in the existing Weld Vision notebook."""
from pathlib import Path
import os,queue,threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from . import final_methods as engine

METHODS={'Compare CNN and Median':('cnn','median'),'CNN':('cnn',),'Median':('median',)}


class FinalPanel(ttk.Frame):
    def __init__(self,app):
        super().__init__(app.notebook,padding=12)
        self.app=app;self.busy=False;self.events=queue.Queue();self.inspector=None;self.result=None;self.output=None
        self.source=tk.StringVar(value=str(engine.example_path()))
        self.method=tk.StringVar(value=next(iter(METHODS)))
        self.status=tk.StringVar(value='Ready. Load the example or choose a raw TOPO array, then run detection.')
        self.seed=tk.StringVar(value='2811000023');self.kind=tk.StringVar(value='open_intrinsic')
        self.counts=tk.StringVar(value='64');self.dwell=tk.StringVar(value='0.2')
        ttk.Label(self,text='Final thesis methods | CNN and Median',style='Heading.TLabel').pack(anchor='w')
        ttk.Label(self,text='Mixed-noise CNN: 96 training backgrounds. Median: selected filtering and localisation settings.',wraplength=1050).pack(anchor='w',pady=(2,8))
        bar=ttk.Frame(self);bar.pack(fill='x');bar.columnconfigure(0,weight=1)
        ttk.Entry(bar,textvariable=self.source).grid(row=0,column=0,sticky='ew',padx=(0,6))
        ttk.Button(bar,text='Add image',command=self.choose).grid(row=0,column=1,padx=3)
        ttk.Button(bar,text='Load example',command=self.example).grid(row=0,column=2,padx=3)
        ttk.Combobox(bar,textvariable=self.method,values=list(METHODS),state='readonly',width=25).grid(row=0,column=3,padx=5)
        self.run_button=ttk.Button(bar,text='Run detection',command=self.start,style='Accent.TButton');self.run_button.grid(row=0,column=4)
        ttk.Label(self,text='Input: raw .npy/.npz TOPO, 160 x 5024 unwrapped pixels; 40 mm ring radius and 0.05 mm radial pitch. PNG/JPEG imports require calibrated ring extraction.',wraplength=1100,style='Muted.TLabel').pack(anchor='w',pady=(6,8))
        gen=ttk.LabelFrame(self,text='Create a new image');gen.pack(fill='x')
        for label,var,width in [('Seed',self.seed,14),('Dwell (us)',self.dwell,7)]:
            ttk.Label(gen,text=label).pack(side='left',padx=4);ttk.Entry(gen,textvariable=var,width=width).pack(side='left')
        ttk.Combobox(gen,textvariable=self.kind,values=['clean','open_intrinsic','open_keyhole'],state='readonly',width=17).pack(side='left',padx=8)
        ttk.Label(gen,text='Counts').pack(side='left');ttk.Combobox(gen,textvariable=self.counts,values=['1','4','64'],state='readonly',width=5).pack(side='left',padx=5)
        self.generate_button=ttk.Button(gen,text='Generate and inspect',command=lambda:self.start(True));self.generate_button.pack(side='left',padx=8)
        ttk.Button(gen,text='Open results',command=self.open_results).pack(side='right')
        ttk.Label(self,textvariable=self.status,wraplength=1100).pack(anchor='w',pady=7)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.figure=Figure(figsize=(12,5),dpi=90);self.canvas=FigureCanvasTkAgg(self.figure,master=self)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.figure.text(.5,.5,'Load an example to view the full ring and detection overlays.',ha='center',fontsize=15)
        self.canvas.draw();self.after(100,self.poll)

    def choose(self):
        if self.busy:return
        value=filedialog.askopenfilename(parent=self,title='Select raw annular TOPO',filetypes=[('Raw TOPO arrays','*.npy *.npz')])
        if value:self.source.set(value)

    def example(self):
        if not self.busy:self.source.set(str(engine.example_path()));self.start()

    def start(self,generate=False):
        if self.busy:return
        if getattr(self.app,'_batch_running',False) or any(getattr(getattr(self.app,k,None),'busy',False) for k in ('current_panel','raised_panel','acquisition_panel','inspection_panel')):
            self.status.set('Wait for the other task to finish.');return
        path=self.source.get();methods=METHODS[self.method.get()]
        try:
            settings=(int(self.seed.get()),self.kind.get(),int(self.counts.get()),float(self.dwell.get())) if generate else None
        except ValueError:
            messagebox.showerror('Generation settings','Seed and counts must be integers; dwell must be numeric.',parent=self);return
        self.busy=True;self.run_button.configure(state='disabled');self.generate_button.configure(state='disabled')
        self.tab_states={tab:self.app.notebook.tab(tab,'state') for tab in self.app.notebook.tabs() if tab!=str(self)}
        for tab in self.tab_states:self.app.notebook.tab(tab,state='disabled')
        self.status.set('Generating TOPO and running final methods...' if generate else 'Checking final model files and running detection...')
        def work():
            try:
                if self.inspector is None:self.inspector=engine.FinalInspector()
                actual=engine.generate(*settings) if generate else Path(path)
                result=self.inspector.infer(actual,methods)
                output=engine.export_result(result)
                self.events.put(('done',(result,output)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            event,value=self.events.get_nowait();self.busy=False
            self.run_button.configure(state='normal');self.generate_button.configure(state='normal')
            for tab,state in self.tab_states.items():self.app.notebook.tab(tab,state=state)
            if event=='error':self.status.set(value)
            else:
                self.result,self.output=value;self.source.set(self.result['source'])
                engine.draw_result(self.figure,self.result);self.canvas.draw()
                counts=' | '.join(f'{m.upper() if m=="cnn" else "Median"}: {len(b)} candidate(s)' for m,b in self.result['boxes'].items())
                self.status.set(f'{counts} | {self.result["seconds"]:.1f} s | Saved: {self.output}')
        except queue.Empty:pass
        self.after(100,self.poll)

    def open_results(self):
        if self.output:os.startfile(self.output)


def attach(app):
    panel=FinalPanel(app);app.final_panel=panel
    app.notebook.add(panel,text='Final methods | Thesis');app.notebook.insert(0,panel);app.notebook.select(panel)
    if hasattr(app,'acquisition_panel'):app.notebook.tab(app.acquisition_panel,text='Historical acquisition | 6 Oct')
    app.root.title('Weld Vision | Final thesis CNN and Median')
    return app
