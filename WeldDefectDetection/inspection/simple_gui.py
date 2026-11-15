"""Focused desktop workflow for the frozen thesis methods."""
from pathlib import Path
import os, queue, subprocess, sys, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import final_methods as engine

BG='#eef3f8';WHITE='#ffffff';NAVY='#15334e';TEXT='#203b50';MUTED='#61778b';BLUE='#146aab';LINE='#dce5ed'
KINDS={'Round pore':'open_intrinsic','Elongated pore':'open_keyhole','Clean weld':'clean'}
COUNTS={'Low noise (64x)':64,'Medium noise (4x)':4,'High noise (1x)':1}


class SimpleStudio:
    def __init__(self):
        self.root=tk.Tk();self.root.title('Weld Vision | Inspect a weld')
        self.root.geometry('1320x880');self.root.minsize(1120,760);self.root.configure(bg=BG)
        self.busy=False;self.inspector=None;self.result=None;self.output=None;self.events=queue.Queue()
        self.source=engine.example_path();self.source_name=tk.StringVar(value='Example weld ready')
        self.status=tk.StringVar(value='Choose an example, open a raw TOPO image, or create a weld.')
        self.kind=tk.StringVar(value='Round pore');self.noise=tk.StringVar(value='Low noise (64x)')
        self.seed=tk.StringVar(value='2811000023');self.dwell=tk.StringVar(value='0.2')
        self.view=tk.StringVar(value='Both');self.cnn_count=tk.StringVar(value='--');self.median_count=tk.StringVar(value='--')
        self.runtime=tk.StringVar(value='Ready');self.buttons=[];self.settings=[]
        self._theme();self._menu();self._build();self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(100,self.poll)

    def _theme(self):
        self.root.option_add('*Font',('Segoe UI',10))
        s=ttk.Style(self.root);s.theme_use('clam')
        s.configure('TButton',font=('Segoe UI',10),padding=(12,9),background=WHITE,foreground=TEXT,bordercolor=LINE,relief='flat')
        s.map('TButton',background=[('active','#e9f2fa'),('disabled','#edf1f5')],foreground=[('disabled','#8a9aa8')])
        s.configure('Primary.TButton',background=BLUE,foreground=WHITE,font=('Segoe UI',11,'bold'),padding=(12,11),bordercolor=BLUE)
        s.map('Primary.TButton',background=[('active','#10568c'),('disabled','#adc4d6')],foreground=[('disabled',WHITE)])
        s.configure('TCombobox',padding=6,fieldbackground=WHITE,background=WHITE,foreground=TEXT)
        s.configure('TEntry',padding=6,fieldbackground=WHITE)
        s.configure('View.TRadiobutton',background=WHITE,foreground=TEXT,padding=(9,6),font=('Segoe UI',10))
        s.configure('Horizontal.TProgressbar',background=BLUE,troughcolor=BG,borderwidth=0)

    def _menu(self):
        menu=tk.Menu(self.root);file=tk.Menu(menu,tearoff=False)
        file.add_command(label='Open image...',command=self.open_file)
        file.add_command(label='Open saved results',command=self.open_results)
        file.add_separator();file.add_command(label='Exit',command=self.close)
        menu.add_cascade(label='File',menu=file)
        tools=tk.Menu(menu,tearoff=False);tools.add_command(label='Open historical research tools',command=self.research)
        menu.add_cascade(label='Research tools',menu=tools)
        helpmenu=tk.Menu(menu,tearoff=False);helpmenu.add_command(label='Image requirements',command=self.input_help)
        helpmenu.add_command(label='About the final methods',command=self.about)
        menu.add_cascade(label='Help',menu=helpmenu);self.root.configure(menu=menu)

    def label(self,parent,text=None,variable=None,size=10,bold=False,color=TEXT,bg=WHITE,**kw):
        widget=tk.Label(parent,text=text,textvariable=variable,font=('Segoe UI',size,'bold' if bold else 'normal'),fg=color,bg=bg,anchor='w',**kw)
        return widget

    def card(self,parent,title):
        frame=tk.Frame(parent,bg=WHITE,highlightbackground=LINE,highlightthickness=1,padx=17,pady=15)
        frame.pack(fill='x',pady=(0,12));self.label(frame,title,size=12,bold=True).pack(anchor='w',pady=(0,12))
        return frame

    def button(self,parent,label,command,primary=False):
        b=ttk.Button(parent,text=label,command=command,style='Primary.TButton' if primary else 'TButton')
        b.pack(fill='x',pady=(0,7));self.buttons.append(b);return b

    def _build(self):
        header=tk.Frame(self.root,bg=NAVY,padx=25,pady=17);header.pack(fill='x')
        self.label(header,'Weld Vision',size=23,bold=True,color=WHITE,bg=NAVY).pack(side='left')
        self.label(header,'CIRCULAR WELD INSPECTION',size=10,color='#b8d1e4',bg=NAVY).pack(side='left',padx=24)
        self.label(header,'Final thesis methods',size=10,color='#c8dfed',bg=NAVY).pack(side='right')
        body=tk.Frame(self.root,bg=BG,padx=20,pady=18);body.pack(fill='both',expand=True)
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        side=tk.Frame(body,bg=BG,width=292);side.grid(row=0,column=0,sticky='ns',padx=(0,18));side.grid_propagate(False)
        # Pack inside a fixed-width grid column so long filenames cannot squeeze the viewer.
        side.configure(width=292);side.pack_propagate(False)
        source=self.card(side,'1  Choose an image')
        self.button(source,'Try example',self.example)
        self.button(source,'Open image...',self.open_file)
        self.label(source,variable=self.source_name,size=10,color=MUTED,wraplength=245,justify='left').pack(fill='x',pady=(5,4))
        self.button(source,'Inspect weld',self.inspect,primary=True)
        generated=self.card(side,'2  Or create a weld')
        self.label(generated,'Surface').pack(anchor='w')
        kind=ttk.Combobox(generated,textvariable=self.kind,values=list(KINDS),state='readonly');kind.pack(fill='x',pady=(3,9));self.settings.append(kind)
        self.label(generated,'Acquisition').pack(anchor='w')
        noise=ttk.Combobox(generated,textvariable=self.noise,values=list(COUNTS),state='readonly');noise.pack(fill='x',pady=(3,10));self.settings.append(noise)
        self.button(generated,'Create and inspect',lambda:self.inspect(True))
        self.advanced_button=ttk.Button(generated,text='Show advanced settings',command=self.toggle_advanced)
        self.advanced_button.pack(fill='x')
        self.advanced=tk.Frame(generated,bg=WHITE)
        for label,var in [('Random seed',self.seed),('Dwell time (us)',self.dwell)]:
            self.label(self.advanced,label,color=MUTED).pack(anchor='w',pady=(8,2))
            entry=ttk.Entry(self.advanced,textvariable=var);entry.pack(fill='x');self.settings.append(entry)
        self.label(side,'TIFF / PNG / JPEG / NPY / NPZ\nExternal images: guided preparation',color=MUTED,bg=BG,wraplength=280,justify='left').pack(anchor='w',padx=4,pady=(2,0))
        main=tk.Frame(body,bg=BG);main.grid(row=0,column=1,sticky='nsew');main.rowconfigure(1,weight=1);main.columnconfigure(0,weight=1)
        cards=tk.Frame(main,bg=BG);cards.grid(row=0,column=0,sticky='ew',pady=(0,14))
        for i,(title,var,col) in enumerate([('CNN candidates',self.cnn_count,BLUE),('Median candidates',self.median_count,'#b65e12'),('Processing time',self.runtime,TEXT)]):
            cards.columnconfigure(i,weight=1)
            box=tk.Frame(cards,bg=WHITE,highlightbackground=LINE,highlightthickness=1,padx=17,pady=12)
            box.grid(row=0,column=i,sticky='ew',padx=(0,10 if i<2 else 0))
            self.label(box,title,color=MUTED).pack(anchor='w');self.label(box,variable=var,size=23,bold=True,color=col).pack(anchor='w')
        viewer=tk.Frame(main,bg=WHITE,highlightbackground=LINE,highlightthickness=1);viewer.grid(row=1,column=0,sticky='nsew')
        toolbar=tk.Frame(viewer,bg=WHITE,padx=15,pady=10);toolbar.pack(fill='x')
        self.label(toolbar,'Inspection view',size=13,bold=True).pack(side='left')
        for name in ['Both','Median','CNN']:
            ttk.Radiobutton(toolbar,text=name,value=name,variable=self.view,command=self.redraw,style='View.TRadiobutton').pack(side='right')
        self.figure=Figure(figsize=(10,6),dpi=100,facecolor=WHITE)
        self.canvas=FigureCanvasTkAgg(self.figure,master=viewer);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.figure.text(.5,.55,'Your inspection appears here',ha='center',fontsize=22,color=NAVY,weight='bold')
        self.figure.text(.5,.45,'Try the example to compare CNN and Median detections.',ha='center',fontsize=12,color=MUTED)
        self.canvas.draw()
        bottom=tk.Frame(main,bg=BG,pady=10);bottom.grid(row=2,column=0,sticky='ew');bottom.columnconfigure(0,weight=1)
        self.label(bottom,variable=self.status,bg=BG,color=MUTED,wraplength=610,justify='left').grid(row=0,column=0,sticky='w')
        self.results_button=ttk.Button(bottom,text='Open saved results',command=self.open_results,state='disabled');self.results_button.grid(row=0,column=1,padx=(10,0))
        self.progress=ttk.Progressbar(main,mode='indeterminate');self.progress.grid(row=3,column=0,sticky='ew')

    def toggle_advanced(self):
        if self.advanced.winfo_manager():self.advanced.pack_forget();self.advanced_button.configure(text='Show advanced settings')
        else:self.advanced.pack(fill='x',pady=(3,0));self.advanced_button.configure(text='Hide advanced settings')

    def open_file(self):
        if self.busy:return
        path=filedialog.askopenfilename(parent=self.root,title='Open a TOPO image',filetypes=[('Supported images','*.npy *.npz *.tif *.tiff *.png *.jpg *.jpeg'),('TIFF','*.tif *.tiff'),('PNG','*.png'),('JPEG','*.jpg *.jpeg'),('Raw TOPO arrays','*.npy *.npz')])
        if not path:return
        self.select_file(path)

    def select_file(self,path):
        from .image_import import FORMATS
        chosen=Path(path)
        if chosen.suffix.lower() in FORMATS:
            from .import_dialog import ImportDialog
            try:dialog=ImportDialog(self.root,chosen)
            except Exception as exc:messagebox.showerror('Open image',str(exc),parent=self.root);return
            if dialog.result is None:return
            self.source=dialog.result[0];self.source_name.set(chosen.name+' (prepared)')
            self.status.set('External image prepared. Click Inspect weld for exploratory detections.')
        else:
            self.source=chosen;self.source_name.set(chosen.name)
            self.status.set('Image selected. Click Inspect weld to run both methods.')
        self.result=None;self.output=None;self.cnn_count.set('--');self.median_count.set('--');self.runtime.set('Ready');self.results_button.configure(state='disabled')
        self.figure.clear();self.figure.text(.5,.55,'Image ready to inspect',ha='center',fontsize=22,color=NAVY)
        self.figure.text(.5,.45,chosen.name,ha='center',fontsize=11,color=MUTED);self.canvas.draw_idle()

    def example(self):
        if not self.busy:self.source=engine.example_path();self.source_name.set('Example: low-noise pore');self.inspect()

    def inspect(self,generate=False):
        if self.busy:return
        try:
            settings=(int(self.seed.get()),KINDS[self.kind.get()],COUNTS[self.noise.get()],float(self.dwell.get())) if generate else None
            if generate and not (0<=settings[0]<=3_000_000_000 and .1<=settings[3]<=20):raise ValueError()
        except ValueError:
            messagebox.showerror('Check generation settings','Use a seed from 0 to 3000000000 and dwell time from 0.1 to 20 us.',parent=self.root);return
        source=self.source;self.busy=True
        for b in self.buttons:b.configure(state='disabled')
        for w in self.settings:w.configure(state='disabled')
        self.progress.start(12);self.runtime.set('Working');self.status.set('Creating the weld, then running both methods...' if generate else 'Inspecting the complete weld with CNN and Median...')
        def work():
            try:
                if self.inspector is None:self.inspector=engine.FinalInspector()
                path=engine.generate(*settings) if generate else source
                result=self.inspector.infer(path);folder=engine.export_result(result)
                self.events.put(('done',(result,folder)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            event,payload=self.events.get_nowait();self.busy=False;self.progress.stop()
            for b in self.buttons:b.configure(state='normal')
            for w in self.settings:w.configure(state='readonly' if isinstance(w,ttk.Combobox) else 'normal')
            if event=='error':
                self.runtime.set('Check input');self.status.set('Inspection could not finish. Check the input and try again.')
                messagebox.showerror('Inspection could not finish',payload,parent=self.root)
            else:
                self.result,self.output=payload;self.source=Path(self.result['source']);self.source_name.set(self.source.name)
                self.cnn_count.set(str(len(self.result['boxes']['cnn'])));self.median_count.set(str(len(self.result['boxes']['median'])))
                self.runtime.set(f'{self.result["seconds"]:.1f} s');self.results_button.configure(state='normal')
                if 'import_metadata' in self.result:
                    self.source_name.set(Path(self.result['import_metadata']['source']).name)
                    self.status.set('Exploratory external-image results saved. Candidate regions are not validated defects.')
                else:self.status.set('Inspection saved. Coloured boxes mark candidate regions; reference labels are not used.')
                self.redraw()
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def redraw(self):
        if self.result is None:return
        methods=('cnn','median') if self.view.get()=='Both' else (self.view.get().lower(),)
        result={**self.result,'boxes':{m:self.result['boxes'][m] for m in methods}}
        engine.draw_result(self.figure,result);self.canvas.draw_idle()

    def open_results(self):
        if self.output:os.startfile(self.output)

    def input_help(self):
        messagebox.showinfo('Image requirements','Supported: TIFF, PNG, JPEG, NPY and NPZ. PDF is excluded.\n\nRaster files open a preparation preview: place the ring centre, set its radius and choose intensity interpretation. The trained methods assume an 80 mm diameter weld. TIFF frames and native grayscale bit depth are supported.\n\nRaw arrays must be 160 x 5024, including the radial halo, at 0.05 mm radial sampling.\n\nDisplay-image imports use an explicit approximate intensity mapping. External-image detections are exploratory.',parent=self.root)

    def about(self):
        messagebox.showinfo('Final thesis methods','CNN: mixed-noise training on 96 backgrounds, seed 1027001, epoch 20.\nMedian: selected Gaussian filtering, threshold and geometric correction.\n\nBoth use frozen settings from the thesis evaluation. This is a synthetic B=0 surface-response study. Measured-image validation remains future work.',parent=self.root)

    def research(self):
        if self.busy:self.status.set('Wait for this inspection to finish before opening research tools.');return
        subprocess.Popen([sys.executable,str(engine.ROOT/'launch_weld_studio.py'),'--research'],cwd=engine.ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

    def close(self):
        if self.busy:
            messagebox.showinfo('Inspection in progress','Please wait for the current inspection to finish before closing.',parent=self.root);return
        self.root.destroy()

    def run(self):self.root.mainloop()


def create_app():return SimpleStudio()
