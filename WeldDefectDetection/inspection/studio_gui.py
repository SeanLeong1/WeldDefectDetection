"""Two keyboard-accessible workspaces for generation and inspection."""
from pathlib import Path
from dataclasses import replace
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .simple_gui import SimpleStudio, BG, WHITE, NAVY, TEXT, MUTED, BLUE, KINDS, COUNTS
from . import final_methods as engine
from . import studio_tasks as tasks


class WeldStudio(SimpleStudio):
    def __init__(self):
        self.cancelled=threading.Event();self.generated=[];self.batch_rows=[];self.batch_folder=None
        self.batch_template=None;self.mask_folder=None;self.batch_output=None
        super().__init__()
        self.root.title('Weld Vision | Create and inspect')
        self.root.minsize(1100,760)
        self.root.bind('<Alt-Key-1>',lambda e:self.tabs.select(self.create_tab))
        self.root.bind('<Alt-Key-2>',lambda e:self.tabs.select(self.inspect_tab))
        self.status.set('Open an image or choose a folder. The final CNN and Median settings are fixed.')

    def _build(self):
        style=ttk.Style(self.root)
        style.configure('TLabel',background=WHITE,foreground=TEXT)
        style.configure('TCheckbutton',background=WHITE,foreground=TEXT,padding=4)
        style.configure('TNotebook',background=BG,borderwidth=0)
        style.configure('TNotebook.Tab',padding=(25,12),font=('Segoe UI',12,'bold'))
        style.map('TNotebook.Tab',background=[('selected',WHITE)],foreground=[('selected',BLUE)])
        style.configure('Treeview',font=('Segoe UI',10),rowheight=30)
        style.configure('Treeview.Heading',font=('Segoe UI',10,'bold'))
        header=tk.Frame(self.root,bg=NAVY,padx=22,pady=12);header.pack(fill='x')
        self.label(header,'Weld Vision',size=21,bold=True,color=WHITE,bg=NAVY).pack(side='left')
        self.label(header,'Create surfaces  →  simulate images  →  inspect',color=WHITE,bg=NAVY).pack(side='left',padx=25)
        self.label(header,'Alt+1 Create   Alt+2 Inspect',color='#d6e4ef',bg=NAVY).pack(side='right')
        self.tabs=ttk.Notebook(self.root);self.tabs.pack(fill='both',expand=True,padx=12,pady=12)
        self.create_tab=tk.Frame(self.tabs,bg=BG);self.inspect_tab=tk.Frame(self.tabs,bg=BG)
        self.tabs.add(self.create_tab,text='1  Create welds');self.tabs.add(self.inspect_tab,text='2  Inspect images')
        self.tabs.enable_traversal()
        self._create_workspace();self._inspect_workspace()

    def sidebar(self,parent,width=340):
        outer=tk.Frame(parent,bg=WHITE,width=width);outer.pack(side='left',fill='y',padx=(0,12));outer.pack_propagate(False)
        canvas=tk.Canvas(outer,bg=WHITE,highlightthickness=0,width=width-20)
        scroll=ttk.Scrollbar(outer,orient='vertical',command=canvas.yview)
        scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        frame=tk.Frame(canvas,bg=WHITE,padx=14,pady=12);item=canvas.create_window((0,0),window=frame,anchor='nw')
        frame.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(item,width=e.width))
        # Focused controls scroll into view during keyboard navigation.
        def reveal(event):
            widget=event.widget
            if str(widget).startswith(str(frame)):
                self.root.update_idletasks()
                y=widget.winfo_rooty()-frame.winfo_rooty()
                total=max(frame.winfo_height(),1);top=canvas.canvasy(0)
                if y<top or y+widget.winfo_height()>top+canvas.winfo_height():
                    canvas.yview_moveto(max(0,(y-30)/total))
        frame.bind_all('<FocusIn>',reveal,add='+')
        def wheel(event):
            if str(event.widget).startswith(str(outer)):canvas.yview_scroll(int(-event.delta/120),'units')
        frame.bind_all('<MouseWheel>',wheel,add='+')
        return frame

    def heading(self,parent,text):
        self.label(parent,text,size=12,bold=True).pack(anchor='w',pady=(12,6))

    def _create_workspace(self):
        body=tk.Frame(self.create_tab,bg=BG,padx=10,pady=10);body.pack(fill='both',expand=True)
        side=self.sidebar(body)
        self.label(side,'Set up a weld',size=15,bold=True).pack(anchor='w')
        self.label(side,'80 mm circular weld · 0.05 mm sampling\nSurface-opening pores · B = 0 response',wraplength=290,justify='left',color=MUTED).pack(fill='x',pady=8)
        self.genvars={};defaults=tasks.asdict(tasks.GenerationSettings())
        groups=[('Weld surface',[('crown_height','Crown height (mm)'),('crown_width','Crown width (mm)'),('variation','Profile variation (0–0.1)')]),
                ('Pore geometry',[('diameter','Diameter (mm)'),('depth','Depth parameter (mm)'),('angle','Angle (degrees)'),('offset','Radial offset (mm)')]),
                ('Acquisition and batch',[('dwell','Dwell time (µs)'),('seed','Starting seed'),('number','Number of welds')])]
        self.heading(side,'Pore type')
        combo=ttk.Combobox(side,textvariable=self.kind,values=list(KINDS),state='readonly');combo.pack(fill='x');self.settings.append(combo)
        for title,fields in groups:
            self.heading(side,title)
            grid=tk.Frame(side,bg=WHITE);grid.pack(fill='x');grid.columnconfigure(1,weight=1)
            for row,(key,label) in enumerate(fields):
                var=tk.StringVar(value=str(defaults[key]));self.genvars[key]=var
                self.label(grid,label).grid(row=row,column=0,sticky='w',padx=(0,6),pady=5)
                entry=ttk.Entry(grid,textvariable=var,width=10);entry.grid(row=row,column=1,sticky='ew',pady=5);self.settings.append(entry)
        self.label(side,'Acquisition level').pack(anchor='w',pady=(10,3))
        combo=ttk.Combobox(side,textvariable=self.noise,values=list(COUNTS),state='readonly');combo.pack(fill='x');self.settings.append(combo)
        self.random_pores=tk.BooleanVar(value=False)
        check=ttk.Checkbutton(side,text='Vary pore size and position per weld',variable=self.random_pores);check.pack(anchor='w',pady=8);self.settings.append(check)
        self.label(side,'Each weld uses the next seed. When variation is enabled, pore size, depth and position use the study ranges instead of the manual pore values.',wraplength=285,justify='left',color=MUTED).pack(fill='x')
        self.label(side,'Custom crown dimensions are exploratory inputs for the pretrained CV methods.',wraplength=285,justify='left',color=MUTED).pack(fill='x',pady=8)
        main=tk.Frame(body,bg=BG);main.pack(side='left',fill='both',expand=True)
        bar=tk.Frame(main,bg=WHITE,padx=12,pady=10);bar.pack(fill='x')
        self.gen_button=ttk.Button(bar,text='Generate welds',command=self.generate,style='Primary.TButton');self.gen_button.pack(side='left');self.buttons.append(self.gen_button)
        self.send_button=ttk.Button(bar,text='Inspect generated images →',command=self.send_generated,state='disabled');self.send_button.pack(side='left',padx=10)
        ttk.Button(bar,text='Open output folder',command=self.open_generation).pack(side='right')
        load=ttk.Button(bar,text='Load saved welds…',command=self.load_generated);load.pack(side='right',padx=6);self.buttons.append(load)
        controls=tk.Frame(main,bg=WHITE,padx=12,pady=8);controls.pack(fill='x',pady=(10,0))
        self.label(controls,'Stage').pack(side='left')
        self.stage=tk.StringVar(value='3D surface')
        stages=['3D surface','Weld profile','Pores and truth','Detector channels','Directional signals','TOPO']
        ttk.Combobox(controls,textvariable=self.stage,values=stages,state='readonly',width=23).pack(side='left',padx=8)
        self.stage.trace_add('write',lambda *a:self.draw_generation())
        self.scene=tk.StringVar();self.scene_picker=ttk.Combobox(controls,textvariable=self.scene,values=[],state='readonly',width=16)
        self.scene_picker.pack(side='right');self.scene.trace_add('write',lambda *a:self.draw_generation())
        self.label(controls,'Weld').pack(side='right',padx=8)
        self.genfigure=Figure(figsize=(9,6),dpi=100,facecolor=WHITE)
        self.gencanvas=FigureCanvasTkAgg(self.genfigure,master=main);self.gencanvas.get_tk_widget().pack(fill='both',expand=True)
        self.genfigure.text(.5,.55,'Explore each stage of image generation',ha='center',fontsize=18,color=NAVY)
        self.genfigure.text(.5,.44,'Surface → pores and mask → detector signals → TOPO',ha='center',fontsize=11,color=MUTED);self.gencanvas.draw()
        self.genstatus=tk.StringVar(value='Set parameters, then select Generate welds. Scroll the settings panel for batch controls.')
        status_label=self.label(main,variable=self.genstatus,bg=BG,wraplength=700,justify='left')
        bottom=tk.Frame(main,bg=BG);bottom.pack(side='bottom',fill='x',before=self.gencanvas.get_tk_widget())
        status_label.pack(side='bottom',fill='x',pady=8,before=self.gencanvas.get_tk_widget())
        self.genprogress=ttk.Progressbar(bottom,mode='determinate');self.genprogress.pack(side='left',fill='x',expand=True)
        ttk.Button(bottom,text='Stop after current weld',command=self.cancelled.set).pack(side='right',padx=8)

    def _inspect_workspace(self):
        body=tk.Frame(self.inspect_tab,bg=BG,padx=10,pady=10);body.pack(fill='both',expand=True)
        side=self.sidebar(body)
        self.heading(side,'Choose inputs')
        self.button(side,'Open image…',self.open_file)
        self.button(side,'Choose folder…',self.choose_folder)
        self.button(side,'Try saved example',self.example)
        self.label(side,variable=self.source_name,color=MUTED,wraplength=285,justify='left').pack(fill='x',pady=10)
        self.recursive=tk.BooleanVar(value=True);ttk.Checkbutton(side,text='Include subfolders',variable=self.recursive,command=self.refresh_files).pack(anchor='w')
        self.heading(side,'Folder image preparation')
        self.button(side,'Prepare raster batch…',self.prepare_batch)
        self.same_registration=tk.BooleanVar(value=False)
        ttk.Checkbutton(side,text='All raster images share this alignment',variable=self.same_registration).pack(anchor='w')
        self.label(side,'PNG / JPEG / TIFF need a checked ring alignment. Batch settings apply to matching image dimensions.',wraplength=285,justify='left',color=MUTED).pack(fill='x',pady=5)
        self.heading(side,'Optional reference masks')
        self.use_masks=tk.BooleanVar(value=False)
        ttk.Checkbutton(side,text='Calculate accuracy using masks',variable=self.use_masks).pack(anchor='w')
        self.button(side,'Choose mask folder…',self.choose_masks)
        self.masklabel=tk.StringVar(value='Default: masks beside each input')
        self.label(side,variable=self.masklabel,wraplength=285,justify='left',color=MUTED).pack(fill='x')
        self.label(side,'Use <image>_mask.npy in prepared annulus coordinates (128 × 5024 or 160 × 5024). Missing or invalid masks are reported.',wraplength=285,justify='left',color=MUTED).pack(fill='x',pady=8)
        self.label(side,'CNN and Median use the final fixed settings. Counts are candidate regions. Accuracy requires matching reference masks.',wraplength=285,justify='left',color=MUTED).pack(fill='x',pady=10)
        main=tk.Frame(body,bg=BG);main.pack(side='left',fill='both',expand=True)
        actions=tk.Frame(main,bg=WHITE,padx=10,pady=8);actions.pack(fill='x')
        run=ttk.Button(actions,text='Run inspection',command=self.inspect,style='Primary.TButton');run.pack(side='left');self.buttons.append(run)
        ttk.Button(actions,text='Stop after current image',command=self.cancelled.set).pack(side='left',padx=10)
        self.summary=tk.StringVar(value='No inspection run yet')
        self.label(main,variable=self.summary,bg=WHITE,size=11,bold=True,wraplength=850,justify='left',padx=12,pady=10).pack(fill='x')
        self.results_tabs=ttk.Notebook(main);self.results_tabs.pack(fill='both',expand=True,pady=10)
        viewer=tk.Frame(self.results_tabs,bg=WHITE);table=tk.Frame(self.results_tabs,bg=WHITE)
        self.results_tabs.add(viewer,text='Image preview');self.results_tabs.add(table,text='Folder statistics')
        toolbar=tk.Frame(viewer,bg=WHITE,padx=10);toolbar.pack(fill='x')
        for name in ['CNN','Median','Both']:
            ttk.Radiobutton(toolbar,text=name,value=name,variable=self.view,command=self.redraw,style='View.TRadiobutton').pack(side='left')
        self.figure=Figure(figsize=(10,6),dpi=100,facecolor=WHITE)
        self.canvas=FigureCanvasTkAgg(self.figure,master=viewer);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.figure.text(.5,.5,'Open an image or a folder to begin',ha='center',fontsize=18,color=NAVY);self.canvas.draw()
        self.metrics_text=tk.StringVar(value='Accuracy will appear here when compatible reference masks are available.')
        self.label(table,variable=self.metrics_text,wraplength=850,justify='left',padx=12,pady=10).pack(fill='x')
        columns=('image','status','cnn','median','seconds','labels')
        self.tree=ttk.Treeview(table,columns=columns,show='headings',selectmode='browse')
        for key,title,width in zip(columns,['Image','Status','CNN','Median','Time (s)','Reference / error'],[210,80,65,65,75,240]):
            self.tree.heading(key,text=title);self.tree.column(key,width=width,minwidth=55,stretch=key in ('image','labels'))
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y');self.tree.configure(yscrollcommand=scroll.set)
        horizontal=ttk.Scrollbar(table,orient='horizontal',command=self.tree.xview);horizontal.pack(side='bottom',fill='x');self.tree.configure(xscrollcommand=horizontal.set)
        self.tree.pack(fill='both',expand=True);self.tree.bind('<<TreeviewSelect>>',self.show_selected)
        self.label(table,'Select a row to see its saved result in Image preview.',color=MUTED,padx=10,pady=5).pack(side='bottom',fill='x',before=self.tree)
        bottom=tk.Frame(main,bg=BG);bottom.pack(side='bottom',fill='x',before=self.results_tabs)
        self.label(bottom,variable=self.status,bg=BG,wraplength=650,justify='left').pack(side='left',fill='x',expand=True)
        self.results_button=ttk.Button(bottom,text='Open saved results',command=self.open_results,state='disabled');self.results_button.pack(side='right')
        self.progress=ttk.Progressbar(main,mode='determinate');self.progress.pack(side='bottom',fill='x',pady=(8,0),before=bottom)

    def set_busy(self,value):
        self.busy=value
        for b in self.buttons:b.configure(state='disabled' if value else 'normal')
        for w in self.settings:w.configure(state='disabled' if value else ('readonly' if isinstance(w,ttk.Combobox) else 'normal'))
        self.send_button.configure(state='normal' if self.generated and not value else 'disabled')

    def generate(self):
        if self.busy:return
        try:
            values={k:(int(v.get()) if k in ('seed','number') else float(v.get())) for k,v in self.genvars.items()}
            settings=tasks.GenerationSettings(**values,kind=KINDS[self.kind.get()],counts=COUNTS[self.noise.get()],random_pores=self.random_pores.get());settings.validate()
        except ValueError as exc:messagebox.showerror('Check settings',str(exc),parent=self.root);return
        self.generated=[];self.generation_folder=tasks.new_folder('generated_batches');self.cancelled.clear();self.set_busy(True)
        self.genprogress.configure(maximum=settings.number,value=0);self.genstatus.set('Generating surface, visibility and detector signals. The first weld may take a few minutes.')
        def work():
            try:
                for i in range(settings.number):
                    if self.cancelled.is_set():break
                    path=tasks.generate_scene(replace(settings,seed=settings.seed+i,number=1),self.generation_folder/f'weld_{i+1:03d}')
                    self.events.put(('generated',path))
                self.events.put(('generation_done',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def draw_generation(self):
        if not self.generated:return
        try:
            index=int(self.scene.get().split()[-1])-1
            tasks.draw_stage(self.genfigure,self.generated[index],self.stage.get());self.gencanvas.draw_idle()
        except (ValueError,IndexError):pass

    def open_generation(self):
        if getattr(self,'generation_folder',None):os.startfile(self.generation_folder)

    def load_generated(self,folder=None):
        if self.busy:return
        folder=folder or filedialog.askdirectory(parent=self.root,title='Choose a saved generation folder')
        if not folder:return
        paths=[p for p in tasks.discover(folder) if p.name=='topo.npy' and (p.parent/'stages.npz').exists()]
        if not paths:messagebox.showinfo('No saved stages','Choose a folder generated in Create welds.',parent=self.root);return
        self.generation_folder=Path(folder);self.generated=paths
        names=[f'Weld {i+1}' for i in range(len(paths))];self.scene_picker.configure(values=names);self.scene.set(names[0])
        self.send_button.configure(state='normal');self.tabs.select(self.create_tab)
        self.genstatus.set(f'{len(paths)} saved weld(s) loaded. Settings at left apply to the next generation; saved parameters are in scene.json.')

    def send_generated(self):
        if self.busy or not self.generated:return
        self.clear_inspection()
        self.batch_folder=self.generation_folder;self.batch_paths=list(self.generated);self.batch_template=None
        self.use_masks.set(True);self.mask_folder=None;self.masklabel.set('Generated geometric masks beside each input')
        self.source_name.set(f'{len(self.batch_paths)} generated welds');self.tabs.select(self.inspect_tab)
        self.status.set('Generated images are ready. Run inspection to compare detections with their masks.')

    def select_file(self,path):
        if self.busy:return
        self.tabs.select(self.inspect_tab)
        super().select_file(path)
        self.batch_folder=None;self.batch_paths=[];self.batch_template=None
        self.summary.set('Single-image inspection');self.results_tabs.select(0)
        self.batch_rows=[];self.tree.delete(*self.tree.get_children())
        self.metrics_text.set('Run inspection to calculate results for this image.')

    def clear_inspection(self):
        self.batch_rows=[];self.tree.delete(*self.tree.get_children());self.result=None;self.output=None
        self.results_button.configure(state='disabled');self.summary.set('Inputs selected. Ready to inspect.')
        self.metrics_text.set('Run inspection to calculate results for these inputs.')
        self.figure.clear();self.figure.text(.5,.5,'Run inspection to view these inputs',ha='center',fontsize=18,color=NAVY);self.canvas.draw_idle()

    def example(self):
        if self.busy:return
        self.select_file(engine.example_path());self.inspect()

    def choose_folder(self):
        if self.busy:return
        path=filedialog.askdirectory(parent=self.root,title='Choose image folder')
        if path:
            self.clear_inspection()
            self.batch_folder=Path(path);self.batch_template=None;self.same_registration.set(False);self.refresh_files()

    def refresh_files(self):
        if self.busy or self.batch_folder is None:return
        self.batch_paths=tasks.discover(self.batch_folder,self.recursive.get())
        self.source_name.set(f'{self.batch_folder.name}\n{len(self.batch_paths)} supported input files')
        self.status.set('Prepare raster alignment if needed, then select Run inspection.')

    def prepare_batch(self):
        if self.busy:return
        rasters=[p for p in getattr(self,'batch_paths',[]) if p.suffix.lower() in tasks.FORMATS]
        if not rasters:messagebox.showinfo('Raster batch','Choose a folder containing TIFF, PNG or JPEG images first.',parent=self.root);return
        from .import_dialog import ImportDialog
        try:
            dialog=ImportDialog(self.root,rasters[0])
            if dialog.result:self.batch_template=dialog.result[1];self.status.set('Alignment prepared. Confirm the other raster images share this alignment.')
        except Exception as exc:messagebox.showerror('Prepare batch',str(exc),parent=self.root)

    def choose_masks(self):
        if self.busy:return
        folder=filedialog.askdirectory(parent=self.root,title='Choose masks with matching relative paths and <image>_mask.npy names')
        if folder:self.mask_folder=Path(folder);self.masklabel.set(str(self.mask_folder));self.use_masks.set(True)

    def inspect(self,generate=False):
        if self.busy:return
        paths=list(self.batch_paths) if self.batch_folder else [self.source]
        if not paths:messagebox.showinfo('No images','Choose an image folder with supported files.',parent=self.root);return
        if any(p.suffix.lower() in tasks.FORMATS for p in paths) and (not self.batch_template or not self.same_registration.get()):
            messagebox.showinfo('Prepare raster images','Use Prepare raster batch, then confirm the shared alignment. Images with different alignment need individual preparation.',parent=self.root);return
        root=self.batch_folder;use_masks=self.use_masks.get();mask_root=self.mask_folder;template=self.batch_template
        self.cancelled.clear();self.set_busy(True);self.batch_rows=[];self.tree.delete(*self.tree.get_children())
        self.batch_output=tasks.new_folder('batch_results');self.output=self.batch_output
        self.results_button.configure(state='disabled');self.progress.configure(maximum=len(paths),value=0)
        self.summary.set(f'Inspecting {len(paths)} image(s)…');self.status.set('Running the final CNN and Median. Stop takes effect after the current image.')
        if len(paths)>1:self.results_tabs.select(1)
        def work():
            try:
                if self.inspector is None:self.inspector=engine.FinalInspector()
                summary,rows=tasks.inspect_folder(paths,root,self.inspector,self.batch_output,
                    lambda i,n,row:self.events.put(('batch_row',(i,n,row))),self.cancelled.is_set,
                    raster_template=template,use_masks=use_masks,mask_root=mask_root)
                self.events.put(('batch_done',summary))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def show_summary(self,summary):
        c=summary['methods']['cnn'];m=summary['methods']['median']
        self.summary.set(f'{summary["completed"]}/{summary.get("requested",summary["total"])} images completed · {summary["errors"]} errors · {summary["seconds"]:.1f} s inference\n'
            f'CNN candidates: {c["candidates"]} (images flagged: {c["flagged_images"]})   |   Median candidates: {m["candidates"]} (images flagged: {m["flagged_images"]})')
        lines=[]
        def fmt(value):return 'N/A' if value is None else f'{value:.3f}'
        for name,stats in [('CNN',c),('Median',m)]:
            if stats['labelled_images']:
                lines.append(f'{name}, {stats["labelled_images"]} labelled images: detected {stats["tp"]}, false positives {stats["fp"]}, missed {stats["fn"]}\n'
                    f'Precision {fmt(stats["precision"])}   Recall {fmt(stats["recall"])}   F1 {fmt(stats["f1"])}')
            else:lines.append(f'{name}: no compatible reference masks. Detection counts only.')
        lines.append('Accuracy uses only labelled images, with one-to-one box IoU ≥ 0.5 matching.')
        self.metrics_text.set('\n'.join(lines))

    def show_selected(self,event=None):
        selected=self.tree.selection()
        if not selected:return
        row=self.batch_rows[int(selected[0])]
        if row['status']!='ok':self.status.set(row['error']);return
        import json
        import numpy as np
        folder=Path(row['output']);record=json.loads((folder/'prediction.json').read_text())
        with np.load(folder/'predictions.npz') as z:
            record.update(raw=engine.load_raw(row['prepared']),scores={m:z[m+'_score'].copy() for m in ('cnn','median')})
        self.result=record;self.redraw()
        self.status.set(f'{Path(row["source"]).name}: CNN {row["cnn_candidates"]}, Median {row["median_candidates"]} candidate regions. Select Image preview to view.')

    def poll(self):
        try:
            while True:
                event,payload=self.events.get_nowait()
                if event=='generated':
                    self.generated.append(payload);names=[f'Weld {i+1}' for i in range(len(self.generated))]
                    self.scene_picker.configure(values=names);self.scene.set(names[-1]);self.genprogress.configure(value=len(names))
                    self.genstatus.set(f'{len(names)} weld(s) saved. Explore stages while the batch continues.')
                elif event=='generation_done':
                    self.set_busy(False);self.genstatus.set(f'{len(self.generated)} weld(s) saved'+(' · Stopped early.' if self.cancelled.is_set() else '. Ready to inspect.'))
                elif event=='batch_row':
                    i,n,row=payload;self.batch_rows.append(row);self.progress.configure(value=i)
                    note=row.get('error') or row.get('label_note') or ('Scored' if row.get('evaluation') else 'Not requested')
                    path=Path(row['source'])
                    name=str(path.relative_to(self.batch_folder)) if self.batch_folder and path.is_relative_to(self.batch_folder) else path.name
                    self.tree.insert('', 'end',iid=str(len(self.batch_rows)-1),values=(name,row['status'],row.get('cnn_candidates',''),row.get('median_candidates',''),f'{row.get("seconds",0):.1f}',note))
                    current=tasks.summarise(self.batch_rows);current['requested']=n;self.show_summary(current)
                elif event=='batch_done':
                    self.set_busy(False);self.show_summary(payload);self.results_button.configure(state='normal')
                    self.status.set('Stopped. Completed results saved.' if payload['cancelled'] else 'Inspection complete. Summary JSON, image CSV and individual results saved.')
                    good=next((str(i) for i,r in enumerate(self.batch_rows) if r['status']=='ok'),None)
                    if good is not None:self.tree.selection_set(good)
                elif event=='error':
                    self.set_busy(False);self.status.set('Could not finish: '+payload);self.genstatus.set('Could not finish: '+payload)
                    messagebox.showerror('Could not finish',payload,parent=self.root)
        except queue.Empty:pass
        self.root.after(100,self.poll)


def create_app():return WeldStudio()
