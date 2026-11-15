"""Preview and explicitly register a raster image before final-method inference."""
import tkinter as tk
from tkinter import ttk,messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import numpy as np
from .image_import import ImportSettings,load_raster,prepare,import_image
from . import final_methods as engine

class ImportDialog:
    def __init__(self,parent,path):
        self.result=None;self.path=path;self.values,self.alpha,self.info=load_raster(path)
        self.norm=engine.study.read(engine.study.OLD/'normalization.json')
        self.win=tk.Toplevel(parent);self.win.title('Prepare image | Weld Vision');self.win.geometry('1100x770');self.win.minsize(1000,730)
        self.win.configure(bg='#eef3f8');self.win.transient(parent);self.win.grab_set()
        self.variables={}
        h,w=self.values.shape
        defaults=dict(layout='Circular weld',centre_x=(w-1)/2,centre_y=(h-1)/2,radius_px=round(min(w,h)*.415,3),
                      start_angle_deg=0,intensity='Display image (exploratory)',gain=1,offset=0,frame=1)
        if (h,w)==(160,5024):defaults['layout']='Already unwrapped'
        for k,v in defaults.items():self.variables[k]=tk.StringVar(value=str(v))
        self.clockwise=tk.BooleanVar(value=False);self.reverse=tk.BooleanVar(value=False)
        self.confirmed=tk.BooleanVar(value=False)
        self.status=tk.StringVar(value='Adjust the dashed rings to enclose the weld. Click the image to place the centre.')
        body=tk.Frame(self.win,bg='#eef3f8',padx=15,pady=15);body.pack(fill='both',expand=True)
        left=tk.Frame(body,bg='white',padx=12,pady=10);left.pack(side='left',fill='y',padx=(0,15))
        ttk.Label(left,text='Prepare your TOPO image',font=('Segoe UI',13,'bold')).pack(anchor='w',pady=(0,12))
        fields=tk.Frame(left,bg='white');fields.pack(fill='x')
        for row,(key,label,choices) in enumerate([('layout','Image layout',['Circular weld','Already unwrapped']),
            ('frame',f'Frame (1 to {self.info["frames"]})',None),('centre_x','Centre X (pixels)',None),
            ('centre_y','Centre Y (pixels)',None),('radius_px','Weld centreline radius (pixels)',None),
            ('start_angle_deg','Start angle (degrees)',None),
            ('intensity','Intensity interpretation',['Display image (exploratory)','Raw TOPO units']),
            ('gain','Raw gain',None),('offset','Raw offset',None)]):
            ttk.Label(fields,text=label,wraplength=125).grid(row=row,column=0,sticky='w',padx=(0,8),pady=7)
            widget=ttk.Combobox(fields,textvariable=self.variables[key],values=choices,state='readonly',width=25) if choices else ttk.Entry(fields,textvariable=self.variables[key],width=27)
            widget.grid(row=row,column=1,sticky='ew',pady=7)
        ttk.Checkbutton(left,text='Clockwise around the ring',variable=self.clockwise).pack(anchor='w',pady=(7,1))
        ttk.Checkbutton(left,text='Reverse radial rows',variable=self.reverse).pack(anchor='w')
        ttk.Button(left,text='Update preview',command=self.preview).pack(fill='x',pady=10)
        right=tk.Frame(body,bg='white');right.pack(side='left',fill='both',expand=True)
        self.figure=Figure(figsize=(8,6),dpi=90);self.canvas=FigureCanvasTkAgg(self.figure,master=right);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.canvas.mpl_connect('button_press_event',self.clicked)
        ttk.Label(right,textvariable=self.status,wraplength=650).pack(fill='x',padx=10,pady=8)
        bottom=tk.Frame(self.win,bg='#eef3f8',padx=15,pady=10);bottom.pack(side='bottom',fill='x',before=body)
        ttk.Checkbutton(bottom,text='This is TOPO contrast for an 80 mm diameter weld, with the full radial image region available.',variable=self.confirmed).pack(anchor='w')
        ttk.Label(bottom,text='External images are exploratory. Display scaling cannot recover original detector intensities.',foreground='#965a13').pack(anchor='w',pady=5)
        ttk.Button(bottom,text='Use this image',command=self.accept).pack(side='right',padx=5)
        ttk.Button(bottom,text='Cancel',command=self.win.destroy).pack(side='right',padx=5)
        self.preview();parent.wait_window(self.win)

    def settings(self):
        v=self.variables
        return ImportSettings(layout='circular' if v['layout'].get()=='Circular weld' else 'unwrapped',
            centre_x=float(v['centre_x'].get()),centre_y=float(v['centre_y'].get()),radius_px=float(v['radius_px'].get()),
            start_angle_deg=float(v['start_angle_deg'].get()),clockwise=self.clockwise.get(),reverse_rows=self.reverse.get(),
            intensity='display' if v['intensity'].get().startswith('Display') else 'raw',gain=float(v['gain'].get()),
            offset=float(v['offset'].get()),frame=int(v['frame'].get())-1)

    def preview(self):
        self.figure.clear()
        try:
            settings=self.settings();self.values,self.alpha,self.info=load_raster(self.path,settings.frame)
            self.original=self.figure.add_subplot(211)
            lo,hi=np.percentile(self.values,[1,99]);self.original.imshow(self.values,cmap='gray',vmin=lo,vmax=hi)
            self.original.set_title('Original image: centreline and extraction limits',fontsize=11)
            if settings.layout=='circular':
                for ratio,color,style in [(1,'#00bce8','-'),((40-3.975)/40,'#e78a32','--'),((40+3.975)/40,'#e78a32','--')]:
                    self.original.add_patch(Circle((settings.centre_x,settings.centre_y),settings.radius_px*ratio,fill=False,ec=color,ls=style,lw=1.4))
                self.original.plot(settings.centre_x,settings.centre_y,'+',color='#00bce8')
            raw,_=prepare(self.values,self.alpha,settings,self.norm)
            ax=self.figure.add_subplot(212);ax.imshow(raw,origin='lower',aspect='auto',cmap='gray')
            ax.set_title('Prepared annulus: 160 x 5024',fontsize=11);ax.set_xlabel('Around weld');ax.set_ylabel('Inner to outer')
            self.status.set(f'{self.info["width"]} x {self.info["height"]} | {self.info["mode"]} | Preview ready. Check the ring alignment before importing.')
        except Exception as exc:self.status.set(str(exc))
        self.figure.tight_layout();self.canvas.draw()

    def clicked(self,event):
        if event.inaxes==getattr(self,'original',None) and event.xdata is not None and self.variables['layout'].get()=='Circular weld':
            self.variables['centre_x'].set(f'{event.xdata:.2f}');self.variables['centre_y'].set(f'{event.ydata:.2f}');self.preview()

    def accept(self):
        if not self.confirmed.get():
            messagebox.showinfo('Confirm the image meaning','Confirm that the image is TOPO contrast for an 80 mm weld. A photograph or height map is a different input.',parent=self.win);return
        try:self.result=import_image(self.path,self.settings(),self.norm,engine.OUTPUT)
        except Exception as exc:messagebox.showerror('Image preparation',str(exc),parent=self.win);return
        self.win.destroy()
