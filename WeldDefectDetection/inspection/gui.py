"""Desktop inspection panel, usable alone or inside the renderer application."""
from __future__ import annotations

import os
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .engine import DEFAULT_CAMPAIGN, REPO, Options, run
from .registry import campaigns,available_methods,example_path,CURRENT

MODES = {'Display image (exploratory)': 'display', 'Raw TOPO (training units)': 'raw',
         'Already normalized [0,1]': 'normalized'}
METHODS = {'CNN': ('cnn',), 'Neighbouring median': ('median',),
           'Compare CNN and median': ('cnn', 'median'),
           'Threshold baseline': ('thresholding',), 'Compare all methods': ('cnn', 'median', 'thresholding')}
V2_CAMPAIGN = REPO/'DataSimulation/data/output/acquisition_campaign_20260928_v2'


def apply_theme(root):
    root.configure(bg='#edf2f7')
    root.option_add('*Font', ('Segoe UI', 10))
    root.option_add('*Background', '#edf2f7')
    root.option_add('*Foreground', '#203249')
    root.option_add('*Entry.Background', '#ffffff')
    root.option_add('*Entry.Relief', 'flat')
    root.option_add('*Entry.BorderWidth', 5)
    root.option_add('*Button.Relief', 'flat')
    root.option_add('*Button.Padding', 5)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background='#edf2f7', foreground='#203249')
    style.configure('TButton', padding=(12, 7))
    style.configure('Accent.TButton', background='#087f8c', foreground='white')
    style.map('Accent.TButton', background=[('active', '#096775'), ('disabled', '#a8b9bf')])
    style.configure('TNotebook', borderwidth=0)
    style.configure('TNotebook.Tab', padding=(18, 9))
    style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#087f8c')])
    style.configure('TLabelframe', padding=12)
    style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
    style.configure('Heading.TLabel', font=('Segoe UI', 21, 'bold'))
    style.configure('Subheading.TLabel', font=('Segoe UI', 12, 'bold'))
    style.configure('Muted.TLabel', foreground='#52677e')
    style.configure('Treeview', rowheight=29, fieldbackground='white', background='white')
    style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), padding=6)


class InspectionPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=18)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.result = None
        self.photo = None
        self.source = tk.StringVar()
        self.output = tk.StringVar(value=str(REPO/'data/inspection_runs'))
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.method = tk.StringVar(value='CNN')
        self.campaigns = campaigns()
        self.campaign = tk.StringVar(value=list(self.campaigns)[-1])
        self.spacing = tk.StringVar(value='')
        self.recursive = tk.BooleanVar(value=False)
        self.colour = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value='Choose an image or folder. Files stay on this computer.')
        self._build()
        self._campaign_changed()
        self.after(120, self._poll)

    def _build(self):
        ttk.Label(self, text='Inspect TOPO images', style='Heading.TLabel').pack(anchor='w')
        ttk.Label(self, text='Import • check • detect • review', style='Muted.TLabel').pack(anchor='w', pady=(2, 14))
        inputs = ttk.LabelFrame(self, text='1   Input and method')
        inputs.pack(fill='x')
        inputs.columnconfigure(1, weight=1)
        ttk.Label(inputs, text='Image / folder').grid(row=0, column=0, sticky='w')
        ttk.Entry(inputs, textvariable=self.source).grid(row=0, column=1, sticky='ew', padx=10)
        ttk.Button(inputs, text='Add image', command=self.choose_image).grid(row=0, column=2, padx=3)
        ttk.Button(inputs, text='Add folder', command=self.choose_folder).grid(row=0, column=3, padx=3)
        ttk.Label(inputs, text='Interpretation').grid(row=1, column=0, sticky='w', pady=8)
        ttk.Combobox(inputs, textvariable=self.mode, values=list(MODES), state='readonly', width=32).grid(row=1, column=1, sticky='w', padx=10)
        self.method_box = ttk.Combobox(inputs, textvariable=self.method, values=list(METHODS), state='readonly', width=27)
        self.method_box.grid(row=1, column=2, columnspan=2, sticky='ew')
        ttk.Label(inputs, text='Pixel size (mm)').grid(row=2, column=0, sticky='w')
        pixelrow = ttk.Frame(inputs)
        pixelrow.grid(row=2, column=1, sticky='w', padx=10)
        ttk.Entry(pixelrow, textvariable=self.spacing, width=12).pack(side='left')
        ttk.Label(pixelrow, text='  Blank = unknown; raw model requires 0.05', style='Muted.TLabel').pack(side='left')
        ttk.Checkbutton(inputs, text='Include subfolders', variable=self.recursive).grid(row=2, column=2, sticky='w')
        ttk.Checkbutton(inputs, text='Convert colour to gray', variable=self.colour).grid(row=2, column=3, sticky='w')
        ttk.Label(inputs, text='Save results to').grid(row=3, column=0, sticky='w', pady=(10, 0))
        ttk.Entry(inputs, textvariable=self.output).grid(row=3, column=1, columnspan=2, sticky='ew', padx=10, pady=(10, 0))
        ttk.Button(inputs, text='Browse', command=self.choose_output).grid(row=3, column=3, sticky='ew', pady=(10, 0))
        ttk.Label(inputs, text='Frozen model').grid(row=4, column=0, sticky='w', pady=(8, 0))
        self.campaign_box = ttk.Combobox(inputs, textvariable=self.campaign, values=list(self.campaigns), state='readonly', width=36)
        self.campaign_box.grid(row=4, column=1, columnspan=3, sticky='w', padx=10, pady=(8, 0))
        self.campaign_box.bind('<<ComboboxSelected>>', self._campaign_changed)
        self.contract_text=tk.StringVar()
        ttk.Label(self, textvariable=self.contract_text,
                  style='Muted.TLabel', wraplength=1000).pack(anchor='w', pady=10)
        bar = ttk.Frame(self)
        bar.pack(fill='x', pady=(0, 10))
        self.check_button = ttk.Button(bar, text='Check files', command=lambda: self.start(True))
        self.check_button.pack(side='left')
        self.run_button = ttk.Button(bar, text='Run detection', style='Accent.TButton', command=self.start)
        self.run_button.pack(side='left', padx=8)
        self.cancel_button = ttk.Button(bar, text='Cancel', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left')
        ttk.Button(bar, text='Load example', command=self.example).pack(side='right')
        ttk.Button(bar, text='Open results', command=self.open_results).pack(side='right', padx=8)
        split = ttk.Panedwindow(self, orient='horizontal')
        split.pack(fill='both', expand=True)
        left = ttk.Frame(split)
        right = ttk.LabelFrame(split, text='Prediction preview')
        split.add(left, weight=3)
        split.add(right, weight=2)
        self.tree = ttk.Treeview(left, columns=('file', 'method', 'status', 'count'), show='headings', height=7)
        for key, label, width in [('file', 'File', 230), ('method', 'Method', 80), ('status', 'Status', 85), ('count', 'Candidates', 85)]:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=50)
        scroll = ttk.Scrollbar(left, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self.show_selected)
        self.preview = ttk.Label(right, text='Select a completed result to view its overlay.', anchor='center')
        self.preview.pack(fill='both', expand=True)
        self.detail = tk.Text(self, height=4, wrap='word', relief='flat', bg='white', padx=10, pady=8)
        self.detail.pack(fill='x', pady=(10, 5))
        self.detail.configure(state='disabled')
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill='x', pady=3)
        ttk.Label(self, textvariable=self.status, style='Muted.TLabel', wraplength=1000).pack(anchor='w')

    def choose_image(self):
        value = filedialog.askopenfilename(title='Select a TOPO image', filetypes=[('Images and arrays', '*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.npy *.npz'), ('All files', '*.*')])
        if value:
            self.source.set(value)
            if Path(value).suffix.lower() == '.npz':
                self.mode.set('Raw TOPO (training units)')

    def _campaign_changed(self, _event=None):
        campaign = self.campaigns[self.campaign.get()]
        available = available_methods(campaign)
        choices = [name for name, arms in METHODS.items() if all(a in available for a in arms)]
        self.method_box.configure(values=choices)
        if self.method.get() not in choices:
            self.method.set('CNN')
        if campaign==CURRENT:
            self.contract_text.set('Current circular study: complete 96 × 5024 annulus at 50 µm radial sampling; 128 rows also accepted.\nCNN and median use periodic whole-ring inference. NPZ/NPY only; no automatic ring localisation or physical validation.')
            self.mode.set('Raw TOPO (training units)')
        else:
            self.contract_text.set('Historical crop model: 64 × 64 patches at 50 µm/pixel. Display-image imports are exploratory.\nMedian requires four genuine context crops. This model does not reproduce the latest circular study.')

    def choose_folder(self):
        value = filedialog.askdirectory(title='Select an image folder')
        if value:
            self.source.set(value)

    def choose_output(self):
        value = filedialog.askdirectory(title='Choose a result location')
        if value:
            self.output.set(value)

    def example(self):
        campaign = self.campaigns[self.campaign.get()]
        self.source.set(str(example_path(campaign)))
        self.mode.set('Raw TOPO (training units)')
        self.spacing.set('0.05')
        self.method.set('Compare CNN and median' if campaign in (CURRENT,V2_CAMPAIGN) else 'Compare all methods')
        self.status.set('Synthetic reference example loaded. Run detection to compare methods.')

    def start(self, validate_only=False):
        if self.busy:
            return
        try:
            if not self.source.get().strip() or not self.output.get().strip():
                raise ValueError('Choose an input and output location.')
            opts = Options(MODES[self.mode.get()], float(self.spacing.get()) if self.spacing.get().strip() else None,
                           self.colour.get(), self.recursive.get(), METHODS[self.method.get()])
            opts.check()
            source, output = self.source.get(), self.output.get()
            campaign = self.campaigns[self.campaign.get()]
            available = available_methods(campaign)
            if any(a not in available for a in opts.arms):
                raise ValueError('Selected method is unavailable in this frozen model.')
        except Exception as exc:
            messagebox.showerror('Check inputs', str(exc), parent=self.winfo_toplevel())
            return
        self.busy = True
        self.cancel.clear()
        self.run_button.configure(state='disabled')
        self.check_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.tree.delete(*self.tree.get_children())
        self.result = None
        self.preview.configure(image='', text='Processing locally…')
        self.progress['value'] = 0
        self.status.set('Checking inputs and frozen model…')
        def worker():
            try:
                result = run(source, output, opts, campaign=campaign, validate_only=validate_only, cancel=self.cancel,
                             progress=lambda n, total, row: self.events.put(('progress', (n, total, row))))
                self.events.put(('done', result))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    n, total, row = value
                    self.progress['value'] = 100*n/total
                    self.status.set(f'{n}/{total} files • {Path(row["source"]).name} • {row.get("status", "checking")}')
                else:
                    self.busy = False
                    self.run_button.configure(state='normal')
                    self.check_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    if kind == 'error':
                        self.status.set(value)
                        self._details(value)
                        self.preview.configure(image='', text='Check the input message below.')
                    else:
                        self.result = value
                        for i, row in enumerate(value['records']):
                            self.tree.insert('', 'end', iid=str(i), values=(Path(row['source']).name, row['method'], row['status'], row.get('candidates', '')))
                        errors = sum(r['status'] == 'error' for r in value['records'])
                        self.status.set(f'{"Cancelled" if value["cancelled"] else "Finished"} • {len(value["records"])} results • {errors} errors • JSON / CSV saved')
                        if value['records']:
                            self.tree.selection_set('0')
                            self.show_selected()
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _details(self, text):
        self.detail.configure(state='normal')
        self.detail.delete('1.0', 'end')
        self.detail.insert('end', text)
        self.detail.configure(state='disabled')

    def show_selected(self, _event=None):
        selected = self.tree.selection()
        if not selected or not self.result:
            return
        row = self.result['records'][int(selected[0])]
        text = row['source']+'\n'+row.get('message', f'{row["status"]} • {row.get("candidates", 0)} candidates')
        text += '\n'+'\n'.join(row.get('warnings', []))
        self._details(text)
        if row.get('overlay'):
            with Image.open(row.get('preview',row['overlay'])) as image:
                image.thumbnail((max(150, self.preview.winfo_width()-16), max(100, self.preview.winfo_height()-16)))
                self.photo = ImageTk.PhotoImage(image.copy(), master=self)
            self.preview.configure(image=self.photo, text='')
        else:
            self.preview.configure(image='', text=row.get('message', 'Preflight complete; run detection for an overlay.'), wraplength=300)

    def open_results(self):
        if self.result:
            os.startfile(self.result['output'])
        else:
            self.status.set('Run file checks or detection first to create a result folder.')


def main():
    root = tk.Tk()
    root.title('Weld Vision | Image inspection')
    root.geometry('1180x850')
    root.minsize(980, 720)
    apply_theme(root)
    panel = InspectionPanel(root)
    panel.pack(fill='both', expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
