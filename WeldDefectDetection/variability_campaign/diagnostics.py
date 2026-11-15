"""Numerical gradient checks, paired render pilot, and constant-B evidence."""
from dataclasses import replace
import time
import numpy as np
from scipy.stats import t as student_t
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.dataset_forms import _disable_noise
from bse_renderer.renderer.scan_engine import ScanEngine
from .renderer import gradients
from .campaign import REPO, FRAME, SPACING, KINDS, read, write, sha, seal, verify_sources, engine, sampled, pore_parameters


def analytic_gradients():
    rows=[]
    for size in (33,65,129):
        x=np.linspace(-1.6,1.6,size);y=x.copy();xx,yy=np.meshgrid(x,y);dx=x[1]-x[0]
        fields={
            'plane':(.2*xx-.1*yy, np.full_like(xx,.2),np.full_like(yy,-.1)),
            'quadratic':(.05*xx**2+.08*yy**2+.03*xx*yy,.1*xx+.03*yy,.16*yy+.03*xx),
            'smooth_wave':(.02*np.sin(3*xx)*np.cos(2*yy),.06*np.cos(3*xx)*np.cos(2*yy),-.04*np.sin(3*xx)*np.sin(2*yy))}
        for name,(h,gx,gy) in fields.items():
            for method in ('finite_difference','cubic_spline'):
                t=time.perf_counter();py,px=gradients(h,dx,method);elapsed=time.perf_counter()-t
                error=(px-gx)**2+(py-gy)**2
                rows.append(dict(surface=name,size=size,spacing_mm=dx,method=method,
                                 gradient_rmse=float(np.sqrt(error.mean())),interior_rmse=float(np.sqrt(error[2:-2,2:-2].mean())),
                                 maximum_error=float(np.sqrt(error.max())),seconds=elapsed))
    return rows


def constant_b_evidence(root):
    folder=REPO/'DataSimulation/data/output/casino_interpolation_review_20260915_v2'
    report=read(folder/'report.json');candidates=read(folder/'frozen_candidates.json')
    samples=np.load(folder/'selected_surface_exits.npz')['samples']
    b=candidates['model_comparison']['constant_b']['fitted_b']
    sine=candidates['model_comparison']['b_alpha_sin']
    rows=[]
    for old in report['run_scores']:
        s=samples[(samples[:,3]==old['repeat']) & np.isclose(samples[:,4],old['tilt_deg'])]
        if len(s)!=old['exits']:raise ValueError('CASINO run membership mismatch')
        alpha,mu,az=s[:,:3].T
        asym=np.sin(alpha)*np.sqrt(np.maximum(0,1-mu**2))*np.cos(az)
        nll={}
        for name,coef in [('lambertian',0.),('constant',b),('sine',sine['b0']+sine['b1']*np.sin(alpha))]:
            nll[name]=float(-np.mean(np.log(mu/np.pi)+np.log1p(coef*asym)))
            if not np.isclose(nll[name],old['nll'][name],atol=1e-10,rtol=0):
                raise ValueError(f'CASINO saved score did not reproduce: {old["run_id"]} {name}')
        rows.append(dict(run_id=old['run_id'],tilt_deg=old['tilt_deg'],repeat=old['repeat'],exits=len(s),nll=nll))
    comparisons={}
    rng=np.random.default_rng(926980001)
    for name,a,baseline in [('constant_over_zero','constant','lambertian'),('sine_over_constant','sine','constant')]:
        gain=np.array([r['nll'][baseline]-r['nll'][a] for r in rows])
        half=float(student_t.ppf(.975,len(gain)-1)*gain.std(ddof=1)/np.sqrt(len(gain)))
        tilts=sorted({r['tilt_deg'] for r in rows})
        tiltmeans=np.array([np.mean([v for r,v in zip(rows,gain) if r['tilt_deg']==t]) for t in tilts])
        draws=tiltmeans[rng.integers(len(tilts),size=(10000,len(tilts)))].mean(1)
        comparisons[name]=dict(mean_gain_nats_per_exit=float(gain.mean()),run_t_interval=[float(gain.mean()-half),float(gain.mean()+half)],
                               tilt_cluster_bootstrap_interval=np.quantile(draws,[.025,.975]).tolist(),tilts=len(tilts),runs=len(rows))
    polar=read(REPO/'DataSimulation/data/output/casino_selection_review_20260915/polar_marginal_check.json')
    result=dict(B=b,refitted=False,recomputed_exit_likelihoods=True,selected_exits=len(samples),comparisons=comparisons,
                per_run=rows,polar_60deg=next(r for r in polar['rows'] if r['tilt_deg']==60.),
                source_hashes={str(folder/name):sha(folder/name) for name in ('report.json','frozen_candidates.json','selected_surface_exits.npz')},
                conclusion='Constant B is supported as a simpler predictive reference versus the frozen sine law on these previously examined intermediate tilts. This is not proof of equivalence or absolute physical adequacy.',
                limits=['Existing historical held-out evaluation reanalysis, not new independent CASINO confirmation.',
                        'Constant and sine share the polar-shape limitation; 60-degree selection evidence contradicts E[mu^2]=0.5.',
                        'Run intervals assume independence; explicit CASINO RNG control remains unverified.',
                        'Tilt bootstrap is descriptive over only ten observed tilt clusters, not a universal material-law interval.'])
    write(root/'constant_b_evidence.json',result)
    return result


def pilot(root):
    p=verify_sources(root);folder=root/'pilot';folder.mkdir(exist_ok=True)
    cfg=load_config_bundle(REPO/'DataSimulation/config');rows=[];files=[]
    analytic=analytic_gradients();write(folder/'analytic_gradients.json',analytic)
    # Physical-spacing and polynomial derivative recovery are independent checks.
    if any(r['gradient_rmse']>1e-10 for r in analytic if r['surface']=='plane'):
        raise ValueError('Plane gradient recovery failed')
    if any(r['gradient_rmse']>1e-10 for r in analytic if r['surface']=='quadratic' and r['method']=='cubic_spline'):
        raise ValueError('Spline quadratic derivative recovery failed')
    for i in range(p['pilot']['backgrounds']):
        sample,target=sampled(cfg,p['pilot']['seed']+i,i)
        engines={m:engine(m) for m in ('finite_difference','cubic_spline')}
        for e in engines.values():e.set_variation(.1)
        for kind in KINDS:
            scene=replace(sample,defects=pore_parameters(kind,np.random.default_rng(sample.seed+310003)))
            results={};values={}
            for method,e in engines.items():
                t=time.monotonic();result=e.run(scene);elapsed=time.monotonic()-t
                sl=slice(e.halo,e.halo+FRAME)
                results[method]=result
                values[method]=dict(quiet=e.quiet_topo[sl].copy(),raw=result.channels.topo[sl].copy(),
                     mask=result.defect_mask[sl].copy(),normals=e.last_normals[sl].copy(),seconds=elapsed)
            a,b=values['finite_difference'],values['cubic_spline']
            if not np.array_equal(a['mask'],b['mask']):raise ValueError('Gradient choice changed labels')
            angle=np.degrees(np.arccos(np.clip(np.sum(a['normals']*b['normals'],axis=-1),-1,1)))
            rows.append(dict(background=i,kind=kind,labels_identical=True,normal_angle_mean_deg=float(angle.mean()),
                normal_angle_p99_deg=float(np.quantile(angle,.99)),topo_rms_difference=float(np.sqrt(np.mean((a['quiet']-b['quiet'])**2))),
                finite_difference_noise_rms=float(np.sqrt(np.mean((a['raw']-a['quiet'])**2))),
                finite_difference_seconds=a['seconds'],spline_seconds=b['seconds']))
            path=folder/f'paired_{i}_{kind}.npz'
            np.savez_compressed(path,finite_difference=a['quiet'],spline=b['quiet'],raw=a['raw'],mask=a['mask'])
            files.append(path)
            print(f'Pilot background={i} kind={kind}: spline comparison complete',flush=True)
    # Verify capture-before-noise against an actual noise-disabled replay.
    e=engines['finite_difference'];saved=e.quiet_topo.copy()
    quiet_result=e.run(_disable_noise(scene))
    quiet_error=float(np.max(np.abs(saved-quiet_result.channels.topo)))
    if quiet_error>1e-6:raise ValueError('Captured quiet channel differs from explicit noiseless replay')
    # Zero-amplitude local strip must reproduce the original full-frame renderer
    # on noiseless data. This checks halo, axes, capture and baseline handling.
    sample,_=sampled(cfg,p['pilot']['seed'],0);sample=_disable_noise(sample)
    e=engine();small=e.run(sample)
    full=ScanEngine(image_size=e.image_size,pixel_size_mm=e.pixel_size_mm,render_model_mode='formula_only',
        emission_b_mode=e.emission_b_mode,emission_b_path=e.emission_b_path,render_view_mode='top_surface',weld_layout=e.weld_layout).run(sample)
    y0=(e.image_size-FRAME)//2
    strip_error=float(np.max(np.abs(full.channels.topo[y0:y0+FRAME]-small.channels.topo[e.halo:e.halo+FRAME])))
    if strip_error>1e-6:raise ValueError('Zero-variation strip does not reproduce historical renderer')
    write(root/'pilot_measurements.json',dict(status='ok',analytic_gradient_rows=analytic,paired_render_rows=rows,
          quiet_capture_max_error=quiet_error,zero_variation_strip_max_error=strip_error,
          interpretation='Analytic surfaces have known derivatives; procedural-surface comparisons measure sensitivity, not physical accuracy.'))
    constant_b_evidence(root)
    seal(root,'pilot',[root/'pilot_measurements.json',root/'constant_b_evidence.json',folder/'analytic_gradients.json',*files])
