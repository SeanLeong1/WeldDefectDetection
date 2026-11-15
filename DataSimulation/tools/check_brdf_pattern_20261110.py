"""Reproduce CASINO angular holdouts and show full-pattern adequacy.

Read-only access to the historical campaign; no CASINO jobs or renderer edits.
"""
from pathlib import Path
import sys, json, csv, hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'DataSimulation'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bse_renderer.validation import casino_compact as cc
from bse_renderer.validation.casino_emission_fit import _parse_run_id

SOURCE=ROOT/'DataSimulation/data/output/casino_surface_reanalysis_20260914'
OUT=ROOT/'docs/brdf_pattern_check_20261110'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def model_mass(b,sin_alpha):
    """Exact integrals of p(mu,az)=mu/pi*(1+B*sin(alpha)*sqrt(1-mu²)*cos(az))."""
    mu=np.linspace(0,1,13);az=np.linspace(-np.pi,np.pi,25)
    base=.5*np.diff(mu**2)[:,None]*np.diff(az)[None,:]/np.pi
    radial=((1-mu[:-1]**2)**1.5-(1-mu[1:]**2)**1.5)/3
    asym=b*sin_alpha*radial[:,None]*np.diff(np.sin(az))[None,:]/np.pi
    result=base+asym
    assert np.min(result)>=0 and abs(result.sum()-1)<1e-12
    assert np.allclose(result.sum(1),np.diff(mu**2),atol=1e-14)
    return result

def reproduce():
    manifest=read(SOURCE/'source_manifest.json');provenance=read(SOURCE/'analysis_provenance.json')
    assert sha(SOURCE/'source_manifest.json')==provenance['source_manifest_sha256']
    current_analysis_sha=sha(cc.__file__)
    parts=[];hashes=set();files={}
    for row in manifest:
        digest=sha(row['source']);assert digest==row['source_sha256'];assert digest not in hashes
        hashes.add(digest);files[row['source']]=digest
        with np.load(row['compact'],allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata']))
            assert metadata['sha256']==digest
        tilt,repeat=_parse_run_id(row['run_id'])
        a,_=cc.select_surface_exits(Path(row['compact']),tilt,repeat,1.)
        parts.append(a)
    samples=np.concatenate(parts)
    with np.load(SOURCE/'selected_surface_exits.npz') as z:stored=z['samples']
    assert samples.shape==stored.shape and np.allclose(samples,stored,rtol=0,atol=1e-10)
    result=cc.analyze_samples(samples);old=read(SOURCE/'surface_b_report.json')
    assert result['selected_exit_count']==old['selected_exit_count']
    assert abs(result['pooled_constant']['b']-old['pooled_constant']['b'])<1e-8
    deltas=[]
    for key,fields in [('repeat_holdouts',('lambertian_nll','scalar_b_nll','angular_histogram_nll')),('model_holdouts',('lambertian_nll','constant_nll','sine_nll'))]:
        assert len(result[key])==len(old[key])
        for a,b in zip(result[key],old[key]):
            for field in fields:deltas.append(abs(a[field]-b[field]))
    assert max(deltas)<1e-8
    write(OUT/'reproduced_angular_report.json',result)
    write(OUT/'reproduction_audit.json',dict(status='passed',source_files=len(files),selected_exits=len(samples),
        source_hashes=files,max_abs_sample_difference=float(np.max(abs(samples-stored))),
        max_abs_heldout_nll_difference=max(deltas),analysis_source_sha256=current_analysis_sha,
        original_analysis_source_sha256=provenance['script_sha256'],
        analysis_source_matches_original=current_analysis_sha==provenance['script_sha256'],
        audit_script_sha256=sha(__file__),new_simulations=0,renderer_changed=False,
        scope='Reproduction of existing development holdouts; not new independent simulation confirmation.'))
    print('Reproduced',len(samples),'exits; maximum held-out score difference',max(deltas),flush=True)
    return samples,result

def assess(samples,report):
    records=[];maps={}
    repeat_models={r['heldout_value']:r for r in report['model_holdouts'] if r['heldout_unit']=='repeat'}
    for tilt in sorted(np.unique(samples[:,4])):
        subset=samples[samples[:,4]==tilt]
        empirical=np.histogram2d(subset[:,1],subset[:,2],bins=(12,24),range=((0,1),(-np.pi,np.pi)))[0]/len(subset)
        constant=np.zeros((12,24));scalar=constant.copy();sine=constant.copy()
        for repeat in np.unique(subset[:,3]):
            train=subset[subset[:,3]!=repeat];test=subset[subset[:,3]==repeat]
            assert len(train) and len(test)
            weight=len(test)/len(subset);sin_alpha=float(np.sin(test[:,0]).mean())
            b=cc.fit_scalar_b(train)['b'] or 0
            constant+=weight*model_mass(repeat_models[repeat]['constant_b'],sin_alpha)
            scalar+=weight*model_mass(b,sin_alpha)
            law=repeat_models[repeat]['sine_b0']+repeat_models[repeat]['sine_b1']*np.sin(test[:,0])
            sine+=weight*model_mass(1.,float(np.mean(law*np.sin(test[:,0]))))
        values=np.sort(subset[:,1]);n=len(values);cdf=values**2
        ks=max(float(np.max(np.arange(1,n+1)/n-cdf)),float(np.max(cdf-np.arange(n)/n)))
        per=next(r for r in report['per_tilt'] if r['nominal_tilt_deg']==tilt)
        records.append(dict(tilt_deg=float(tilt),exits=n,screening=per['screening_decision'],
            observed_mean_cos2=float(np.mean(subset[:,1]**2)),model_mean_cos2=.5,
            polar_cdf_max_difference=ks,
            binned_tv_constant=float(abs(empirical-constant).sum()/2),
            binned_tv_per_tilt=float(abs(empirical-scalar).sum()/2),
            binned_tv_sine=float(abs(empirical-sine).sum()/2)))
        maps[tilt]=(empirical,constant,scalar)
    write(OUT/'pattern_discrepancies.json',dict(rows=records,
        interpretation='Descriptive binned and marginal discrepancies; no new pass threshold or independence assumption. Binned distances include finite-sample noise.',
        prediction='Each model is refitted without the evaluated repeat. Pooled candidates exclude that repeat across every tilt. Map averages weight each held-out repeat by its exit count.',
        normalization='Conditional probability per outgoing solid angle; 12x24 equal-solid-angle cells, dOmega=dmu*dazimuth.'))
    with (OUT/'pattern_discrepancies.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return maps,records

def plots(samples,report,maps,records):
    chosen=(20.,50.,70.)
    fig,axes=plt.subplots(3,4,figsize=(14,10),layout='constrained')
    vmax=max(float(a.max()*100) for t in chosen for a in maps[t])
    limit=max(float(abs(maps[t][0]-maps[t][2]).max()*100) for t in chosen)
    for j,t in enumerate(chosen):
        observed,constant,scalar=maps[t]
        for k,a in enumerate((observed,constant,scalar,observed-scalar)):
            ax=axes[j,k]
            image=ax.imshow(a*100,origin='lower',extent=(-180,180,0,1),aspect='auto',
                           cmap='coolwarm' if k==3 else 'viridis',vmin=-limit if k==3 else 0,vmax=limit if k==3 else vmax)
            ax.set_xlabel('Exit azimuth (degrees)');ax.set_xticks([-180,-90,0,90,180])
            ax.set_ylabel(f'Incidence {t:g}°\ncos(exit polar angle)' if k==0 else 'cos(exit polar angle)')
            fig.colorbar(image,ax=ax,shrink=.75,label='Difference (percentage points)' if k==3 else 'Outgoing electrons per cell (%)')
        for ax,title in zip(axes[0],('CASINO exits','Held-out pooled constant B','Held-out B fitted per tilt','CASINO minus per-tilt B')):ax.set_title(title,fontsize=11)
    fig.suptitle('Does the fitted angular model reproduce the full CASINO pattern?\nCross-validated predictions on existing 304SS, 60 keV runs',fontsize=15)
    fig.savefig(OUT/'01_angular_patterns.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    phi=np.linspace(0,90,500);density=2*np.cos(np.radians(phi))*np.sin(np.radians(phi))*np.pi/180
    for ax,t in zip(axes.ravel(),(0.,30.,60.,70.)):
        subset=samples[samples[:,4]==t]
        ax.hist(np.degrees(np.arccos(subset[:,1])),bins=np.linspace(0,90,19),density=True,alpha=.5,color='#287e9c',label='CASINO exits')
        ax.plot(phi,density,color='#b52c41',lw=2,label='Every B law: same polar marginal')
        value=float(np.mean(subset[:,1]**2))
        ax.set_title(f'Incidence {t:g}°\nMean cos²: CASINO {value:.3f}, model 0.500',fontsize=11)
        ax.set(xlabel='Exit polar angle from surface normal (degrees)',ylabel='Probability density per degree')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Changing B cannot repair the model’s polar-angle distribution',fontsize=15)
    fig.savefig(OUT/'02_polar_pattern_limit.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for key,name,color in [('lambertian_nll','B = 0','#888888'),('scalar_b_nll','One fitted B per tilt','#c56a32'),('angular_histogram_nll','Training-only angular histogram','#24709d')]:
        vals=[np.mean([r[key] for r in report['repeat_holdouts'] if r['nominal_tilt_deg']==t]) for t in sorted(maps)]
        ax.plot(sorted(maps),vals,'-o',label=name,color=color)
    ax.set(xlabel='Incidence angle (degrees)',ylabel='Held-out negative log likelihood (lower is better)',title='Existing repeat-held-out checks: richer angular shape matters at high tilt')
    ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'03_heldout_scores.png',dpi=180);plt.close(fig)

def summary(report,records):
    b=report['pooled_constant']['b']
    lines=['# Does the BRDF-style model reproduce CASINO?','',
        'The existing checks support a conditional approximation, with substantial high-tilt limitations. They do not establish a fully qualified angular response for the current weld renderer.','',
        '## What was checked again?','',
        'Verified the SHA-256 hashes of all 48 original CASINO summary exports, reconstructed 466,055 selected outward same-surface exits from their compact boundary caches, and reproduced the fitted coefficients and every stored repeat/tilt held-out likelihood score. The original artifacts and renderer were preserved.','',
        'Each fold fits on other simulation repeats and evaluates the omitted repeat. Whole-tilt holdouts also test interpolation, with endpoint holdouts explicitly treated as extrapolation. These files have already been examined during development; this recheck supplies no new independent simulation runs.','',
        f'The pooled constant is B = {b:.6f}. Its cross-validation uses a separate training-only estimate in each fold, rather than testing the final all-data constant on its own fitting data.','',
        '## Does the full angular pattern match?','',
        '![Two-dimensional comparison](01_angular_patterns.png)','',
        'The maps use equal-solid-angle bins and normalized probability masses. Their units are the percentage of selected outgoing electrons in each cell. The per-tilt model is a more flexible comparator than a single pooled B. Its remaining high-tilt residuals show limitations of the angular formula itself.','',
        '![Polar distribution](02_polar_pattern_limit.png)','',
        'For this model, integrating over exit azimuth gives p(mu) = 2 mu, where mu = cos(exit polar angle). Consequently every choice of B requires E[mu²] = 0.5. CASINO gives approximately 0.534 at normal incidence, 0.517 at 30°, 0.432 at 60°, and 0.370 at 70°. Adjusting B changes azimuthal asymmetry but cannot remove this polar-shape discrepancy.','',
        '## What did the held-out scores establish?','',
        '| Held-out grouping | B = 0 NLL | Pooled constant NLL | Sine-law NLL |','|---|---:|---:|---:|']
    for unit in ('repeat','tilt'):
        folds=[r for r in report['model_holdouts'] if r['heldout_unit']==unit]
        lines.append('| '+unit+' | '+' | '.join(f'{np.mean([r[k] for r in folds]):.6f}' for k in ('lambertian_nll','constant_nll','sine_nll'))+' |')
    lines+=['','Lower negative log likelihood (NLL) means that the model assigns more probability to the CASINO directions it did not fit. Improvement over B = 0 is a relative comparison; full-pattern agreement requires additional adequacy checks.','',
        '![Held-out scores](03_heldout_scores.png)','',
        'The original scalar-B screening passes at 10–30°, is provisional at 5° and 40–60°, and rejects a single full-hemisphere scalar description at 70°. At normal incidence B has no effect, so its unidentifiability is separate from checking whether the B = 0 polar shape fits. The moderate-angle screening checks consistency of B across polar bands; it does not certify the entire distribution.','',
        'At 70°, the training-only angular histogram achieves held-out NLL 1.473561 versus 1.651355 for the per-tilt scalar B. This indicates recoverable structure beyond the scalar formula. The final all-run histogram is an analysis reference; it must not serve as independent test evidence on its source runs.','',
        '## What remains to establish renderer validity?','',
        'A separate later development campaign investigated a richer eight-coefficient angular model. Its saved report states that the predefined advancement condition did not pass, detector-acceptance qualification remained incomplete, and no production qualification was issued. That campaign uses a different simulation population and must be kept separate from the 466,055-exit analysis above.','',
        'The next decisive test is a fixed-model comparison on new CASINO runs covering the renderer’s actual incidence, material and energy domain. It should report full polar/azimuth residuals and the collected fraction integrated over the actual detector apertures. The current raised-crown images use the explicitly selected B = 0 option; the corrected B fits were not loaded into that image campaign. Agreement with CASINO would establish agreement with that transport simulation under its assumptions. Measured-image validation remains a separate requirement.','',
        '## Reproducible evidence','',
        '- `reproduction_audit.json`: original source hashes and replay differences.',
        '- `reproduced_angular_report.json`: all fitted and held-out numerical results.',
        '- `pattern_discrepancies.csv`: per-tilt marginal and binned discrepancies. These are descriptive; no new acceptance thresholds were selected.',
        '',
        'Reproduce from the project root: `.venv/Scripts/python.exe DataSimulation/tools/check_brdf_pattern_20261110.py`.','']
    text='\n'.join(lines);(OUT/'README.md').write_text(text,encoding='utf8')
    # Reuse the standalone local-report Markdown renderer without executing its study.
    sys.path.insert(0,str(ROOT/'WeldDefectDetection'))
    from report_median_localisation_20261110 import local_markdown
    html='<!doctype html><meta charset="utf-8"><title>CASINO angular-pattern check</title><style>body{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#172534}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}h1,h2{color:#123c67}code{font-size:.9em}</style>'+local_markdown(text)
    (OUT/'OPEN_REPORT.html').write_text(html,encoding='utf8')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    samples,report=reproduce();maps,records=assess(samples,report)
    plots(samples,report,maps,records);summary(report,records)
    print('Report:',OUT/'OPEN_REPORT.html',flush=True)
