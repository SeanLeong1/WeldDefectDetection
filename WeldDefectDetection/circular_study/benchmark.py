"""Warm CPU inference timing after the research run, excluding file loading."""
import platform,time
import cv2,numpy as np,torch
from . import generation as g,training,median,evaluation


def run():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    frozen=evaluation.verify_freeze();full=g.load('test');data={k:v[:1] for k,v in full.items()}
    model=training.load_model(frozen['cnn'])
    calls=dict(cnn=lambda:training.predict(data,model,frozen['cnn']['config']),
               median=lambda:median.scores(data,frozen['median']['spec']))
    timings={}
    for name,call in calls.items():
        call();values=[]
        for _ in range(5):
            start=time.perf_counter();scores=call();values.append(time.perf_counter()-start)
            assert scores.shape==(1,64,5024) and np.isfinite(scores).all()
        timings[name]=dict(seconds=values,median_seconds=float(np.median(values)),minimum_seconds=min(values))
    g.write(g.OUT/'runtime_benchmark.json',dict(scope='Warm full-ring score inference, one fixed clean test ring, five repeats; excludes file loading, model loading and final region extraction. Run after other compute jobs finish.',
        torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),platform=platform.platform(),processor=platform.processor(),
        torch_version=torch.__version__,opencv_version=cv2.__version__,timings=timings,
        fit_timing_note='Training report seconds measure the current fit invocation. Interrupted/resumed runs are not cumulative total training times.'))
    print(timings,flush=True)


if __name__=='__main__':run()
