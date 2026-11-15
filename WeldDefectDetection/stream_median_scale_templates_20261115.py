"""Evaluate frozen median policies while fresh backgrounds finish generating."""
import concurrent.futures,time
import median_scale_templates_20261115 as st

def main():
    st.verify(True);(st.OUT/'confirmation_scenes').mkdir(exist_ok=True)
    pending=set(range(st.N_FRESH));start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(2) as pool:
        while pending:
            ready=[i for i in sorted(pending) if (st.OUT/f'datasets/test/{i:03d}/index.json').exists()]
            if not ready:
                if time.monotonic()-start>10800:raise TimeoutError('Fresh generation has not completed within three hours')
                time.sleep(3);continue
            for i in ready:
                list(pool.map(st.confirm_worker,st.read(st.OUT/f'datasets/test/{i:03d}/index.json')))
                pending.remove(i);print('FROZEN EVALUATION',i,'remaining',len(pending),flush=True)

if __name__=='__main__':main()
