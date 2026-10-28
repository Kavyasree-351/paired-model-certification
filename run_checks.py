"""Run bounded correctness checks without launching research experiments."""
import argparse,json,time,unittest
from pathlib import Path
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');args=parser.parse_args()
    names=['test_correctness','selective_smoke.Smoke','staged_audit.AuditTests','revelation_audit.Checks']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names);start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.record:
        payload=dict(successful=result.wasSuccessful(),tests_run=result.testsRun,
            failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors],
            seconds=time.perf_counter()-start,
            scope='Bounded correctness/smoke checks only; no experiment outputs rewritten and no final data evaluated.')
        Path(__file__).with_name('PUBLICATION_TESTS.json').write_text(json.dumps(payload,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
