"""Run build/audit/upload in sequence; verify every staged object's SHA-256."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT = 'bold-bastion-509200-f9'
ROOT = Path('/mnt/vcc/split-release')
DEST = 'gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924'


def boundary():
    active = subprocess.check_output(['gcloud','config','configurations','list','--filter=is_active:true','--format=value(name)'],text=True).strip()
    project = subprocess.check_output(['gcloud','config','get-value','project'],text=True).strip()
    assert active == 'vcc-2026' and project == PROJECT


def upload(source, destination, recursive=False):
    boundary()
    subprocess.run(['gcloud','storage','cp',*(['--recursive'] if recursive else []),str(source),destination],check=True)


def verify(item):
    local, remote = item
    with local.open('rb') as f:
        expected = hashlib.file_digest(f,'sha256').hexdigest()
    proc = subprocess.Popen(['gcloud','storage','cat',remote],stdout=subprocess.PIPE)
    digest = hashlib.file_digest(proc.stdout,'sha256').hexdigest()
    assert proc.wait() == 0 and digest == expected, remote
    return {'uri':remote,'sha256':digest,'bytes':local.stat().st_size}


def main():
    boundary()
    status = {'release':'split-v1-20260924','project':PROJECT,'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running','accepted_contexts':[],'artifacts':{}}
    def checkpoint():
        path=ROOT/'status.json';path.write_text(json.dumps(status,indent=2)+'\n')
        upload(path,DEST+'/status.json')
    checkpoint()
    try:
        subprocess.run([sys.executable,'-m','pytest','tests/test_context_split_release.py','-q'],cwd=ROOT,check=True)
        for c in ['HepG2','Jurkat','K562_essential','RPE1','HCT116','HEK293T']:
            status['current_context']=c;checkpoint()
            for script in ['build_context_splits.py','audit_context_splits.py']:
                subprocess.run([sys.executable,str(ROOT/'scripts/cloud'/script),str(ROOT/'input'),str(ROOT/'work'),c],check=True)
            out=ROOT/'work/outputs'/c
            upload(out,DEST+'/',recursive=True)
            files=sorted(p for p in out.rglob('*') if p.is_file())
            with ThreadPoolExecutor(max_workers=4) as pool:
                hashes=list(pool.map(verify,[(p,DEST+'/'+c+'/'+str(p.relative_to(out))) for p in files]))
            record={'status':'passed','context':c,'files':hashes,'independent_acceptance':json.loads((out/'acceptance.json').read_text())}
            path=out/'transport-audit.json';path.write_text(json.dumps(record,indent=2)+'\n')
            upload(path,DEST+'/'+c+'/transport-audit.json')
            status['artifacts'][c]=verify((path,DEST+'/'+c+'/transport-audit.json'))
            status['accepted_contexts'].append(c);checkpoint()
        status['state']='completed_staging_validated'
    except BaseException as exc:
        status['state']='failed';status['error']=repr(exc)
        raise
    finally:
        status['updated_utc']=datetime.now(timezone.utc).isoformat();checkpoint()


if __name__ == '__main__':
    main()
