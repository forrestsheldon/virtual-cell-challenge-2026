"""Download approved compact artifacts first, then verify the complete local core."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/context-transfer-blog/split-v1-20260924/completion'
DATA=ROOT/'data/derived/context_transfer_splits/split-v1-20260924'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    status=json.loads((REPORT/'status.json').read_text())
    progress={'state':'downloading','scope':'compact core; responders retained in cloud','completed_contexts':[]}
    def record():
        (REPORT/'local-download-status.json').write_text(json.dumps(progress,indent=2)+'\n')
    try:
        for c,entry in status['artifacts'].items():
            dest=DATA/c;dest.mkdir(exist_ok=True)
            p=REPORT/c/'transport-audit.json';assert sha(p)==entry['sha256']
            audit=json.loads(p.read_text())
            urls=[]
            for item in [*audit['files'],entry]:
                if '/responder_cells/' in item['uri']:
                    continue
                local=dest/item['uri'].rsplit('/',1)[1]
                if local.exists() and local.stat().st_size==item['bytes'] and sha(local)==item['sha256']:
                    continue
                urls.append(item['uri'])
            progress['current_context']=c;record()
            if urls:
                subprocess.run(['gcloud','storage','cp',*urls,str(dest)+'/'],check=True)
            progress['completed_contexts'].append(c);record()
        progress['state']='verifying';record()
        subprocess.run([sys.executable,str(ROOT/'scripts/analysis/verify_local_split_release.py'),'--core-only'],check=True)
        progress['state']='complete_verified';record()
    except BaseException as exc:
        progress['state']='failed';progress['error']=repr(exc);record()
        raise


if __name__=='__main__':
    main()
