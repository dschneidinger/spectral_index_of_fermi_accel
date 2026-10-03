import os, sys, glob, json, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
S3 = "4.226978816696084932553273"
S2 = "3.2985080563907181433382986"

def load_glob(pattern):
    out = {}
    for f in glob.glob(pattern):
        out.update(json.load(open(f)))
    return out

def aniso_log(path):
    D = {}
    for line in open(path):
        m = re.search(r"N= *(\d+).*F.\/F=\[([0-9.]+)", line)
        if m:
            D[int(m.group(1))] = m.group(2)
    return D
