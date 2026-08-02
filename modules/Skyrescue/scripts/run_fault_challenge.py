#!/usr/bin/env python3
"""Score weak-signal detection with labels opened only after inference."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skyrescue.fault_detection import detect
def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def overlaps(a,b): return a["uav_id"] == b["uav_id"] and a["start_time_s"] < b["end_time_s"] and b["start_time_s"] < a["end_time_s"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    predicted=list(detect(rows(a.dataset / "telemetry.jsonl")))
    truth=list(rows(a.dataset / "faults.jsonl")); hits=sum(any(overlaps(f,x) for x in predicted) for f in truth); matched=sum(any(overlaps(x,f) for f in truth) for x in predicted)
    precision=matched/max(1,len(predicted)); recall=hits/max(1,len(truth)); result={"faults":len(truth),"predicted_intervals":len(predicted),"precision":round(precision,4),"recall":round(recall,4),"f1":round(2*precision*recall/max(.0001,precision+recall),4),"synthetic_data":True,"truth_separation":"labels opened after inference"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__ == "__main__": main()
