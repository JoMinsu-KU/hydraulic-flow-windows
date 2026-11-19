"""Reconstruct PONAR minutes, features, predictions and day weights independently."""
import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median

for env in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[env]="1"
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
RUN = HERE/"runs/u-ponar-public-transfer"


def audit(source):
    result=json.loads((RUN/"results.json").read_bytes())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==result["archive_sha256"]
    minutes=defaultdict(dict)
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            plate=Path(name).stem.split("_")[1]
            with archive.open(name) as binary:
                reader=csv.DictReader(io.TextIOWrapper(binary,encoding="utf8",newline=""))
                for row in reader:
                    dt=datetime.fromisoformat(row["Czas"])
                    assert row["Czas"]==row["Czas2"]
                    key=(plate,dt.replace(second=0).isoformat())
                    assert dt.second not in minutes[key]
                    def number(s):
                        return float(s) if s else np.nan
                    minutes[key][dt.second]=(number(row["Flow - output"]),number(row["Pressure - output"]))
    samples={}
    for key, by_second in minutes.items():
        if len(by_second)!=60:
            continue
        z=np.array([by_second[t] for t in range(60)])
        if not np.isfinite(z).all():
            continue
        pressure=z[:,1]
        loaded=all(25<=np.mean(v)<=235 for v in (pressure[:5],pressure[45:55]))
        samples[key]=(z[:,0],loaded)
    with (RUN/"predictions.csv").open(newline="") as fp:
        saved={(r["analysis"],r["method"],r["plate"],r["minute_utc"]):r for r in csv.DictReader(fp)}
    checked=0
    n_forests=0
    for analysis,need_load in (("all_complete",False),("operating_pressure",True)):
        selected=sorted(k for k,(_,loaded) in samples.items() if loaded or not need_load)
        train=[k for k in selected if k[1][:4]=="2023"]
        test=[k for k in selected if k[1][:4]=="2024"]
        assert set(train).isdisjoint(test)
        for arm in ("single_shape","pair_shape","two_mean_difference","single_level_mean"):
            labels=np.array([int(k[0]!="OT") for k in train])
            def representation(k):
                q=samples[k][0]
                if arm=="single_level_mean":
                    return np.mean(q[50:55])
                if arm=="two_mean_difference":
                    return np.mean(q[:5])-np.mean(q[50:55])
                z=q[45:55] if arm=="single_shape" else np.r_[q[:5],q[50:55]]
                d=z-np.mean(z)
                return np.r_[d,np.std(d),np.ptp(d),np.mean(d[:5]),np.mean(d[5:])]
            x=np.array([representation(k) for k in train])
            x_test=np.array([representation(k) for k in test])
            y_test=np.array([int(k[0]!="OT") for k in test])
            if arm.endswith("shape"):
                est=ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,max_features=1.0,class_weight="balanced",n_jobs=1,random_state=181119)
                pred=est.fit(x,labels).predict(x_test)
                robust=(pred==y_test)
                n_forests+=1
            else:
                m=[median(x[labels==y]) for y in (0,1)]
                def predict(x):
                    return min(range(2),key=lambda y:(abs(x-m[y]),y))
                pred=np.array([predict(z) for z in x_test])
                b=.30 if arm=="single_level_mean" else 0.
                robust=np.array([predict(z-b)==y==predict(z+b) for z,y in zip(x_test,y_test)])
            days=defaultdict(list)
            for i,(plate,stamp) in enumerate(test):
                record=saved[analysis,arm,plate,stamp]
                assert int(record["prediction"])==pred[i] and int(record["robust_correct"])==robust[i]
                if plate!="UT1":
                    days[plate,stamp[:10]].append(bool(robust[i]))
                checked+=1
            recalls={key:np.mean(v) for key,v in days.items()}
            official=result["experiments"][analysis]["methods"][arm]["primary"]["robust_recall"]["date_macro_ba"]
            assert np.isclose(official,np.mean(list(recalls.values()))) and len(recalls)==4
    assert len(saved)==checked
    out={"completed_utc":datetime.now(timezone.utc).isoformat(),"independent_source":"Python csv per absolute UTC minute, independent representations and split, sklearn refit",
         "published_prediction_rows_checked":checked,"fit_forests":n_forests,
         "unique_complete_minutes":len(samples),"source_recording_date_macro_scores_verified":8,
         "scope":"PONAR only, no ZeMA rows","results_sha256":hashlib.sha256((RUN/"results.json").read_bytes()).hexdigest()}
    print(json.dumps(out))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",required=True,type=Path,help="Path to the original v1 PONAR ZIP")
    audit(parser.parse_args().archive)
