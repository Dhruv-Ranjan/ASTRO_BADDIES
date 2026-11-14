"""
AURORA V5 publication-readiness validation.
Selects models on validation RMSE, evaluates once on chronological test data,
and compares against persistence and mean baselines.
"""
import argparse, json, logging, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
log=logging.getLogger("AURORA")
TARGETS={"surface":"future_30d_surface_soil_mean","root_zone":"future_30d_root_zone_soil_mean","profile":"future_30d_profile_soil_mean"}
STATE={"surface":"sm_surface_m3m3","root_zone":"sm_root_zone_m3m3","profile":"sm_profile_mean_m3m3"}

def models(seed):
    return {
    "Ridge":Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=10))]),
    "RandomForest":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=3,max_features=.7,random_state=seed,n_jobs=1))]),
    "ExtraTrees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=500,max_depth=10,min_samples_leaf=3,max_features=.7,random_state=seed,n_jobs=1))]),
    "GradientBoosting":Pipeline([("imp",SimpleImputer(strategy="median")),("m",GradientBoostingRegressor(n_estimators=250,learning_rate=.035,max_depth=2,min_samples_leaf=5,loss="huber",random_state=seed))])}

def score(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); rm=math.sqrt(mean_squared_error(y,p))
    r=pearsonr(y,p)[0] if np.std(y)>0 and np.std(p)>0 else np.nan
    return {"MAE":mean_absolute_error(y,p),"RMSE":rm,"R2":r2_score(y,p),"Pearson_r":r,
            "target_std":np.std(y,ddof=1),"NRMSE":rm/np.std(y,ddof=1),"n":len(y)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-file",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--random-state",type=int,default=42); ap.add_argument("--rolling-folds",type=int,default=5)
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/"run.log",encoding="utf8")])
    df=pd.read_csv(a.input_file); df["datetime_utc"]=pd.to_datetime(df["datetime_utc"],errors="coerce")
    df=df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)
    for c in list(df.columns):
        if c=="datetime_utc": continue
        n=pd.to_numeric(df[c],errors="coerce")
        if n.notna().sum()==0 or n.nunique(dropna=True)<=1: df=df.drop(columns=c)
    final=[]; all_reports={}
    for name,target in TARGETS.items():
        y=pd.to_numeric(df[target],errors="coerce"); valid=y.notna()
        dates=df.loc[valid,"datetime_utc"].reset_index(drop=True); y=y[valid].reset_index(drop=True)
        cols=[]
        for c in df.columns:
            if c=="datetime_utc" or c==target or c.startswith("future_") or "rain" in c.lower() or "precip" in c.lower(): continue
            n=pd.to_numeric(df[c],errors="coerce")
            if n.notna().sum()>0 and n.nunique(dropna=True)>1: cols.append(c)
        X=df.loc[valid,cols].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
        n=len(X); tr=int(.6*n); va=int(.8*n)
        Xt,Xv,Xe=X.iloc[:tr],X.iloc[tr:va],X.iloc[va:]
        yt,yv,ye=y.iloc[:tr],y.iloc[tr:va],y.iloc[va:]
        val=[]
        for mn,m in models(a.random_state).items():
            m.fit(Xt,yt); pv=m.predict(Xv); val.append({"model":mn,**score(yv,pv)})
        persistence_val=Xv[STATE[name]].fillna(yt.mean()).to_numpy() if STATE[name] in Xv else np.full(len(yv),yt.mean())
        val += [{"model":"PersistenceBaseline",**score(yv,persistence_val)},{"model":"MeanBaseline",**score(yv,np.full(len(yv),yt.mean()))}]
        vd=pd.DataFrame(val); vd.to_csv(out/f"{name}_validation_metrics.csv",index=False)
        chosen=vd[~vd.model.str.contains("Baseline")].sort_values("RMSE").iloc[0].model
        m=models(a.random_state)[chosen]; m.fit(pd.concat([Xt,Xv]),pd.concat([yt,yv]))
        pred=m.predict(Xe); persistence=Xe[STATE[name]].fillna(pd.concat([yt,yv]).mean()).to_numpy() if STATE[name] in Xe else np.full(len(ye),pd.concat([yt,yv]).mean())
        tm=score(ye,pred); base=score(ye,persistence)
        tm.update({"target":name,"target_column":target,"selected_model":chosen,"baseline_RMSE":base["RMSE"],
                   "RMSE_improvement_vs_persistence_pct":100*(base["RMSE"]-tm["RMSE"])/base["RMSE"]})
        final.append(tm)
        pd.DataFrame({"datetime_utc":dates.iloc[va:],"observed":ye,"predicted":pred,"persistence":persistence}).to_csv(out/f"{name}_test_predictions.csv",index=False)
        pd.DataFrame([tm,{"target":name,"model":"PersistenceBaseline",**base}]).to_csv(out/f"{name}_test_metrics.csv",index=False)
        est=m.named_steps["m"]
        imp=getattr(est,"feature_importances_",None)
        if imp is None: imp=np.abs(getattr(est,"coef_",np.zeros(len(cols))))
        pd.DataFrame({"feature":cols,"importance":imp}).sort_values("importance",ascending=False).to_csv(out/f"{name}_feature_importance.csv",index=False)
        plt.figure(figsize=(12,5)); plt.plot(dates.iloc[va:],ye,label="Observed"); plt.plot(dates.iloc[va:],pred,label="Predicted"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/f"{name}_test_plot.png",dpi=180); plt.close()
        all_reports[name]={"selected_model":chosen,"validation":val,"test":tm,"baseline":base,"features":cols}
    pd.DataFrame(final).to_csv(out/"PUBLICATION_READY_SUMMARY.csv",index=False)
    (out/"AURORA_V5_REPORT.txt").write_text("AURORA V5 REPORT\n\n"+pd.DataFrame(final).to_string(index=False)+"\n\nModels selected on validation RMSE only. Test set evaluated once. Future columns excluded.\n",encoding="utf8")
    (out/"summaries.json").write_text(json.dumps(all_reports,indent=2,default=str),encoding="utf8")
    log.info("Finished successfully. Results saved to %s",out)

if __name__=="__main__": main()
