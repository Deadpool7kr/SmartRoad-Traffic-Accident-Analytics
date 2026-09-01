from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings('ignore')

OFFICIAL_URL = 'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2025.csv'

COLUMN_ALIASES = {
    'accident_index':['accident_index','accidentindex','accident_id','accidentid'],
    'latitude':['latitude','lat','start_lat'],
    'longitude':['longitude','lng','lon','start_lng'],
    'accident_severity':['accident_severity','collision_severity','severity'],
    'number_of_vehicles':['number_of_vehicles','vehicles','num_units'],
    'number_of_casualties':['number_of_casualties','casualties','injuries_total'],
    'date':['date','crash_date','start_date'],
    'day_of_week':['day_of_week','day'],
    'time':['time','start_time'],
    'road_type':['road_type'],
    'speed_limit':['speed_limit'],
    'junction_detail':['junction_detail'],
    'light_conditions':['light_conditions','lighting_condition'],
    'weather_conditions':['weather_conditions','weather','weather_condition'],
    'road_surface_conditions':['road_surface_conditions','roadway_surface_cond'],
    'urban_or_rural_area':['urban_or_rural_area','urban_rural','urban_or_rural'],
    'first_road_class':['first_road_class'],
}

# DfT STATS19 collision severity coding used by the raw collision CSV:
# 1 = Fatal, 2 = Serious, 3 = Slight.
SEVERITY_LABELS_DFT = {1:'Fatal',2:'Serious',3:'Slight',4:'Slight'}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    original_cols = {str(c).strip().lower().replace(' ','_').replace('-','_') for c in d.columns}
    severity_scheme = 'collision' if 'collision_severity' in original_cols else 'accident'

    d.columns = [str(c).strip().lower().replace(' ','_').replace('-','_') for c in d.columns]
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in d.columns:
                rename[a] = canonical
                break
    d = d.rename(columns=rename)
    d['_severity_scheme'] = severity_scheme
    return d

def load_data(path: str|Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    required = ['accident_severity','date','time']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.drop_duplicates().reset_index(drop=True)
    d['date'] = pd.to_datetime(d['date'], dayfirst=True, errors='coerce')
    d['time_parsed'] = pd.to_datetime(d['time'].astype(str), format='%H:%M', errors='coerce')
    d['hour'] = d['time_parsed'].dt.hour
    d['month'] = d['date'].dt.month
    d['day_name'] = d['date'].dt.day_name()
    d['weekend'] = (d['date'].dt.dayofweek >= 5).astype(int)
    d['night'] = ((d['hour'] < 6) | (d['hour'] >= 22)).astype(int)
    d['rush_hour'] = d['hour'].isin([7,8,9,16,17,18,19]).astype(int)
    # The DfT raw collision dataset uses the STATS19 accident_severity coding
    # even when a source/export labels the field as collision_severity:
    # 1 = Fatal, 2 = Serious, 3 = Slight.
    sev = pd.to_numeric(d['accident_severity'], errors='coerce')
    d['severity_label'] = sev.map(SEVERITY_LABELS_DFT).fillna('Unknown')
    d['high_severity'] = d['severity_label'].isin(['Fatal','Serious']).astype(int)
    # Guard against an inverted severity mapping silently producing a nearly
    # all-positive target, which would make model scores misleading.
    known_rate = d.loc[sev.isin(SEVERITY_LABELS_DFT.keys()), 'high_severity']
    if len(known_rate) and float(known_rate.mean()) > 0.50:
        raise ValueError('Severity mapping produced an implausibly high high_severity rate; check STATS19 severity coding.')
    if 'number_of_vehicles' in d.columns:
        d['multi_vehicle'] = (pd.to_numeric(d['number_of_vehicles'], errors='coerce') >= 3).astype(float)
    return d

def clean_numeric_ranges(d: pd.DataFrame) -> pd.DataFrame:
    out=d.copy()
    for c in ['speed_limit','number_of_vehicles','number_of_casualties','latitude','longitude','hour','month']:
        if c in out: out[c]=pd.to_numeric(out[c], errors='coerce')
    if 'speed_limit' in out:
        out.loc[(out.speed_limit<0)|(out.speed_limit>100),'speed_limit']=np.nan
    if 'latitude' in out:
        out.loc[(out.latitude<-90)|(out.latitude>90),'latitude']=np.nan
    if 'longitude' in out:
        out.loc[(out.longitude<-180)|(out.longitude>180),'longitude']=np.nan
    return out

def build_model(df: pd.DataFrame, model_name='Random Forest'):
    d=clean_numeric_ranges(df)
    numeric=['hour','month','weekend','night','rush_hour','speed_limit','number_of_vehicles','multi_vehicle']
    categorical=['road_type','junction_detail','light_conditions','weather_conditions','road_surface_conditions','urban_or_rural_area','first_road_class']
    numeric=[c for c in numeric if c in d.columns]
    categorical=[c for c in categorical if c in d.columns]
    features=numeric+categorical
    X=d[features].copy(); y=d['high_severity'].copy()
    stratify = y if y.nunique()>1 else None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=stratify)

    if model_name == 'Gradient Boosting':
        # HistGradientBoosting requires numeric input, so categorical values are
        # ordinal-encoded with a reserved value for unseen categories.
        prep=ColumnTransformer([
            ('num',Pipeline([ ('imputer',SimpleImputer(strategy='median')) ]),numeric),
            ('cat',Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]),categorical)
        ])
        estimator=HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=30, class_weight='balanced', random_state=42
        )
    else:
        num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median'))])
        cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                           ('onehot',OneHotEncoder(handle_unknown='ignore'))])
        prep=ColumnTransformer([('num',num_pipe,numeric),('cat',cat_pipe,categorical)])
        if model_name=='Logistic Regression':
            estimator=LogisticRegression(max_iter=1000,class_weight='balanced')
        else:
            estimator=RandomForestClassifier(n_estimators=250,random_state=42,
                class_weight='balanced_subsample',n_jobs=-1,min_samples_leaf=2)

    pipe=Pipeline([('preprocess',prep),('model',estimator)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    metrics={
        'accuracy':accuracy_score(y_test,pred),
        'precision':precision_score(y_test,pred,zero_division=0),
        'recall':recall_score(y_test,pred,zero_division=0),
        'f1':f1_score(y_test,pred,zero_division=0),
        'n_train':len(X_train),'n_test':len(X_test),'positive_rate':float(y.mean())
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_f1=cross_val_score(pipe,X,y,cv=cv,scoring='f1',n_jobs=1)
    metrics['cv_f1_mean']=float(cv_f1.mean()); metrics['cv_f1_std']=float(cv_f1.std())
    return pipe, X_test, y_test, pred, metrics, features

def permutation_importance_table(pipe, X_test, y_test, n_repeats=8):
    r=permutation_importance(pipe,X_test,y_test,n_repeats=n_repeats,random_state=42,scoring='f1',n_jobs=1)
    return pd.DataFrame({'feature':X_test.columns,'importance_mean':r.importances_mean,'importance_std':r.importances_std}).sort_values('importance_mean',ascending=False)

def fit_cluster(df: pd.DataFrame, n_clusters=8):
    if not {'latitude','longitude'}.issubset(df.columns): return pd.DataFrame()
    geo=df[['latitude','longitude']].copy()
    geo['latitude']=pd.to_numeric(geo.latitude,errors='coerce'); geo['longitude']=pd.to_numeric(geo.longitude,errors='coerce')
    valid=geo.notna().all(axis=1)
    idx=df.index[valid]
    km=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
    labels=km.fit_predict(geo.loc[valid])
    out=df.loc[idx,['latitude','longitude','high_severity']].copy()
    out['cluster']=labels
    summary=out.groupby('cluster').agg(accidents=('high_severity','size'), high_severity_rate=('high_severity','mean'), latitude=('latitude','mean'), longitude=('longitude','mean')).reset_index()
    return out, summary

def make_charts(df: pd.DataFrame, out_dir: str|Path):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style='whitegrid')
    # 1 line chart by hour
    h=df.groupby('hour',dropna=False).size().reindex(range(24),fill_value=0)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(h.index,h.values,marker='o'); ax.set(title='Accidents by Hour',xlabel='Hour',ylabel='Accident Count'); fig.tight_layout(); fig.savefig(out_dir/'01_accidents_by_hour.png',dpi=150); plt.close(fig)
    # 2 bar severity
    sev=df['severity_label'].value_counts().reindex(['Fatal','Serious','Slight']).fillna(0)
    fig,ax=plt.subplots(figsize=(8,5)); sev.plot.bar(ax=ax); ax.set(title='Accident Severity Distribution',xlabel='',ylabel='Count'); fig.tight_layout(); fig.savefig(out_dir/'02_severity_distribution.png',dpi=150); plt.close(fig)
    # 3 histogram speed
    fig,ax=plt.subplots(figsize=(8,5)); df['speed_limit'].dropna().plot.hist(ax=ax,bins=12); ax.set(title='Speed Limit Distribution',xlabel='Speed limit'); fig.tight_layout(); fig.savefig(out_dir/'03_speed_limit_histogram.png',dpi=150); plt.close(fig)
    # 4 heatmap hour vs weekday
    ctab=pd.crosstab(df['day_name'],df['hour']).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    fig,ax=plt.subplots(figsize=(11,5)); sns.heatmap(ctab,ax=ax,cmap='Blues'); ax.set(title='Accident Heatmap: Weekday vs Hour',xlabel='Hour',ylabel='Day'); fig.tight_layout(); fig.savefig(out_dir/'04_weekday_hour_heatmap.png',dpi=150); plt.close(fig)
    # 5 boxplot speed by severity
    plot_df=df[df['severity_label'].isin(['Fatal','Serious','Slight'])]
    fig,ax=plt.subplots(figsize=(8,5)); sns.boxplot(data=plot_df,x='severity_label',y='speed_limit',ax=ax,order=['Fatal','Serious','Slight']); ax.set(title='Speed Limit by Accident Severity',xlabel='',ylabel='Speed Limit'); fig.tight_layout(); fig.savefig(out_dir/'05_speed_by_severity_boxplot.png',dpi=150); plt.close(fig)
    # 6 scatter geo
    if {'latitude','longitude'}.issubset(df.columns):
        geo=df.dropna(subset=['latitude','longitude']).sample(min(2500,len(df)),random_state=42)
        fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(geo['longitude'],geo['latitude'],s=6,alpha=.25); ax.set(title='Geographic Distribution of Accidents',xlabel='Longitude',ylabel='Latitude'); fig.tight_layout(); fig.savefig(out_dir/'06_accident_scatter_map.png',dpi=150); plt.close(fig)
    return sorted(str(p.name) for p in out_dir.glob('*.png'))

def run_pipeline(data_path: str, out_dir: str, model_path: str):
    df0=load_data(data_path); df=preprocess(df0); df=clean_numeric_ranges(df)
    charts=make_charts(df,out_dir)
    candidates={}
    for name in ['Random Forest','Logistic Regression','Gradient Boosting']:
        candidates[name]=build_model(df,name)
    chosen_name=max(candidates, key=lambda name: candidates[name][4]['cv_f1_mean'])
    chosen, X_test, y_test, pred, metrics, features = candidates[chosen_name]
    # Evaluate both models on their own held-out sets for comparison.
    comparison={name:{k:float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v for k,v in vals[4].items()} for name,vals in candidates.items()}
    imp=permutation_importance_table(chosen,X_test,y_test)
    joblib.dump(chosen,model_path)
    # Save confusion matrix for the report/submission evidence.
    cm=confusion_matrix(y_test,pred)
    fig,ax=plt.subplots(figsize=(5.5,4.5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,cbar=False); ax.set(title=f'Confusion Matrix - {chosen_name}',xlabel='Predicted',ylabel='Actual'); fig.tight_layout(); fig.savefig(Path(out_dir)/'07_confusion_matrix.png',dpi=150); plt.close(fig)
    cmp=pd.DataFrame(comparison).T[['accuracy','precision','recall','f1','cv_f1_mean']]
    fig,ax=plt.subplots(figsize=(9,5)); cmp.plot.bar(ax=ax); ax.set(title='Model Comparison',ylabel='Score',ylim=(0,1)); ax.tick_params(axis='x',rotation=0); fig.tight_layout(); fig.savefig(Path(out_dir)/'08_model_comparison.png',dpi=150); plt.close(fig)
    cluster_info=fit_cluster(df,8)
    if cluster_info and isinstance(cluster_info,tuple):
        cluster_info[1].to_csv(Path(out_dir)/'cluster_summary.csv',index=False)
    summary={'rows_raw':len(df0),'rows_clean':len(df),'charts':sorted([str(p.name) for p in Path(out_dir).glob('*.png')]),'selected_model':chosen_name,'metrics':metrics,'model_comparison':comparison,'features':features,'top_features':imp.head(10).to_dict('records')}
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    Path(out_dir,'run_summary.json').write_text(json.dumps(summary,indent=2))
    return summary, df, imp, cluster_info
