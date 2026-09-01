from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import json
sys.path.insert(0, str(Path(__file__).parent/'src'))
from analytics import load_data, preprocess, clean_numeric_ranges
from report_generator import generate_pdf_report
from live_weather import fetch_current_weather, UK_CITIES

st.set_page_config(page_title='SmartRoad', page_icon='🚦', layout='wide')
st.title('SmartRoad - Traffic Accident Analytics')
st.caption('Computational Analytics Project | Severity prediction, hotspot analysis and safety insights')

@st.cache_data

def load_selected(path):
    return clean_numeric_ranges(preprocess(load_data(path)))

@st.cache_resource

def load_model(path):
    return joblib.load(path)

uploaded=st.file_uploader('Upload a collision CSV (official DfT STATS19 data or compatible CSV)',type=['csv'])
data_path=None
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.csv') as f:
        f.write(uploaded.getvalue()); data_path=f.name
else:
    official=Path('data/road_collisions_2025.csv')
    demo=Path('data/demo_accidents.csv')
    data_path=str(official if official.exists() else demo)
    st.info('Using official 2025 DfT STATS19 collision dataset.')

df=load_selected(data_path)

with st.sidebar:
    st.header('Filters')
    severity_options=['All']+sorted(df['severity_label'].dropna().unique().tolist())
    sev=st.selectbox('Severity',severity_options)
    if sev!='All': df=df[df['severity_label']==sev]
    if 'urban_or_rural_area' in df:
        vals=sorted(df['urban_or_rural_area'].dropna().unique().tolist())
        ur=st.multiselect('Urban/Rural code',vals,default=vals)
        df=df[df['urban_or_rural_area'].isin(ur)]

c1,c2,c3,c4=st.columns(4)
c1.metric('Accidents',f'{len(df):,}')
c2.metric('High severity',f'{int(df.high_severity.sum()):,}')
c3.metric('High severity rate',f'{100*df.high_severity.mean():.1f}%')
c4.metric('Avg vehicles',f'{df.number_of_vehicles.mean():.2f}' if 'number_of_vehicles' in df else 'N/A')

left,right=st.columns(2)
with left:
    by_hour=df.groupby('hour').size().reindex(range(24),fill_value=0).reset_index(name='accidents')
    st.plotly_chart(px.line(by_hour,x='hour',y='accidents',markers=True,title='Accidents by Hour'),use_container_width=True)
with right:
    sev_counts=df['severity_label'].value_counts().rename_axis('severity').reset_index(name='accidents')
    st.plotly_chart(px.bar(sev_counts,x='severity',y='accidents',title='Severity Distribution'),use_container_width=True)

if {'latitude','longitude'}.issubset(df.columns):
    geo=df.dropna(subset=['latitude','longitude']).sample(min(3000,len(df)),random_state=42)
    st.plotly_chart(px.scatter_geo(geo,lat='latitude',lon='longitude',color='severity_label',hover_data=['hour','speed_limit'],scope='europe',title='Accident Locations'),use_container_width=True)

st.subheader('High-Risk Zone Clustering')
if {'latitude','longitude'}.issubset(df.columns):
    geo=df.dropna(subset=['latitude','longitude']).copy()
    if len(geo)>=8:
        from sklearn.cluster import KMeans
        k=st.slider('Number of hotspot clusters',4,12,8)
        km=KMeans(n_clusters=k,n_init=10,random_state=42)
        geo['cluster']=km.fit_predict(geo[['latitude','longitude']])
        summ=geo.groupby('cluster').agg(accidents=('high_severity','size'),high_severity_rate=('high_severity','mean'),latitude=('latitude','mean'),longitude=('longitude','mean')).reset_index()
        summ['risk_score']=(summ['high_severity_rate']*100)*np.log1p(summ['accidents'])
        st.dataframe(summ.sort_values('risk_score',ascending=False).style.format({'high_severity_rate':'{:.1%}','latitude':'{:.4f}','longitude':'{:.4f}','risk_score':'{:.2f}'}),use_container_width=True)

st.subheader('Live Weather Conditions')
st.write('Current weather context from the Open-Meteo API. These live values are shown for situational context and are not injected into the trained 2025 classifier, which was trained on historical STATS19 variables.')
with st.form('live_weather_form'):
    city = st.selectbox('UK location', list(UK_CITIES), index=list(UK_CITIES).index('London'))
    weather_submit = st.form_submit_button('Get Live Weather')
if weather_submit:
    try:
        live = fetch_current_weather(city)
        a,b,c,d = st.columns(4)
        a.metric('Temperature', f"{live.get('temperature_2m', 'N/A')} °C")
        b.metric('Feels like', f"{live.get('apparent_temperature', 'N/A')} °C")
        c.metric('Wind', f"{live.get('wind_speed_10m', 'N/A')} km/h")
        d.metric('Precipitation', f"{live.get('precipitation', 'N/A')} mm")
        st.success(f"{city}: {live.get('weather_description', 'Unknown')}")
        st.caption(f"Live data retrieved from Open-Meteo at {live.get('time', 'current time')} (local time).")
    except Exception as exc:
        st.error(f'Could not retrieve live weather right now: {exc}')

st.subheader('Model Performance')
summary_path=Path('outputs/run_summary.json')
if summary_path.exists():
    try:
        summary=json.loads(summary_path.read_text())
        comp=summary.get('model_comparison', {})
        if comp:
            cmp_df=pd.DataFrame(comp).T.reset_index(names='Model')
            show_cols=[c for c in ['accuracy','precision','recall','f1','cv_f1_mean'] if c in cmp_df.columns]
            st.dataframe(cmp_df[['Model']+show_cols].style.format({c:'{:.3f}' for c in show_cols}),use_container_width=True)
        cm_path=Path('outputs/07_confusion_matrix.png')
        if cm_path.exists():
            st.image(str(cm_path),caption=f"Confusion matrix — {summary.get('selected_model','selected model')}")
    except Exception as exc:
        st.warning(f'Could not load saved evaluation results: {exc}')

st.subheader('Severity Prediction')
st.write('Prediction target: high-severity collision (Fatal or Serious) vs Slight.')
model_path=Path('models/smartroad_model.joblib')
if model_path.exists():
    model=load_model(str(model_path))
    road_type_map = {
        1:'Roundabout', 2:'One-way street', 3:'Dual carriageway',
        6:'Single carriageway', 7:'Slip road', 9:'Unknown', 12:'One-way street (code 12)'
    }
    weather_map = {
        1:'Fine / no high winds', 2:'Raining / no high winds', 3:'Snowing / no high winds',
        4:'Fine / high winds', 5:'Raining / high winds', 6:'Snowing / high winds',
        7:'Fog or mist', 8:'Other', 9:'Unknown'
    }
    surface_map = {1:'Dry', 2:'Wet / damp', 3:'Snow', 4:'Frost / ice', 5:'Flood over 3cm', 9:'Unknown'}
    light_map = {1:'Daylight', 4:'Darkness - lights lit', 5:'Darkness - lights unlit', 6:'Darkness - no lighting', 7:'Darkness - lighting unknown'}
    junction_map = {0:'Not at junction', 1:'Roundabout', 2:'Mini-roundabout', 3:'T or staggered junction', 5:'Other junction', 6:'Crossroads', 7:'More than 4 arms / multi-junction', 8:'Using private drive/access'}
    urban_map = {1:'Urban', 2:'Rural', 3:'Unallocated'}
    road_class_map = {1:'Motorway', 2:'A(M)', 3:'A road', 4:'B road', 5:'C road', 6:'Unclassified'}

    with st.form('prediction_form'):
        a,b,c=st.columns(3)
        hour=a.slider('Time of collision',0,23,18,help='Hour in 24-hour format.')
        month=b.slider('Month',1,12,8)
        speed=c.selectbox('Speed limit (mph)',[20,30,40,50,60,70],index=1)

        d,e,f=st.columns(3)
        road_type=d.selectbox('Road type',list(road_type_map),index=list(road_type_map).index(6),format_func=lambda x:f'{road_type_map[x]} ({x})')
        weather=e.selectbox('Weather conditions',list(weather_map),index=0,format_func=lambda x:f'{weather_map[x]} ({x})')
        surface=f.selectbox('Road surface',list(surface_map),index=0,format_func=lambda x:f'{surface_map[x]} ({x})')

        g,h,i=st.columns(3)
        light=g.selectbox('Light conditions',list(light_map),index=0,format_func=lambda x:f'{light_map[x]} ({x})')
        junction=h.selectbox('Junction type',list(junction_map),index=0,format_func=lambda x:f'{junction_map[x]} ({x})')
        urban=i.selectbox('Area type',list(urban_map),index=0,format_func=lambda x:f'{urban_map[x]} ({x})')

        j,k=st.columns(2)
        road_class=j.selectbox('First road class',list(road_class_map),index=2,format_func=lambda x:f'{road_class_map[x]} ({x})')
        vehicles=k.slider('Number of vehicles',1,8,2)

        l,m,n=st.columns(3)
        weekend=l.checkbox('Weekend')
        night=m.checkbox('Night-time')
        rush=n.checkbox('Rush hour',value=True)
        submitted=st.form_submit_button('Predict Severity Risk')

    if submitted:
        X=pd.DataFrame([{
            'hour':hour,'month':month,'weekend':int(weekend),'night':int(night),'rush_hour':int(rush),
            'speed_limit':speed,'number_of_vehicles':vehicles,'multi_vehicle':int(vehicles>=3),
            'road_type':road_type,'junction_detail':junction,'light_conditions':light,
            'weather_conditions':weather,'road_surface_conditions':surface,
            'urban_or_rural_area':urban,'first_road_class':road_class
        }])
        p=int(model.predict(X)[0]); pro=float(model.predict_proba(X)[0,1]) if hasattr(model,'predict_proba') else None
        risk_label = 'HIGH' if p else 'LOW'
        if p:
            st.error(f'Predicted risk: {risk_label}')
        else:
            st.success(f'Predicted risk: {risk_label}')
        if pro is not None: st.metric('High-severity probability',f'{pro:.1%}')
else:
    st.warning('Model file not found. Run: python run_analysis.py')

st.subheader('Automated PDF Report')
st.write('Generate a self-contained analytical PDF report from the current real-data results and saved charts.')
if st.button('Generate Automated PDF Report'):
    try:
        report_path = Path('outputs/SmartRoad_Automated_Report.pdf')
        generated = generate_pdf_report(
            summary_path='outputs/run_summary.json',
            output_path=report_path,
            charts_dir='outputs',
        )
        st.success('Automated PDF report generated successfully.')
        st.download_button(
            'Download Automated PDF Report',
            data=generated.read_bytes(),
            file_name=generated.name,
            mime='application/pdf',
        )
    except Exception as exc:
        st.error(f'Could not generate the PDF report: {exc}')

st.subheader('Data Quality')
q=pd.DataFrame({'metric':['Rows after duplicate removal','Missing cells','Columns'], 'value':[len(df),int(df.isna().sum().sum()),df.shape[1]]})
st.dataframe(q,use_container_width=True)
