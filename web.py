import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# res = requests.get('http://localhost:8000/')
# st.write(res.json())


st.header('클레임 빈도-심도 시뮬레이션')

st.sidebar.header('시뮬레이션 설정')

with st.sidebar.expander('설정 불러오기'):
  conf_init = {}
  conf_uploaded = st.file_uploader('')
  conf_apply_btn = st.button('설정 적용')
  if conf_apply_btn and conf_uploaded:
    conf_init = json.loads(conf_uploaded.read())

with st.sidebar.expander('빈도 모형'):
  
  freq_model_list = ['binom', 'poisson', 'nbinom']
  freq_model = st.selectbox('빈도 분포', freq_model_list, index=freq_model_list.index(conf_init['freq_model']) if 'freq_model' in conf_init else 2)
  
  st.caption('빈도 모수')
  if freq_model == 'binom':
    n = st.number_input('n', min_value=1, max_value=100, step=1, 
      value=conf_init['freq_param']['n'] if ('freq_param' in conf_init) and ('n' in conf_init['freq_param']) else 1)
    p = st.number_input('p', min_value=0.001, max_value=1., step=0.001, format='%.3f',
      value=conf_init['freq_param']['p'] if ('freq_param' in conf_init) and ('p' in conf_init['freq_param']) else .1)
    freq_param = dict(n=n, p=p)
  elif freq_model == 'poisson':
    mu = st.number_input('μ', min_value=0.1, max_value=100., step=0.1,
      value=conf_init['freq_param']['mu'] if ('freq_param' in conf_init) and ('mu' in conf_init['freq_param']) else .5)
    freq_param = dict(mu=mu)
  elif freq_model == 'nbinom':
    n = st.number_input('n', min_value=1, max_value=100, step=1,
      value=conf_init['freq_param']['n'] if ('freq_param' in conf_init) and ('n' in conf_init['freq_param']) else 1)
    p = st.number_input('p', min_value=0.001, max_value=1., step=0.001, format='%.3f',
      value=conf_init['freq_param']['p'] if ('freq_param' in conf_init) and ('p' in conf_init['freq_param']) else .1)
    freq_param = dict(n=n, p=p)
  else:
    pass

  
with st.sidebar.expander('심도 모형'):
    sev_model_list = ['norm', 'gamma', 'lognorm']
    sev_model = st.selectbox('빈도 분포', sev_model_list, index=sev_model_list.index(conf_init['sev_model']) if 'sev_model' in conf_init else 2)

    st.caption('심도 모수')
    loc = st.number_input('loc', min_value=-1e4, max_value=1e4, step=1e3, format='%.0f',
      value=conf_init['sev_param']['loc'] if ('sev_param' in conf_init) and ('loc' in conf_init['sev_param']) else 0.)
    scale = st.number_input('scale', min_value=0., max_value=1e4, step=1e3, format='%.0f'
      , value=conf_init['sev_param']['scale'] if ('sev_param' in conf_init) and ('scale' in conf_init['sev_param']) else 1e3)
    if sev_model == 'norm':
      sev_param = dict(loc=loc, scale=scale)
    elif sev_model == 'gamma':
      a = st.number_input('a', min_value=1., max_value=10., step=0.1,
        value=conf_init['sev_param']['a'] if ('sev_param' in conf_init) and ('a' in conf_init['sev_param']) else 1.)
      sev_param = dict(a=a, loc=loc, scale=scale)
    elif sev_model == 'lognorm':
      s = st.number_input('s', min_value=0.1, max_value=10., step=0.1,
        value=conf_init['sev_param']['s'] if ('sev_param' in conf_init) and ('s' in conf_init['sev_param']) else 1.)
      sev_param = dict(s=s, loc=loc, scale=scale)
    else:
      pass


with st.sidebar.expander('시나리오 설정'):
  exposure = st.number_input('익스포져 계수', min_value=0.1, max_value=10., step=0.1, format='%.1f',
    value=conf_init['exposure'] if 'exposure' in conf_init else 1.)
  num_scen = st.number_input('시나리오 수', min_value=100, max_value=2000, step=10,
    value=conf_init['num_scen'] if 'num_scen' in conf_init else 200)


conf = dict(freq_model=freq_model, freq_param=freq_param,
      sev_model=sev_model, sev_param=sev_param, exposure=exposure, num_scen=num_scen)

col1, col2 = st.sidebar.columns([1, 2])
with col1:
  run_btn = st.button('실행')
with col2:
  now = datetime.now().strftime('%Y%m%d%H%M%S')
  conf_download = st.download_button('설정 저장', data=json.dumps(conf), file_name=f'conf_{now}.json', mime='application/json')
 

if run_btn:

  # API 요청
  loss_scen = requests.post(
    url='http://127.0.0.1:8000/scen',
    headers={'Content-Type': 'application/json; charset=utf-8', 'accept': 'application/json'},
    data=json.dumps(conf)
  ).json()
  loss_scen_df = pd.DataFrame.from_dict(loss_scen)

  # 요약 통계량
  st.subheader('요약 통계량')
  loss_scen_summary = loss_scen_df.describe(percentiles=[0.75, 0.95, 0.995]).T.drop('count', axis=1)
  st.table(loss_scen_summary)

  # 히스토그램
  st.subheader('히스토그램')
  fig = px.histogram(loss_scen_df, x='num_claims', histnorm='probability')
  fig.update_layout(dict())
  st.plotly_chart(fig)
  
  fig = px.histogram(loss_scen_df, x='num_claims', histnorm='probability', cumulative=True)
  st.plotly_chart(fig)

  fig = px.histogram(loss_scen_df, x='total_loss', histnorm='probability')
  st.plotly_chart(fig)

  fig = px.histogram(loss_scen_df, x='total_loss', histnorm='probability', cumulative=True)
  st.plotly_chart(fig)

  # 결과 다운
  now = datetime.now().strftime('%Y%m%d%H%M%S')
  result_download = st.download_button('결과 다운', data=loss_scen_df.to_csv(index=False),
    file_name=f'loss_scen_{now}.csv', mime='text/csv')