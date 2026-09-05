import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title='TwinPulse', layout='wide')

@st.cache_data
def make_data():
    rng=np.random.default_rng(42)
    stations=[f'S{i:02d}' for i in range(1,31)]
    rows=[]
    for vehicle in range(1,401):
        for i,s in enumerate(stations,1):
            cycle=max(35,60+rng.normal(0,3))
            queue=max(0,int(rng.normal(3,1.5)))
            torque=100+rng.normal(0,4)
            temp=65+rng.normal(0,2)
            vib=1.2+rng.normal(0,.12)
            if s=='S08' and vehicle>220:
                drift=(vehicle-220)/180
                cycle+=18*drift; queue+=int(8*drift); torque+=10*drift
                temp=np.nan; vib=np.nan
            rows.append([vehicle,s,cycle,queue,torque,temp,vib])
    return pd.DataFrame(rows,columns=['vehicle','station','cycle_time','queue_length','torque','temperature','vibration'])

df=make_data()
st.sidebar.header('Twin Controls')
vehicle=st.sidebar.slider('Production progress',1,400,400)
mode=st.sidebar.radio('Operating mode',['Live line','S08 degradation scenario'],index=1)
if mode=='Live line': vehicle=min(vehicle,220)
view=df[df.vehicle<=vehicle]
latest=view.sort_values('vehicle').groupby('station').tail(1).set_index('station')
stations=[f'S{i:02d}' for i in range(1,31)]

st.title('⚙️ TwinPulse')
st.caption('Adaptive Digital Twin for Manufacturing Lines | MVP demonstration using synthetic production data')

# risk
base=df[df.vehicle<=200].groupby('station')[['cycle_time','queue_length','torque']].mean()
cur=latest[['cycle_time','queue_length','torque']]
z=(cur-base.reindex(cur.index))/base.reindex(cur.index)
risk=(100*(0.45*np.clip(z.cycle_time,0,2)/2+0.35*np.clip(z.queue_length,0,2)/2+0.2*np.clip(z.torque,0,2)/2)).clip(0,100)
latest['risk']=risk.fillna(0)
latest['confidence']=np.where(latest.index=='S08',82,94)
latest['status']=np.where(latest.risk>55,'AT RISK',np.where(latest.risk>30,'WATCH','NORMAL'))

c1,c2,c3,c4=st.columns(4)
c1.metric('Active stations',30)
c2.metric('Vehicles simulated',vehicle)
c3.metric('Highest bottleneck risk',f"{latest.risk.max():.0f}%")
c4.metric('Priority station',latest.risk.idxmax())

st.subheader('Production Line Digital Twin')
cols=st.columns(6)
for idx,s in enumerate(stations):
    r=latest.loc[s]
    with cols[idx%6]:
        icon='🟢' if r.status=='NORMAL' else ('🟡' if r.status=='WATCH' else '🔴')
        st.markdown(f"**{icon} {s}**")
        st.caption(f"Risk {r.risk:.0f}% | Q {r.queue_length}")

left,right=st.columns([1.1,1])
with left:
    st.subheader('Early-Risk Signal')
    s08=view[view.station=='S08']
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=s08.vehicle,y=s08.cycle_time,name='Cycle time (s)'))
    fig.add_trace(go.Scatter(x=s08.vehicle,y=s08.queue_length,name='Queue length',yaxis='y2'))
    fig.update_layout(height=330,xaxis_title='Vehicle sequence',yaxis_title='Cycle time',yaxis2=dict(title='Queue',overlaying='y',side='right'))
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.subheader('S08 Explainable Alert')
    r=latest.loc['S08']
    if mode=='S08 degradation scenario' and vehicle>220:
        st.error(f"🚨 Bottleneck risk: {r.risk:.0f}%")
        st.write('**Why:** cycle time ↑ + queue growth ↑ + torque drift ↑')
        st.write('**Direct sensors unavailable:** temperature, vibration')
        st.write(f"**Inference confidence:** {r.confidence:.0f}%")
        st.write('**Likely downstream exposure:** S09 → S14')
    else:
        st.success('No elevated S08 bottleneck signal in the current production window.')

st.subheader('Measured vs AI-Inferred State')
source=pd.DataFrame({
    'Signal':['Cycle time','Queue length','Torque','Temperature','Vibration'],
    'State':['Measured','Measured','Measured','AI-inferred' if mode=='S08 degradation scenario' else 'Measured','AI-inferred' if mode=='S08 degradation scenario' else 'Measured'],
    'Value':[f"{latest.loc['S08','cycle_time']:.1f} s",f"{latest.loc['S08','queue_length']:.0f}",f"{latest.loc['S08','torque']:.1f}", 'Unavailable → inferred' if mode=='S08 degradation scenario' else f"{latest.loc['S08','temperature']:.1f} °C", 'Unavailable → inferred' if mode=='S08 degradation scenario' else f"{latest.loc['S08','vibration']:.2f}"]
})
st.dataframe(source,use_container_width=True,hide_index=True)

with st.expander('Technical mechanism implemented in this MVP'):
    st.write('Station-specific baseline → deviation detection → combine available signals → risk score → affected-station identification → downstream exposure → confidence-aware presentation.')
