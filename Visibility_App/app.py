import streamlit as st
import pandas as pd
import numpy as np
import torch
import importlib
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import datetime
import plotly.graph_objects as go
import os

# ==========================================
# 全局页面配置
# ==========================================
st.set_page_config(
    page_title="机场短临能见度预测系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏 Streamlit 默认 UI 组件，提升前端界面的独立封装感
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 侧边栏：系统参数配置区
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airport.png", width=80)
    st.title("系统配置")
    st.markdown("---")
    
    st.markdown("### 📥 1. 数据导入")
    uploaded_file = st.file_uploader("上传气象观测数据 (CSV)", type=['csv'])
    st.caption("要求：需包含连续 48 小时历史气象特征，且包含 visibility 列。")
    
    st.markdown("### ⚙️ 2. 预警参数设置")
    # 动态 LVP 预警阈值调节控件
    alert_threshold = st.slider("低能见度程序 (LVP) 触发阈值", min_value=400, max_value=1500, value=800, step=100)
    st.caption(f"当前系统设定的 LVP 警报线为：{alert_threshold} 米")
    


# ==========================================
# 核心推理引擎加载模块 (结合 st.cache_resource 实现单例模式，降低 IO 开销)
# ==========================================
@st.cache_resource
def load_model():
    # 模型超参数配置字典
    class Config:
        def __init__(self):
            self.seq_len = 48
            self.label_len = 1
            self.pred_len = 1
            self.freq = 'h'
            self.batch_size = 32
            self.dec_in = 6    # 解码器输入特征维度
            self.enc_in = 6    # 编码器输入特征维度
            self.c_out = 6     # 输出特征维度
            self.d_model = 64
            self.n_heads = 8
            self.dropout = 0.01
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 128
            self.factor = 3
            self.activation = 'gelu'
            self.embed = 'timeF'
            self.output_attention = 0
            self.task_name = 'short_term_forecast'
            
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 动态构建基于 TCN-FECAM-Transformer 的网络拓扑
    model_module = importlib.import_module('models.Transformer')
    net = model_module.Model(config).to(device)
    
    # 载入离线训练的权重字典
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net, device

# ==========================================
# 主界面前端渲染逻辑
# ==========================================
st.title("机场短临能见度预测系统")
st.markdown("基于 TCN-FECAM-Transformer 深度学习模型，实现机场能见度的短临预测，辅助航班调度与决策。")

if uploaded_file is None:
    st.info("👈 请在左侧【系统配置】栏上传测试数据文件以运行推理程序。")
else:
    # 1. 数据解析与特征工程

    df = pd.read_csv(uploaded_file)

    df = df.dropna(subset=['date'])  # 删掉所有没有时间的空行
    df = df.reset_index(drop=True)   # 重新整理序号
    
    data_dim = df[df.columns.drop('date')].shape[1] 
    data = df[df.columns.drop('date')]
    
    tab1, tab2, tab3 = st.tabs(["📊 预测结果与分析", "📋 历史数据预览", "📖 系统说明"])
    
    with tab1:
        with st.spinner('模型正在进行前向推理计算，请稍候...'):
            # 时间戳周期特征编码
            df_stamp = df[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=1, freq='h') 

            # 序列全局归一化
            scaler = MinMaxScaler()
            data_inverse = scaler.fit_transform(np.array(data))
            
            # 独立装载目标变量(visibility)的缩放器，用于后续的一维特征反解
            target_scaler = MinMaxScaler()
            target_scaler.fit(df[['visibility']])
            
            # 滑动窗口样本切片构建
            window = 48
            length_size = 1
            sequence_length = window + length_size
            
            x_temp = np.array([data_inverse[-sequence_length: -length_size]])
            x_temp_mark = np.array([data_stamp[-sequence_length: -length_size]])
            y_temp = np.array([data_inverse[-(length_size + int(window / 2)):]])
            y_temp_mark = np.array([data_stamp[-(length_size + int(window / 2)):]])
            
            # 注入计算图
            net, device = load_model()
            x_enc = torch.tensor(x_temp).type(torch.float32).to(device)
            x_mark_enc = torch.tensor(x_temp_mark).type(torch.float32).to(device)
            x_dec = torch.tensor(y_temp).type(torch.float32).to(device)
            x_mark_dec = torch.tensor(y_temp_mark).type(torch.float32).to(device)
            
            # 2. 模型推理
            with torch.no_grad():
                pred = net(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred = pred.detach().cpu()
                # 取时间维度的最后一个步长，进行逆归一化映射
                pred_uninverse = target_scaler.inverse_transform(pred[:, :, -1].numpy())
                
            # 预测值与基准值计算
            final_pred_value = pred_uninverse[0][0]
            current_visibility = df['visibility'].iloc[-1]
            visibility_change = final_pred_value - current_visibility
            
            # 绝对时间轴推算
            last_time = pd.to_datetime(df['date'].iloc[-1])
            predict_time = last_time + datetime.timedelta(hours=1)
            
        # 3. 核心预测指标渲染
        st.subheader("预测结果")
        st.markdown(f"**数据观测截止时间：** {last_time.strftime('%Y-%m-%d %H:%M:%S')} | **预测目标时间：** **{predict_time.strftime('%Y-%m-%d %H:%M:%S')}**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="当前实况 (T=0)", value=f"{current_visibility:.0f} m", delta="实况观测", delta_color="off")
        col2.metric(label=f"预测能见度 (T+1)", value=f"{final_pred_value:.0f} m", delta=f"较当前变化 {visibility_change:.0f} m")
        
        # 二期多步预测(Multi-step Forecast)路线图占位展示
        col3.metric(label="多步预测规划 (T+n)", value="后续迭代 🚀", delta="二期目标: 长程序列连续预测", delta_color="off")
        
        st.markdown("---")
        
        # 4. LVP 业务判定逻辑引擎
        st.subheader("运行建议")
        if final_pred_value <= alert_threshold:
            st.error(f"⚠️ **警报**：预测能见度 ({final_pred_value:.0f}m) 将触及或低于设定的运行阈值 ({alert_threshold}m)。建议相关部门做好低能见度程序 (LVP) 启动准备。")
        elif final_pred_value <= alert_threshold + 500:
            st.warning(f"🔔 **提示**：预测能见度呈现下降趋势，正逼近起降标准临界值，请密切关注天气演变。")
        else:
            st.success(f"✅ **正常**：预测能见度处于安全范围，满足常规航班起降标准。")
            
        st.markdown("---")
        
        # 5. 可视化模块：交互式时序数据对比追踪图
        st.write("📈 **历史能见度与预测走势对比**")
        
        # 截取近 168 个时间窗口的数据进行前端渲染
        display_window = 168 
        chart_data = df.copy()
        chart_data['date'] = pd.to_datetime(chart_data['date'])
        chart_data = chart_data.set_index('date').tail(display_window)
        
        fig = go.Figure()
        
        # 历史实况数据流
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['visibility'],
            mode='lines',
            name='实况观测',
            line=dict(color='#2E86C1', width=2),
            hovertemplate='%{x|%Y-%m-%d %H:%M} <br>实况: %{y:.0f} m<extra></extra>'
        ))
        
        # T+1 推理数据流（虚线区分）
        last_val = chart_data['visibility'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_time, predict_time],
            y=[last_val, final_pred_value],
            mode='lines+markers',
            name='模型预测',
            line=dict(color='#E74C3C', width=3, dash='dash'), 
            marker=dict(size=8, symbol='circle', color='#E74C3C'),
            hovertemplate='%{x|%Y-%m-%d %H:%M} <br>预测: %{y:.0f} m<extra></extra>'
        ))
        
        # 图表交互与布局优化
        fig.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode='x unified',
            dragmode='pan',
            xaxis=dict(title="时间 (支持滚轮缩放与鼠标拖拽)", showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
            yaxis=dict(title="能见度 (米)", showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
            # 将图例锚定至左下角
            legend=dict(
                x=0.02,
                y=0.05,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(200, 200, 200, 0.5)",
                borderwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # 数据集检视面板
        st.subheader("历史观测数据")
        st.write(f"当前系统已读取 {len(df)} 条记录。展示最后 48 小时数据切片：")
        display_df = df.copy()
        display_df['date'] = pd.to_datetime(display_df['date'])
        display_df = display_df.set_index('date')
        st.dataframe(display_df.tail(48), use_container_width=True)
        
    with tab3:
        # 算法原理说明
        st.subheader("系统工作原理")
        st.markdown(
            """
            * **数据输入**：系统需读取过去连续 48 小时的逐时气象观测数据（含温度、湿度、风向等 6 维特征）。
            * **推理引擎**：基于团队自研的 TCN-FECAM-Transformer 混合模型，提取局部突变特征与长程时序依赖。
            * **结果输出**：推演出紧接该序列的**未来 1 小时 (T+1)** 能见度数值，结合用户自定义的预警阈值，生成业务决策建议。
            """
        )