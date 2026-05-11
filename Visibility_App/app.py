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
    page_title="机场短临能见度预测系统 (12H多步版)",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏 Streamlit 默认 UI 组件
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 核心推理引擎加载
# ==========================================
@st.cache_resource
def load_model():
    class Config:
        def __init__(self):
            self.seq_len = 48
            self.label_len = 24  
            self.pred_len = 12   # 【关键修改】：改为预测未来 12 小时
            self.freq = 'h'
            self.batch_size = 32
            self.dec_in = 6      
            self.enc_in = 6      
            self.c_out = 6       
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
    
    # 动态加载模型定义
    model_module = importlib.import_module('models.Transformer')
    net = model_module.Model(config).to(device)
    
    # 载入权重
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net, device, config.pred_len, config.label_len

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airport.png", width=80)
    st.title("系统配置")
    st.markdown("---")
    
    st.markdown("### 📥 数据导入")
    uploaded_file = st.file_uploader("上传气象观测数据 (CSV)", type=['csv'])
    
    st.markdown("### ⚙️ 预警参数设置")
    alert_threshold = st.slider("LVP 触发阈值 (米)", min_value=400, max_value=4000, value=800, step=100)
    st.caption(f"当前 LVP 警报线：{alert_threshold} m")

# ==========================================
# 主界面逻辑
# ==========================================
st.title("机场短临能见度预测系统")
st.markdown("基于 **TCN-FECAM-Transformer** 模型，实现未来 12 小时能见度深度演变预测。")

if uploaded_file is None:
    st.info("👈 请在左侧上传包含连续 48 小时历史数据的 CSV 文件。")
else:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['date']).reset_index(drop=True)
    
    data_columns = df.columns.drop('date')
    data = df[data_columns]
    
    tab1, tab2, tab3 = st.tabs(["📊 12H预测结果", "📋 历史数据预览", "📖 系统说明"])
    
    with tab1:
        with st.spinner('正在执行 12 小时延展推理...'):
            # 1. 预处理
            df_stamp = df[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=1, freq='h') 

            scaler = MinMaxScaler()
            data_inverse = scaler.fit_transform(np.array(data))
            
            target_scaler = MinMaxScaler()
            target_scaler.fit(df[['visibility']])
            
            # 准备推理
            net, device, p_len, l_len = load_model()
            window = 48
            
            x_temp = np.array([data_inverse[-window:]])
            x_temp_mark = np.array([data_stamp[-window:]])
            
            # 构建 Decoder 输入 (引导序列 + 0占位)
            dec_input_known = data_inverse[-(l_len):]
            dec_input_placeholder = np.zeros((p_len, data_inverse.shape[-1]))
            y_temp = np.concatenate([dec_input_known, dec_input_placeholder], axis=0)
            y_temp = np.expand_dims(y_temp, axis=0)
            
            # 时间戳对齐
            y_temp_mark = data_stamp[-(l_len + p_len):]
            y_temp_mark = np.expand_dims(y_temp_mark, axis=0)
            
            # 2. 推理
            x_enc = torch.tensor(x_temp).type(torch.float32).to(device)
            x_mark_enc = torch.tensor(x_temp_mark).type(torch.float32).to(device)
            x_dec = torch.tensor(y_temp).type(torch.float32).to(device)
            x_mark_dec = torch.tensor(y_temp_mark).type(torch.float32).to(device)
            
            with torch.no_grad():
                outputs = net(x_enc, x_mark_enc, x_dec, x_mark_dec)
                outputs = outputs.detach().cpu().numpy()
                
                # 提取 visibility (最后一列)
                pred_vis_scaled = outputs[0, :, -1].reshape(-1, 1)
                final_preds = target_scaler.inverse_transform(pred_vis_scaled).flatten()
            
            # 3. 时间序列推算
            last_time = pd.to_datetime(df['date'].iloc[-1])
            predict_times = [last_time + datetime.timedelta(hours=i+1) for i in range(p_len)]
            current_visibility = df['visibility'].iloc[-1]

       # --- UI 渲染：12小时精美动态卡片 ---
        st.subheader(f"📅 未来 12 小时逐时预报监测面板")
        
        # 1. 注入自定义 CSS 样式（让卡片拥有阴影、圆角、悬浮动效和状态色）
        st.markdown("""
        <style>
        .metric-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #2ecc71; /* 默认安全绿色 */
            transition: all 0.3s ease;
            border-right: 1px solid #eee;
            border-top: 1px solid #eee;
            border-bottom: 1px solid #eee;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .metric-card.warning {
            border-left-color: #f1c40f; /* 临界警告黄色 */
            background-color: #fffdf5;
        }
        .metric-card.danger {
            border-left-color: #e74c3c; /* 危险红色 */
            background-color: #fff5f5;
        }
        .time-label {
            color: #7f8c8d;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .vis-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .delta-value {
            font-size: 13px;
            margin-top: 5px;
            font-weight: 500;
        }
        .delta-up { color: #2ecc71; }
        .delta-down { color: #e74c3c; }
        </style>
        """, unsafe_allow_html=True)

        # 2. 动态渲染 3 行 4 列的卡片网格
        for row in range(3):
            cols = st.columns(4)
            for col_idx in range(4):
                idx = row * 4 + col_idx
                if idx < p_len:
                    p_val = final_preds[idx]
                    p_time = predict_times[idx]
                    prev_val = current_visibility if idx == 0 else final_preds[idx-1]
                    delta = p_val - prev_val
                    
                    # 核心业务逻辑：根据 LVP 阈值动态决定卡片的颜色和图标
                    if p_val <= alert_threshold:
                        card_class = "metric-card danger"
                        icon = "🚨"
                    elif p_val <= alert_threshold + 300:
                        card_class = "metric-card warning"
                        icon = "⚠️"
                    else:
                        card_class = "metric-card"
                        icon = "✅"
                        
                    # 计算趋势箭头
                    delta_class = "delta-up" if delta >= 0 else "delta-down"
                    delta_arrow = "↗" if delta >= 0 else "↘"
                    
                    # 组装 HTML 注入 Streamlit
                    html_str = f"""
                    <div class="{card_class}">
                        <div class="time-label">{icon} T+{idx+1}h ({p_time.strftime('%H:%M')})</div>
                        <div class="vis-value">{p_val:.0f} m</div>
                        <div class="delta-value {delta_class}">
                            {delta_arrow} {abs(delta):.0f} m (较前时)
                        </div>
                    </div>
                    """
                    cols[col_idx].markdown(html_str, unsafe_allow_html=True)
        
        # --- UI 渲染：预警逻辑 ---
        min_pred = np.min(final_preds)
        min_idx = np.argmin(final_preds)
        
        if min_pred <= alert_threshold:
            st.error(f"🚨 **中长期风险警告**：预计在 **{predict_times[min_idx].strftime('%m-%d %H:%M')}** 能见度将跌至最低点 ({min_pred:.0f}m)。")
        else:
            st.success("✅ **气象条件良好**：未来 12 小时预测均处于安全阈值以上。")

        # --- 图表展示 ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.tail(48)['date'], y=df.tail(48)['visibility'],
            mode='lines', name='历史实况', line=dict(color='#2E86C1', width=2)
        ))
        
        combined_times = [pd.to_datetime(df['date'].iloc[-1])] + predict_times
        combined_vals = [current_visibility] + list(final_preds)
        
        fig.add_trace(go.Scatter(
            x=combined_times, y=combined_vals,
            mode='lines+markers', name='12H 预测趋势',
            line=dict(color='#E74C3C', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        fig.update_layout(
            title=dict(
                text="✈️ 机场能见度历史实况与未来 12 小时演变趋势图", # 这里加上你的图表名字
                font=dict(size=18),
                x=0.5, # 0.5 表示居中，0 表示靠左
                xanchor='center'
            ),
            height=500, 
            hovermode='x unified', 
            xaxis=dict(title="时间"), 
            yaxis=dict(title="能见度 (m)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(df.tail(48), use_container_width=True)

    with tab3:
        st.markdown(f"### 系统运行参数\n- **预测跨度**: {p_len} 小时\n- **输入窗口**: {window} 小时\n- **模型状态**: 已加载权重 `best_model.pth`")