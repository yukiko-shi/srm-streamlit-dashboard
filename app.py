# -*- coding: utf-8 -*-
"""
预制菜企业 · SRM 决策仪表盘（单文件MVP）

功能分区（通过 tabs 实现“多页面/分区布局”）：
1) 企业自评输入
2) SRM 引入必要性评估
3) 模块优先级推荐
4) 成本收益分析（含情景）
5) 综合报告输出（Markdown 下载）

依赖：streamlit, pandas, numpy, plotly
安装：
  pip install --user streamlit pandas numpy plotly
运行：
  python -m streamlit run app.py

备注：
- 本MVP以“规则/权重 + 少量参数”驱动；后续可替换为AHP/Delphi等权重方法。
- 数据口径均为“年度”便于演示；实施一次性费用以“摊销年限”折算为年化成本。
"""

from __future__ import annotations
import io
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.io as pio
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_cjk_font():
    """
    尝试注册中文字体；将 NotoSansSC-Regular.ttf 放到 ./fonts 下。
    若失败则回退英文字体。
    """
    try:
        font_path = os.path.join("fonts", "NotoSansSC-Regular.ttf")
        pdfmetrics.registerFont(TTFont("NotoSansSC", font_path))
        return "NotoSansSC"
    except Exception:
        return "Helvetica"

def to_wan(x: float) -> float:
    return float(x) / 10000.0


# =========================
# 0) 全局配置与默认参数
# =========================
st.set_page_config(
    page_title="预制菜 · SRM 决策仪表盘",
    page_icon="📈",
    layout="wide",
)

# -- 维度权重（可在页面调整）
DEFAULT_DIM_WEIGHTS = {
    "成本效益": 0.35,
    "供应链复杂度": 0.25,
    "风险/质量": 0.25,
    "数字化成熟度": 0.15,
}

# -- SRM 模块清单（可按需增删）
MODULES = [
    "供应商准入与档案",
    "在线询价/比价/招投标",
    "采购订单与交付跟踪",
    "对账结算",
    "供应商绩效评价",
    "风险预警",
]

# -- 情景默认参数（采购降价%、人力节省%、损失减少%）
SCENARIOS = {
    "保守": {"price_drop": 0.03, "hr_save": 0.10, "loss_drop": 0.30},
    "中性": {"price_drop": 0.06, "hr_save": 0.20, "loss_drop": 0.60},
    "乐观": {"price_drop": 0.10, "hr_save": 0.30, "loss_drop": 0.75},
}

# ==============
# 1) 小工具函数
# ==============

def _clamp(v: float, lo: float = 0, hi: float = 100) -> float:
    return float(np.clip(v, lo, hi))

@st.cache_data
def _radar_fig(scores: dict[str, float]):
    cats = list(scores.keys())
    vals = list(scores.values())
    # 闭合雷达图
    cats += cats[:1]
    vals += vals[:1]
    fig = go.Figure(
        data=[go.Scatterpolar(r=vals, theta=cats, fill='toself', name='得分')],
        layout=go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
        ),
    )
    return fig

@st.cache_data
def _priority_bar(df: pd.DataFrame):
    fig = px.bar(
        df.sort_values("优先指数", ascending=True),
        x="优先指数",
        y="模块",
        orientation="h",
        text="优先指数",
        height=380,
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=20))
    return fig

@st.cache_data
def _value_difficulty_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="难度(0-100)",
        y="价值(0-100)",
        text="模块",
        size=np.maximum(5, df["优先指数"] - df["优先指数"].min() + 5),
        height=420,
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=20))
    return fig

@st.cache_data
def _group_bar(df_melt: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str = ""):
    fig = px.bar(df_melt, x=x_col, y=y_col, color=color_col, barmode="group", height=420, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=20))
    return fig

# =====================
# 2) 侧边栏：全局设置
# =====================
st.sidebar.title("⚙️ 全局设置")
with st.sidebar.expander("维度权重（用于必要性评分）", expanded=False):
    dim_weights = {}
    total_w = 0.0
    for k, v in DEFAULT_DIM_WEIGHTS.items():
        dim_weights[k] = st.slider(f"{k} 权重", 0.0, 1.0, float(v), 0.05)
        total_w += dim_weights[k]
    if abs(total_w - 1.0) > 1e-6:
        st.warning("权重和不为1，系统已自动归一化。")
        for k in dim_weights:
            dim_weights[k] /= total_w if total_w > 0 else 1.0

with st.sidebar.expander("情景参数（可改）", expanded=False):
    scenario_name = st.selectbox("选择情景", list(SCENARIOS.keys()), index=1)
    scen = SCENARIOS[scenario_name].copy()
    # 允许用户微调
    scen["price_drop"] = st.slider("采购降价比例", 0.0, 0.30, float(scen["price_drop"]), 0.01)
    scen["hr_save"] = st.slider("人力节省比例", 0.0, 0.50, float(scen["hr_save"]), 0.01)
    scen["loss_drop"] = st.slider("供应问题损失减少比例", 0.0, 1.0, float(scen["loss_drop"]), 0.05)

st.sidebar.write("——")
st.sidebar.caption("提示：页面内输入发生变化会实时刷新图表和结论。")

# =====================
# 3) 主体：分区/多标签页
# =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 企业自评输入",
    "🧭 SRM引入必要性评估",
    "🧩 模块优先级推荐",
    "💰 成本收益分析",
    "📄 综合报告输出",
])

# ----------------------
# Tab 1) 企业自评输入
# ----------------------
with tab1:
    st.header("🏁 企业自评输入")
    st.caption("填写企业现状与财务基线，驱动后续评估与可视化。")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("业务与供应链现状")
        suppliers = st.number_input("供应商数量（家）", min_value=0, max_value=10000, value=120, step=1)
        sku = st.number_input("采购SKU种类（个）", min_value=0, max_value=50000, value=300, step=1)
        raw_cost_rate = st.slider("原料成本率（%）", 0, 100, 70)
        delay_events = st.slider("年内供应延误次数（次）", 0, 200, 12)
    with c2:
        st.subheader("质量与数字化基础")
        reject_rate = st.slider("来料不合格率（%）", 0, 50, 3)
        it_readiness = st.slider("数字化成熟度（1-5）", 1, 5, 3)
        mgmt_support = st.slider("管理层支持度（1-5）", 1, 5, 4)
        top5_conc = st.slider("前五大供应商集中度（%）", 0, 100, 60)

    st.markdown("---")
    st.subheader("财务基线（年度，万元）")
    c3, c4, c5 = st.columns(3)
    with c3:
        base_purchase = st.number_input("年原料采购额", 0, 100000000, 100000, step=1000)
        annual_loss = st.number_input("供应问题损失（停工/不合格等）", 0, 1000000, 200)
    with c4:
        procurement_hr_cost = st.number_input("采购及相关人力成本", 0, 1000000, 500)
        srm_annual_cost = st.number_input("SRM年化成本（订阅/运维）", 0, 1000000, 300)
    with c5:
        implement_cost = st.number_input("一次性实施费用", 0, 10000000, 800)
        amort_years = st.number_input("摊销年限（年）", 1, 10, 3)

    st.info(
        "提示：原料成本率越高、供应复杂度越高，SRM的降本空间和治理价值通常越大。"
    )

# ---------------------------------
# 维度打分（供 Tab2/报告复用）
# ---------------------------------

def score_cost_benefit(raw_cost_rate_pct: int) -> float:
    # 原料成本率越高 → 降本空间越大 → 得分越高（0-100）
    return _clamp(raw_cost_rate_pct)

def score_complexity(num_suppliers: int, num_sku: int) -> float:
    s = _clamp((num_suppliers / 500) * 100)
    k = _clamp((num_sku / 1000) * 100)
    return _clamp(0.6 * s + 0.4 * k)

def score_risk_quality(delay_events: int, reject_rate_pct: int) -> float:
    d = _clamp((delay_events / 30) * 100)        # 30次≈100分
    r = _clamp(reject_rate_pct * 5)              # 20%≈100分
    return _clamp(0.6 * d + 0.4 * r)

def score_it_maturity(it_lv: int, mgmt_lv: int) -> float:
    return _clamp((it_lv + mgmt_lv) / 10 * 100)

# 计算维度分与总分
DIM_SCORES = {
    "成本效益": score_cost_benefit(raw_cost_rate),
    "供应链复杂度": score_complexity(suppliers, sku),
    "风险/质量": score_risk_quality(delay_events, reject_rate),
    "数字化成熟度": score_it_maturity(it_readiness, mgmt_support),
}
TOTAL_SCORE = float(sum(DIM_SCORES[k] * dim_weights.get(k, 0) for k in DIM_SCORES))

# --------------------------------------
# Tab 2) SRM 引入必要性评估
# --------------------------------------
with tab2:
    st.header("🧭 SRM 引入必要性评估")
    st.caption("基于多维度加权得分（MVP版：规则+权重）")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("综合评分")
        st.metric("SRM 引入必要性（0-100）", f"{TOTAL_SCORE:.1f}")
        if TOTAL_SCORE >= 60:
            st.success("建议：**尽快引入SRM（优先推进）**")
        elif TOTAL_SCORE >= 45:
            st.warning("建议：**择机引入（先试点/小范围）**")
        else:
            st.info("建议：**暂缓，引入前先补齐数据/流程基础**")
    with c2:
        st.subheader("维度雷达图")
        st.plotly_chart(_radar_fig(DIM_SCORES), use_container_width=True)

    st.markdown("**维度明细**")
    detail_df = pd.DataFrame({
        "维度": list(DIM_SCORES.keys()),
        "得分": [round(v, 1) for v in DIM_SCORES.values()],
        "权重": [round(dim_weights[k], 2) for k in DIM_SCORES.keys()],
    })
    st.dataframe(detail_df, use_container_width=True)

# --------------------------------------
# Tab 3) 模块优先级推荐
# --------------------------------------
with tab3:
    st.header("🧩 模块优先级推荐")
    st.caption("半定量：价值 × 难度 → 优先指数（可叠加人工偏好）")

    st.subheader("价值/难度权重调节")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        value_weight = st.slider("价值权重", 0.0, 2.0, 1.0, 0.1)
    with c2:
        diff_weight = st.slider("难度权重", 0.0, 2.0, 0.5, 0.1)
    with c3:
        bias_bonus = st.selectbox("优先照顾的模块（可选）", ["(无)"] + MODULES)

    # 基于企业现状生成“价值/难度”的初始评分（0-100）
    value_map = {
        "供应商准入与档案": 70,
        "在线询价/比价/招投标": 90 if raw_cost_rate >= 50 else 75,
        "采购订单与交付跟踪": 85 if delay_events >= 5 else 70,
        "对账结算": 60,
        "供应商绩效评价": 65,
        "风险预警": 70 if (delay_events >= 10 or reject_rate >= 5 or top5_conc >= 70) else 55,
    }
    difficulty_map = {
        "供应商准入与档案": 35,
        "在线询价/比价/招投标": 45,
        "采购订单与交付跟踪": 55,
        "对账结算": 30,
        "供应商绩效评价": 60,
        "风险预警": 65,
    }

    rows = []
    for m in MODULES:
        val = value_map[m]
        diff = difficulty_map[m]
        prio = value_weight * val - diff_weight * diff
        if m == bias_bonus:
            prio += 5  # 轻微人工干预
        rows.append((m, val, diff, round(prio, 1)))

    prio_df = pd.DataFrame(rows, columns=["模块", "价值(0-100)", "难度(0-100)", "优先指数"])

    c4, c5 = st.columns([1.1, 0.9])
    with c4:
        st.subheader("优先级排序（条形图）")
        st.plotly_chart(_priority_bar(prio_df), use_container_width=True)
    with c5:
        st.subheader("价值-难度矩阵")
        st.plotly_chart(_value_difficulty_scatter(prio_df), use_container_width=True)

    st.markdown("**明细表**")
    st.dataframe(prio_df.sort_values("优先指数", ascending=False), use_container_width=True)

# --------------------------------------
# Tab 4) 成本收益分析（情景）
# --------------------------------------
with tab4:
    st.header("💰 成本收益分析（年度口径）")
    st.caption("情景＝关键假设的不同档位。一次性实施费按摊销年限折算为年化成本。")

    # 计算“引入前/后”
    price_drop = scen["price_drop"]
    hr_save = scen["hr_save"]
    loss_drop = scen["loss_drop"]

    after_purchase = base_purchase * (1 - price_drop)
    after_hr = procurement_hr_cost * (1 - hr_save)
    after_loss = annual_loss * (1 - loss_drop)

    before_df = pd.DataFrame({
        "科目": ["原料采购额", "采购人力成本", "供应问题损失"],
        "金额": [base_purchase, procurement_hr_cost, annual_loss],
        "状态": ["引入前"] * 3,
    })
    after_df = pd.DataFrame({
        "科目": ["原料采购额", "采购人力成本", "供应问题损失"],
        "金额": [after_purchase, after_hr, after_loss],
        "状态": ["引入后"] * 3,
    })
    res_df = pd.concat([before_df, after_df], ignore_index=True)

    st.subheader(f"情景：{scenario_name}")
    st.plotly_chart(_group_bar(res_df, "科目", "金额", "状态", title="引入前后对比"), use_container_width=True)

    # 年度收益与ROI
    annual_benefit = (base_purchase - after_purchase) + (procurement_hr_cost - after_hr) + (annual_loss - after_loss)
    annualized_cost = srm_annual_cost + (implement_cost / max(1, amort_years))
    net_benefit = annual_benefit - annualized_cost
    roi = (net_benefit / annualized_cost) if annualized_cost > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年度综合收益（万元）", f"{annual_benefit:,.0f}")
    c2.metric("年化成本（万元）", f"{annualized_cost:,.0f}")
    c3.metric("净收益（万元）", f"{net_benefit:,.0f}")
    c4.metric("ROI", f"{roi*100:,.1f}%" if np.isfinite(roi) else "-")

    # 简单灵敏度：对“采购降价%”±2个百分点
    st.subheader("灵敏度：采购降价% 对 ROI 的影响")
    deltas = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    records = []
    for d in deltas:
        pd_pct = max(0.0, price_drop + d)
        ap = base_purchase * (1 - pd_pct)
        ab = (base_purchase - ap) + (procurement_hr_cost - after_hr) + (annual_loss - after_loss)
        r = (ab - annualized_cost) / annualized_cost if annualized_cost > 0 else np.nan
        records.append({"采购降价%": round(pd_pct * 100, 1), "ROI%": r * 100})
    sens_df = pd.DataFrame(records)
    fig = px.line(sens_df, x="采购降价%", y="ROI%", markers=True, height=380)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# Tab 5) 综合报告输出（Markdown下载）
# --------------------------------------
with tab5:
    st.header("📄 综合报告输出")
    st.caption("自动汇总关键输入与结论，生成 Markdown 报告用于评审/汇报。")

    prio_sorted = prio_df.sort_values("优先指数", ascending=False)
    top3 = ", ".join(prio_sorted.head(3)["模块"].tolist())

    md = io.StringIO()
    print(f"# 预制菜 · SRM 决策评估报告\n", file=md)
    print(f"**日期**：{date.today().isoformat()}\n", file=md)
    print("## 一、企业关键输入\n", file=md)
    print(f"- 供应商数：{suppliers} 家；SKU：{sku}；原料成本率：{raw_cost_rate}%\n", file=md)
    print(f"- 延误次数：{delay_events}；来料不合格率：{reject_rate}%\n", file=md)
    print(f"- 数字化成熟度：{it_readiness}/5；管理支持：{mgmt_support}/5\n", file=md)
    print(f"- 年采购额：{base_purchase} 万；采购人力：{procurement_hr_cost} 万；供应损失：{annual_loss} 万\n", file=md)
    print(f"- SRM 年化成本：{srm_annual_cost} 万；实施费：{implement_cost} 万（摊 {amort_years} 年）\n", file=md)

    print("## 二、SRM 引入必要性\n", file=md)
    print(f"- 综合评分（0-100）：**{TOTAL_SCORE:.1f}**\n", file=md)
    for k in DIM_SCORES:
        print(f"  - {k}：得分 {DIM_SCORES[k]:.1f}，权重 {dim_weights[k]:.2f}", file=md)
    print("\n", file=md)

    print("## 三、模块优先级（Top 3）\n", file=md)
    print(f"- 推荐优先实施：**{top3}**\n", file=md)

    print("## 四、成本-收益（年度情景）\n", file=md)
    print(f"- 情景：{scenario_name}；采购降价：{scen['price_drop']*100:.1f}%；人力节省：{scen['hr_save']*100:.1f}%；损失减少：{scen['loss_drop']*100:.1f}%\n", file=md)
    print(f"- 年度综合收益：{annual_benefit:,.0f} 万；年化成本：{annualized_cost:,.0f} 万；净收益：{net_benefit:,.0f} 万；ROI：{roi*100:,.1f}%\n", file=md)

    radar_fig_obj = _radar_fig(DIM_SCORES)
prio_bar_fig_obj = _priority_bar(prio_df)
val_diff_fig_obj = _value_difficulty_scatter(prio_df)
res_fig_obj = _group_bar(res_df, "科目", "金额", "状态", title=f"情景：{scenario_name} 引入前后对比")

# 复算 ROI 灵敏度图（与 Tab4 一致，但变量独立）
_deltas = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
_records = []
for d in _deltas:
    _pd_pct = max(0.0, scen["price_drop"] + d)
    _ap = base_purchase * (1 - _pd_pct)
    _ab = (base_purchase - _ap) + (procurement_hr_cost - after_hr) + (annual_loss - after_loss)
    _r = (_ab - annualized_cost) / annualized_cost if annualized_cost > 0 else np.nan
    _records.append({"采购降价%": round(_pd_pct * 100, 1), "ROI%": _r * 100})
_sens_df = pd.DataFrame(_records)
sens_fig_obj = px.line(_sens_df, x="采购降价%", y="ROI%", markers=True, height=380)
sens_fig_obj.update_layout(margin=dict(l=10, r=10, t=30, b=20))

# —— Figure → PNG（kaleido） ——
def fig_to_rlimage(fig, width_cm=16):
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)  # 需要 kaleido
    except Exception as e:
        st.error(f"图表导出失败，请先安装 kaleido：`pip install --user kaleido`。错误：{e}")
        raise
    return RLImage(BytesIO(png_bytes), width=width_cm*cm)

radar_img = fig_to_rlimage(radar_fig_obj, 16)
prio_bar_img = fig_to_rlimage(prio_bar_fig_obj, 16)
val_diff_img = fig_to_rlimage(val_diff_fig_obj, 16)
res_img = fig_to_rlimage(res_fig_obj, 16)
sens_img = fig_to_rlimage(sens_fig_obj, 16)

# —— 构建 PDF（内存） ——
buffer = BytesIO()
doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
styles = getSampleStyleSheet()

# 应用中文字体（若可用）
_font_name = register_cjk_font()
styles["Normal"].fontName = _font_name
styles["Heading1"].fontName = _font_name
styles["Heading2"].fontName = _font_name
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=10, leading=14))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=6))

story = []

# Markdown 文本粗粒度渲染
for line in md.getvalue().split("\n"):
    if line.startswith("# "):
        story.append(Paragraph(line[2:], styles["H1"]))
    elif line.startswith("## "):
        story.append(Paragraph(line[3:], styles["H2"]))
    elif line.strip() == "":
        story.append(Spacer(1, 0.2*cm))
    else:
        story.append(Paragraph(line, styles["Small"]))

story.append(Spacer(1, 0.4*cm))
story.append(PageBreak())

# 插图页
story.append(Paragraph("附录一：评估可视化", styles["H1"]))
for title, img in [
    ("维度雷达图", radar_img),
    ("模块优先级（条形图）", prio_bar_img),
    ("价值-难度矩阵", val_diff_img),
    ("引入前后对比", res_img),
    ("ROI 灵敏度分析", sens_img),
]:
    story.append(Paragraph(title, styles["H2"]))
    story.append(img)
    story.append(Spacer(1, 0.5*cm))

doc.build(story)
pdf_bytes = buffer.getvalue()
st.download_button(
    "⬇️ 下载 PDF 报告（含图表）",
    data=pdf_bytes,
    file_name=f"SRM_评估报告_{date.today().isoformat()}.pdf",
    mime="application/pdf",
)
st.caption("已嵌入：雷达图、模块优先级、价值-难度、前后对比、ROI灵敏度。")
