import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer segmentation dashboard",
    page_icon="🧺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading & processing (cached so clustering only runs once per config)
# ---------------------------------------------------------------------------

CLUSTER_NAMES = ["Premium Patrons", "Steady Spenders", "Occasional Shoppers", "Budget Households"]
CLUSTER_COLORS = ["#8B5E3C", "#C77B4D", "#5C8374", "#B23A48"]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Income"])
    df = df[df["Income"] < 200000]
    df = df[df["Year_Birth"] > 1940]

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    df["Age"] = 2014 - df["Year_Birth"]
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spent"] = df[spend_cols].sum(axis=1)
    df["Family_Size"] = df["Marital_Status"].map(lambda x: 2 if x in ["Married", "Together"] else 1) \
        + df["Kidhome"] + df["Teenhome"]
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    ref_date = df["Dt_Customer"].max()
    df["Customer_For"] = (ref_date - df["Dt_Customer"]).dt.days
    df["Education"] = df["Education"].replace({
        "2n Cycle": "Master", "Basic": "Undergraduate", "Graduation": "Graduate",
        "Master": "Master", "PhD": "PhD"
    })
    df["Marital_Status"] = df["Marital_Status"].replace({
        "Married": "Partnered", "Together": "Partnered",
        "Single": "Single", "Divorced": "Single", "Widow": "Single",
        "Alone": "Single", "Absurd": "Single", "YOLO": "Single"
    })
    channel_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df[channel_cols].sum(axis=1)
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    df["Total_Campaigns_Accepted"] = df[campaign_cols].sum(axis=1)
    return df


@st.cache_data
def cluster_data(df: pd.DataFrame, n_clusters: int, seed: int):
    features = ["Age", "Income", "Total_Spent", "Family_Size", "Customer_For",
                "Total_Purchases", "NumWebVisitsMonth", "Recency"]
    X = df[features].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=seed)
    Xp = pca.fit_transform(Xs)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)

    out = df.copy()
    out["Cluster"] = labels
    out["PC1"] = Xp[:, 0]
    out["PC2"] = Xp[:, 1]

    # order clusters by average spend, descending, and assign friendly names/colors
    order = out.groupby("Cluster")["Total_Spent"].mean().sort_values(ascending=False).index.tolist()
    name_map = {c: CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"Segment {i+1}" for i, c in enumerate(order)}
    color_map = {c: CLUSTER_COLORS[i] if i < len(CLUSTER_COLORS) else "#888888" for i, c in enumerate(order)}
    out["Segment"] = out["Cluster"].map(name_map)
    out["SegmentColor"] = out["Cluster"].map(color_map)
    return out, order, name_map, color_map


def segment_description(row):
    income_lvl = "high" if row["income_norm"] >= 60 else ("moderate" if row["income_norm"] >= 35 else "modest")
    spend_lvl = "big spenders" if row["spend_norm"] >= 60 else ("moderate spenders" if row["spend_norm"] >= 35 else "light spenders")
    fam = "larger households" if row["Family_Size"] >= 2.6 else "smaller households"
    return f"{income_lvl.capitalize()}-income, {fam} who are {spend_lvl}."


def segment_strategy(row, avg_accept_rate):
    bits = []
    if row["income_norm"] >= 60 and row["deal_share"] <= 0.18:
        bits.append("lead with premium and exclusive product offers rather than discounts")
    if row["deal_share"] > 0.18:
        bits.append("promote deals, bundles, and loyalty rewards, since this group responds well to discounts")
    if row["Family_Size"] >= 2.6:
        bits.append("highlight family packs and value bundles")
    if row["campaign_accept_rate"] < avg_accept_rate:
        bits.append("re-engage with more targeted, personalized campaigns given below-average response rates")
    else:
        bits.append("keep investing here, as this segment already responds well to campaigns")
    return "; ".join(bits[:3]).capitalize() + "."


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Controls")
csv_path = "marketing_campaign.csv"
n_clusters = st.sidebar.slider("Number of segments (k)", min_value=2, max_value=6, value=4)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

try:
    raw = load_data(csv_path)
except FileNotFoundError:
    st.error(f"Couldn't find `{csv_path}`. Place the CSV next to app.py, or update the path in the sidebar.")
    st.stop()

df, cluster_order, name_map, color_map = cluster_data(raw, n_clusters, seed)

segment_filter = st.sidebar.multiselect(
    "Filter segments",
    options=[name_map[c] for c in cluster_order],
    default=[name_map[c] for c in cluster_order],
)
df_view = df[df["Segment"].isin(segment_filter)]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧺 Customer segmentation dashboard")
st.caption(
    "Behavioral segments derived from income, spending, tenure, and engagement data — "
    "K-means clustering on standardized features."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
overall_accept_rate = 100 * df["Total_Campaigns_Accepted"].gt(0).mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers", f"{len(df_view):,}")
k2.metric("Avg. income", f"${df_view['Income'].mean():,.0f}")
k3.metric("Avg. spend", f"${df_view['Total_Spent'].mean():,.0f}")
k4.metric("Total revenue", f"${df_view['Total_Spent'].sum():,.0f}")
k5.metric("Campaign acceptance", f"{100*df_view['Total_Campaigns_Accepted'].gt(0).mean():.1f}%")
k6.metric("Avg. recency", f"{df_view['Recency'].mean():.0f} days")

st.divider()

# ---------------------------------------------------------------------------
# Segment summary table + selector
# ---------------------------------------------------------------------------

st.subheader("Segments")

summary_rows = []
spend_cols = {"Wine": "MntWines", "Fruits": "MntFruits", "Meat": "MntMeatProducts",
              "Fish": "MntFishProducts", "Sweets": "MntSweetProducts", "Gold": "MntGoldProds"}
channel_cols_map = {"Deals": "NumDealsPurchases", "Web": "NumWebPurchases",
                     "Catalog": "NumCatalogPurchases", "Store": "NumStorePurchases"}

income_min, income_max = df.groupby("Cluster")["Income"].mean().min(), df.groupby("Cluster")["Income"].mean().max()
spend_min, spend_max = df.groupby("Cluster")["Total_Spent"].mean().min(), df.groupby("Cluster")["Total_Spent"].mean().max()

for c in cluster_order:
    sub = df[df["Cluster"] == c]
    total_ch = sub[list(channel_cols_map.values())].sum().sum()
    deal_share = sub["NumDealsPurchases"].sum() / total_ch if total_ch else 0
    income_mean = sub["Income"].mean()
    spend_mean = sub["Total_Spent"].mean()
    income_norm = 100 * (income_mean - income_min) / (income_max - income_min) if income_max > income_min else 50
    spend_norm = 100 * (spend_mean - spend_min) / (spend_max - spend_min) if spend_max > spend_min else 50
    row = {
        "Cluster": c,
        "Segment": name_map[c],
        "Color": color_map[c],
        "Count": len(sub),
        "Pct": round(100 * len(sub) / len(df), 1),
        "Income": income_mean,
        "Total_Spent": spend_mean,
        "Family_Size": sub["Family_Size"].mean(),
        "Age": sub["Age"].mean(),
        "Recency": sub["Recency"].mean(),
        "Web_Visits": sub["NumWebVisitsMonth"].mean(),
        "campaign_accept_rate": 100 * sub["Total_Campaigns_Accepted"].gt(0).mean(),
        "Complain_rate": 100 * sub["Complain"].mean(),
        "Revenue_share": 100 * sub["Total_Spent"].sum() / df["Total_Spent"].sum(),
        "deal_share": deal_share,
        "income_norm": income_norm,
        "spend_norm": spend_norm,
    }
    row["description"] = segment_description(row)
    row["strategy"] = segment_strategy(row, overall_accept_rate)
    for label, col in spend_cols.items():
        row[f"spend_{label}"] = sub[col].mean()
    for label, col in channel_cols_map.items():
        row[f"channel_{label}"] = sub[col].mean()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

cols = st.columns(len(cluster_order))
for i, c in enumerate(cluster_order):
    row = summary_df[summary_df["Cluster"] == c].iloc[0]
    with cols[i]:
        st.markdown(
            f"""
            <div style="border-left:6px solid {row['Color']}; padding:10px 14px; border-radius:6px; background:rgba(128,128,128,0.06);">
              <div style="font-size:16px; font-weight:600;">{row['Segment']}</div>
              <div style="font-size:12px; color:gray; min-height:32px;">{row['description']}</div>
              <div style="font-size:13px; margin-top:6px;">
                <b>{row['Pct']}%</b> of customers<br>
                <b>${row['Total_Spent']:,.0f}</b> avg. spend<br>
                <b>{row['Revenue_share']:.1f}%</b> of revenue
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

selected_segment = st.selectbox("Select a segment for a detailed profile", options=[name_map[c] for c in cluster_order])
sel_row = summary_df[summary_df["Segment"] == selected_segment].iloc[0]

p1, p2 = st.columns([1, 1.4])
with p1:
    st.markdown(f"### {sel_row['Segment']}")
    st.write(sel_row["description"])
    st.markdown("**Recommended approach**")
    st.write(sel_row["strategy"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg. income", f"${sel_row['Income']:,.0f}")
    m2.metric("Avg. age", f"{sel_row['Age']:.1f} yrs")
    m3.metric("Family size", f"{sel_row['Family_Size']:.2f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Web visits/mo", f"{sel_row['Web_Visits']:.1f}")
    m5.metric("Recency", f"{sel_row['Recency']:.0f} days")
    m6.metric("Complaint rate", f"{sel_row['Complain_rate']:.2f}%")

with p2:
    radar_features = ["Income", "spend_norm", "Family_Size", "Web_Visits", "campaign_accept_rate"]
    radar_labels = ["Income", "Spending", "Family size", "Web engagement", "Campaign response"]
    mins = summary_df[radar_features].min()
    maxs = summary_df[radar_features].max()
    norm_vals = [
        100 * (sel_row[f] - mins[f]) / (maxs[f] - mins[f]) if maxs[f] > mins[f] else 50
        for f in radar_features
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        line_color=sel_row["Color"],
        name=sel_row["Segment"],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Segment size + revenue share
# ---------------------------------------------------------------------------

c1, c2 = st.columns(2)

with c1:
    st.subheader("Segment size")
    fig = px.pie(
        summary_df, names="Segment", values="Count", color="Segment",
        color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
        hole=0.55,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Revenue share")
    fig = px.bar(
        summary_df.sort_values("Revenue_share"), x="Revenue_share", y="Segment", orientation="h",
        color="Segment", color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
        text=summary_df.sort_values("Revenue_share")["Revenue_share"].round(1).astype(str) + "%",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, showlegend=False, xaxis_title="% of total revenue", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Income vs spend scatter
# ---------------------------------------------------------------------------

st.subheader("Income vs. total spend")
sample = df_view.sample(min(1500, len(df_view)), random_state=1) if len(df_view) > 0 else df_view
fig = px.scatter(
    sample, x="Income", y="Total_Spent", color="Segment",
    color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
    opacity=0.65, hover_data=["Age", "Family_Size"],
)
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Spend by category + channel mix
# ---------------------------------------------------------------------------

c3, c4 = st.columns(2)

with c3:
    st.subheader("Average spend by category")
    cat_long = summary_df.melt(
        id_vars=["Segment", "Color"],
        value_vars=[f"spend_{k}" for k in spend_cols],
        var_name="Category", value_name="Amount"
    )
    cat_long["Category"] = cat_long["Category"].str.replace("spend_", "")
    fig = px.bar(
        cat_long, x="Category", y="Amount", color="Segment", barmode="group",
        color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Purchase channel mix")
    ch_long = summary_df.melt(
        id_vars=["Segment", "Color"],
        value_vars=[f"channel_{k}" for k in channel_cols_map],
        var_name="Channel", value_name="Avg_purchases"
    )
    ch_long["Channel"] = ch_long["Channel"].str.replace("channel_", "")
    totals = ch_long.groupby("Segment")["Avg_purchases"].transform("sum")
    ch_long["Share"] = 100 * ch_long["Avg_purchases"] / totals
    fig = px.bar(
        ch_long, x="Share", y="Segment", color="Channel", orientation="h",
        color_discrete_sequence=["#D8B25C", "#5C8374", "#C77B4D", "#8B5E3C"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend_title_text="", xaxis_title="% of purchases", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Campaign acceptance + age distribution
# ---------------------------------------------------------------------------

c5, c6 = st.columns(2)

with c5:
    st.subheader("Campaign acceptance rate")
    camp_names = ["Campaign 1", "Campaign 2", "Campaign 3", "Campaign 4", "Campaign 5", "Last campaign"]
    camp_rows = []
    for c in cluster_order:
        sub = df[df["Cluster"] == c]
        for name, col in zip(camp_names, campaign_cols):
            camp_rows.append({"Segment": name_map[c], "Color": color_map[c], "Campaign": name, "Rate": 100 * sub[col].mean()})
    camp_df = pd.DataFrame(camp_rows)
    fig = px.bar(
        camp_df, x="Campaign", y="Rate", color="Segment", barmode="group",
        color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend_title_text="", yaxis_title="% accepted")
    st.plotly_chart(fig, use_container_width=True)

with c6:
    st.subheader("Age distribution")
    fig = px.histogram(
        df_view, x="Age", color="Segment", nbins=20, barmode="overlay", opacity=0.55,
        color_discrete_map={row["Segment"]: row["Color"] for _, row in summary_df.iterrows()},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("View raw clustered data"):
    st.dataframe(df_view)
