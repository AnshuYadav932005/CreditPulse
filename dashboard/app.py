"""FinSight decision dashboard.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from finsight.config import DATA_PROCESSED, DUCKDB_PATH  # noqa: E402
from finsight.features.build import load_raw  # noqa: E402
from finsight.models.common import load_json, load_model  # noqa: E402
from finsight.models.credit_scoring import CreditScorer, applicant_features  # noqa: E402
from finsight.pipeline import LIVE_ASOF  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow) and fixed status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
MUTED = "#8a8984"

st.set_page_config(page_title="CreditPulse", page_icon="🏦", layout="wide")


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified",
                      font=dict(size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


@st.cache_resource
def db():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return db().execute(sql).df()


@st.cache_resource
def scoring_context():
    return (load_raw(), pd.read_parquet(DATA_PROCESSED / "monthly_panel.parquet"),
            pd.read_parquet(DATA_PROCESSED / "account_profile.parquet"),
            CreditScorer(load_model("credit_model.joblib")))


if not DUCKDB_PATH.exists():
    st.error("No warehouse found. Run `python -m finsight.pipeline` first.")
    st.stop()

S = load_json("summary.json")
CR, EW, NB, SG = (load_json(f) for f in
                  ("credit_report.json", "ews_report.json", "nbo_report.json", "segmentation_report.json"))

st.title("CreditPulse · Retail Banking Decision Intelligence")
st.caption(f"Berka (PKDD'99) Czech bank · {S['customers']:,} customers · 1.06M transactions · "
           f"decisions as of {LIVE_ASOF}")

tab_over, tab_credit, tab_ews, tab_seg = st.tabs(
    ["Overview", "Alternative credit scoring", "Early warning", "Segments & next-best-offer"])

# ------------------------------------------------------------------------------ overview
with tab_over:
    c = S["credit_scoring"]; e = S["early_warning"]; n = S["next_best_offer"]
    st.subheader("Results vs. the brief")
    k1, k2, k3 = st.columns(3)
    k1.metric("NPA reduction at 85% approval", f"{c['npa_reduction_at_85pct_approval']:.0%}",
              "target 15%", delta_color="off", delta_arrow="off")
    k1.caption(f"Behavioural model AUC {c['auc']:.3f} vs traditional {c['auc_traditional']:.3f}; "
               f"{c['straight_through_rate']:.0%} of applications auto-decided.")
    k2.metric("Red-alert precision", f"{e['red_precision']:.0%}", "target 80%+", delta_color="off", delta_arrow="off")
    k2.caption(f"Median {e['red_median_lead_days']:.0f} days before delinquency; "
               f"amber tier catches {e['amber_recall']:.0%} of delinquent borrowers.")
    k3.metric("Cross-sell conversion uplift", f"+{n['campaign_uplift_vs_heuristic']:.0%}",
              "target +20%", delta_color="off", delta_arrow="off")
    k3.caption(f"vs. targeting highest-balance customers; {n['campaign_uplift_vs_random'] + 1:.1f}x random.")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("**How the three engines connect**")
        st.markdown(
            "1. **Credit scoring** prices every application from transaction behaviour, "
            "no bureau needed.\n"
            "2. Its origination PD feeds the **early-warning** model, which re-scores every "
            "running loan monthly.\n"
            "3. **Next-best-offer** ranks products by propensity × margin. Credit offers are "
            "only made to customers who pass credit pre-approval and are not on the watchlist.")
    with b:
        st.markdown("**Live book**")
        live = q("SELECT preapproval_decision AS decision, count(*) AS customers FROM customer_360 GROUP BY 1")
        st.dataframe(live, hide_index=True, use_container_width=True)
        st.write(f"EWS watchlist: **{e['live_red']} red**, **{e['live_amber']} amber** "
                 f"of {len(q('SELECT 1 FROM ews_watchlist'))} running loans.")

# ------------------------------------------------------------------------------ credit
with tab_credit:
    champ = CR["champion"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC (5×5 CV)", f"{CR['metrics'][champ]['auc']:.3f}")
    m2.metric("AUC out-of-time", f"{CR['out_of_time'][champ]['auc']:.3f}")
    m3.metric("AUC thin-file (<6m history)", f"{CR['segment_metrics']['thin_file_lt6m']['auc']:.3f}")
    m4.metric("KS", f"{CR['metrics'][champ]['ks']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        curve = pd.DataFrame(CR["approval_curve"])
        fig = go.Figure()
        fig.add_scatter(x=curve.approval_rate, y=curve.npa_traditional, name="Traditional (demographics + terms)",
                        line=dict(color=SERIES[1], width=2), mode="lines+markers", marker=dict(size=8))
        fig.add_scatter(x=curve.approval_rate, y=curve.npa_model, name="Behavioural ML model",
                        line=dict(color=SERIES[0], width=2), mode="lines+markers", marker=dict(size=8))
        fig.update_layout(title="NPA rate of the approved book at each approval rate")
        fig.update_xaxes(tickformat=".0%", title="Approval rate")
        fig.update_yaxes(tickformat=".0%", title="NPA rate", rangemode="tozero")
        st.plotly_chart(style(fig), use_container_width=True)
    with right:
        imp = pd.Series(CR["top_features"]).head(10).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Top drivers (mean |SHAP|)", hovermode="closest")
        st.plotly_chart(style(fig), use_container_width=True)

    b = CR["decision_bands"]
    st.markdown(
        f"**Decision bands:** auto-approve PD < {b['approve_below_pd']:.1%} "
        f"({b['auto_approve_share']:.0%} of applications, NPA {b['npa_auto_approved']:.1%}) · "
        f"manual review {b['manual_review_share']:.0%} · auto-decline PD ≥ {b['decline_above_pd']:.0%} "
        f"({b['auto_decline_share']:.0%}, catches {b['bads_caught_by_decline']:.0%} of bad loans).")

    st.divider()
    st.subheader("Score a new application")
    raw, panel, profile, scorer = scoring_context()
    c1, c2, c3 = st.columns(3)
    acct = c1.number_input("Customer (account_id)", min_value=1, value=2, step=1)
    amount = c2.slider("Loan amount (CZK)", 10_000, 600_000, 120_000, step=10_000)
    duration = c3.select_slider("Duration (months)", [12, 24, 36, 48, 60], value=36)
    if acct not in set(profile["account_id"]):
        st.warning("Unknown account_id.")
    else:
        X = applicant_features(raw, panel, profile, [acct], LIVE_ASOF, amount, duration)
        s = scorer.score(X).iloc[0]
        colour = {"auto_approve": "good", "manual_review": "warning", "decline": "critical"}[s["decision"]]
        icon = {"good": "✅", "warning": "⚠️", "critical": "⛔"}[colour]
        r1, r2, r3 = st.columns(3)
        r1.metric("Credit score", int(s["score"]))
        r2.metric("Probability of default", f"{s['pd']:.1%}")
        r3.markdown(f"**Decision**<br><span style='color:{STATUS[colour]};font-size:1.4rem'>"
                    f"{icon} {s['decision'].replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        reasons = pd.DataFrame(scorer.reasons(X, top=5)[0])
        reasons["value"] = reasons["value"].map(lambda v: "n/a" if v is None or pd.isna(v) else f"{v:,.2f}")
        st.markdown("**Reason codes** (SHAP)")
        st.dataframe(reasons[["label", "value", "direction", "impact"]], hide_index=True,
                     use_container_width=True)

# ------------------------------------------------------------------------------ EWS
with tab_ews:
    red, amb = EW["red_alert"], EW["amber_alert"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Red precision", f"{red['precision']:.0%}", f"{red['true_flags']}/{red['borrowers_flagged']} borrowers",
              delta_color="off", delta_arrow="off")
    m2.metric("Red recall", f"{red['recall']:.0%}")
    m3.metric("Amber recall", f"{amb['recall']:.0%}", f"precision {amb['precision']:.0%}", delta_color="off", delta_arrow="off")
    m4.metric("Median lead time", f"{red['median_lead_days']:.0f} days", f"P25 {red['lead_days_p25']:.0f} days",
              delta_color="off", delta_arrow="off")
    st.caption(f"Label: {EW['panel']['label']}. Row-level AUC {EW['metrics_row_level']['auc']:.3f} over "
               f"{EW['panel']['rows']:,} loan-months. {EW['note']}")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Live watchlist**")
        w = q("SELECT loan_id, account_id, alert_level, risk, buffer_last, pti_3m, months_remaining "
              "FROM ews_watchlist ORDER BY risk DESC LIMIT 25")
        w["alert_level"] = w["alert_level"].map({"red": "⛔ red", "amber": "⚠️ amber", "green": "✅ green"})
        st.dataframe(w, hide_index=True, use_container_width=True,
                     column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1,
                                                                           format="%.2f")})
    with right:
        bt = q("SELECT * FROM ews_backtest")
        # Most clearly flagged delinquent loans first.
        bad = (bt[bt["onset_date"].notna()].groupby("loan_id")["p_oof"].max()
               .sort_values(ascending=False).index.tolist())
        loan = st.selectbox("Back-test a loan that went delinquent", bad, index=0)
        d = bt[bt["loan_id"] == loan].sort_values("asof_month")
        fig = go.Figure()
        fig.add_scatter(x=d.asof_month, y=d.p_oof, name="Out-of-fold risk", mode="lines+markers",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=8))
        fig.add_hline(y=red["threshold"], line=dict(color=STATUS["critical"], dash="dash", width=1),
                      annotation_text="red", annotation_position="top left")
        fig.add_hline(y=amb["threshold"], line=dict(color=STATUS["warning"], dash="dot", width=1),
                      annotation_text="amber", annotation_position="top left")
        onset = pd.Timestamp(d["onset_date"].iloc[0])
        fig.add_vline(x=onset, line=dict(color=MUTED, width=1))
        fig.add_annotation(x=onset, y=1, yref="paper", text="delinquency onset", showarrow=False,
                           xanchor="right", font=dict(color=MUTED))
        fig.update_layout(title=f"Loan {loan}: monthly risk score before onset", showlegend=False)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(style(fig), use_container_width=True)

# ------------------------------------------------------------------------------ segments & NBO
with tab_seg:
    prof = pd.DataFrame(SG["profiles"])
    seg_names = list(prof["segment"])
    cmap = dict(zip(seg_names, SERIES))
    left, right = st.columns([3, 2])
    with left:
        segs = q("SELECT * FROM segments")
        fig = px.scatter(segs, x="pc1", y="pc2", color="segment", symbol="segment",
                         color_discrete_map=cmap, opacity=0.55,
                         hover_data={"account_id": True, "pc1": False, "pc2": False})
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
        fig.update_layout(title=f"{SG['k']} behavioural segments (PCA projection)", hovermode="closest")
        st.plotly_chart(style(fig, 420), use_container_width=True)
    with right:
        st.markdown("**Segment profiles**")
        show = prof[["segment", "customers", "avg_balance", "avg_inflow", "avg_age", "card_share",
                     "loan_share", "avg_products"]]
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"avg_balance": st.column_config.NumberColumn(format="%.0f"),
                                    "avg_inflow": st.column_config.NumberColumn(format="%.0f"),
                                    "avg_age": st.column_config.NumberColumn(format="%.0f"),
                                    "card_share": st.column_config.NumberColumn(format="percent"),
                                    "loan_share": st.column_config.NumberColumn(format="percent")})

    st.divider()
    pp = NB["per_product"]
    rows = []
    for p, mtr in pp.items():
        if mtr["low_data"]:
            continue
        for label, key in [("Random", "conversion_random"), ("Highest balance rule", "conversion_top20_heuristic"),
                           ("Propensity model", "conversion_top20_model")]:
            rows.append({"product": p.replace("_", " ").title(), "targeting": label, "conversion": mtr[key]})
    conv = pd.DataFrame(rows)
    a, b = st.columns([3, 2])
    with a:
        fig = px.bar(conv, x="product", y="conversion", color="targeting", barmode="group",
                     color_discrete_sequence=[MUTED, SERIES[1], SERIES[0]], text_auto=".0%")
        fig.update_traces(marker_line_color="white", marker_line_width=2, textposition="outside")
        fig.update_layout(title="Conversion when contacting the top 20% of eligible customers (1998 holdout)",
                          hovermode="closest")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig), use_container_width=True)
    with b:
        st.markdown("**Customer next-best-offer**")
        cid = st.number_input("account_id", min_value=1, value=2, step=1, key="nbo")
        cust = q(f"SELECT * FROM customer_360 WHERE account_id = {int(cid)}")
        if cust.empty:
            st.warning("Unknown or too-new account.")
        else:
            r = cust.iloc[0]
            st.write(f"Segment: **{r['segment']}** · pre-approval: **{r['preapproval_decision']}** "
                     f"(score {r['preapproval_score']})")
            off = q(f"SELECT product, propensity, expected_value, eligible, blocked_reason, rank "
                    f"FROM offers WHERE account_id = {int(cid)} ORDER BY rank NULLS LAST")
            st.dataframe(off, hide_index=True, use_container_width=True,
                         column_config={"propensity": st.column_config.NumberColumn(format="%.3f"),
                                        "expected_value": st.column_config.NumberColumn(format="%.3f")})
    low = [p for p, m in pp.items() if m["low_data"]]
    if low:
        st.caption(f"{', '.join(low)}: too few adopters in the outcome window ({pp[low[0]]['adopters']}) "
                   "for a reliable uplift estimate. In this bank, standing-order products are set up at "
                   "onboarding, so they're offered from the model but excluded from the headline uplift.")
