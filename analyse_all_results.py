# ============================================================
#  COMPLETE RESULTS ANALYSIS — Phitcha Khrueapaeng
#  Ascencia Malta Business School | 2026
#
#  Analyses ALL output files from the thesis study:
#    experiment_results.csv     → participant-level search data
#    product_catalogue.csv      → 9,560 real Amazon products
#    category_analysis.csv      → category-level review stats
#    descriptive_statistics.csv → pre-computed descriptives
#    inferential_statistics.csv → pre-computed test results
#
#  Analysis sections:
#    A.  Data loading & overview
#    B.  Participant-level descriptive statistics
#    C.  Normality testing (Shapiro-Wilk)
#    D.  Primary inferential tests
#        D1. Search accuracy     — paired t-test / Wilcoxon
#        D2. Search efficiency   — paired t-test / Wilcoxon
#        D3. User satisfaction   — Mann-Whitney U
#    E.  Effect sizes (Cohen's d, rank-biserial r, eta-squared)
#    F.  Sub-group analysis by query type (specific/vague/browse)
#    G.  Per-task analysis (Tasks 1–6)
#    H.  SUS item-level analysis
#    I.  Per-result-position analysis (Result 1–5)
#    J.  Correlation analysis
#    K.  Product catalogue analysis (9,560 real products)
#    L.  Category-level analysis (real Amazon data)
#    M.  Regression — what predicts user relevance rating?
#    N.  Visualisations (12 publication-quality figures)
#    O.  Auto-generated thesis Chapter 4 text
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kruskal, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, textwrap

warnings.filterwarnings('ignore')

# ── Output folder ──────────────────────────────────────────────────────────
OUT = "analysis_output"
os.makedirs(OUT, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.20,
    "grid.linestyle"    : "--",
    "figure.dpi"        : 120,
})
C = {
    "ai"   : "#1D9E75",   # teal
    "kw"   : "#D85A30",   # coral
    "sus"  : "#7F77DD",   # purple
    "neu"  : "#888780",   # grey
    "dark" : "#2C2C2A",   # near-black
    "sig"  : "#0F6E56",   # dark teal
    "ns"   : "#993C1D",   # dark coral
    "amber": "#BA7517",
    "blue" : "#185FA5",
}

ALPHA = 0.05

def sep(title=""):
    print("\n" + "="*65)
    if title: print(f"  {title}")
    print("="*65)

def subsep(title):
    print(f"\n  {'─'*55}")
    print(f"  {title}")
    print(f"  {'─'*55}")

def p_fmt(p):
    if p < 0.001: return "p < .001"
    if p < 0.01:  return "p < .01"
    if p < 0.05:  return "p < .05"
    return f"p = {p:.3f}"

def eff_label(d):
    d = abs(float(d))
    if d >= 0.8: return "Large"
    if d >= 0.5: return "Medium"
    return "Small"

def sus_grade(score):
    score = float(score)
    if score >= 85: return "Excellent (A)"
    if score >= 72: return "Good (B)"
    if score >= 52: return "OK (C)"
    if score >= 38: return "Poor (D)"
    return "Awful (F)"

def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return abs(np.mean(diff)) / np.std(diff, ddof=1)

def eta_squared(f_stat, df_between, df_within):
    return (f_stat * df_between) / (f_stat * df_between + df_within)

def save_fig(name):
    path = f"{OUT}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {name}.png")

# ══════════════════════════════════════════════════════════════════════════
# A. DATA LOADING & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
sep("A.  DATA LOADING & OVERVIEW")

DATA_DIR = "/thesis_results_real"

exp   = pd.read_csv(f"{DATA_DIR}/experiment_results.csv")
prod  = pd.read_csv(f"{DATA_DIR}/product_catalogue.csv")
cats  = pd.read_csv(f"{DATA_DIR}/category_analysis.csv")
desc  = pd.read_csv(f"{DATA_DIR}/descriptive_statistics.csv")
infr  = pd.read_csv(f"{DATA_DIR}/inferential_statistics.csv")

# SUS score (Brooke 1996)
def sus_score(row):
    odd  = [row[f"sus_{i}"] - 1 for i in [1,3,5,7,9]]
    even = [5 - row[f"sus_{i}"] for i in [2,4,6,8,10]]
    return sum(odd + even) * 2.5

exp["sus_score"] = exp.apply(sus_score, axis=1)

# Rating columns
rat_cols = [f"result_{i}_rating" for i in range(1, 6)]
sus_cols = [f"sus_{i}" for i in range(1, 11)]

print(f"  Experiment data  : {len(exp):,} rows × {len(exp.columns)} cols")
print(f"  Participants     : {exp['participant_id'].nunique()}")
print(f"  Conditions       : {exp['condition'].unique().tolist()}")
print(f"  Tasks per cond   : {exp['task_id'].nunique()}")
print(f"  Task types       : {exp['task_type'].unique().tolist()}")
print(f"  Product catalogue: {len(prod):,} items")
print(f"  Categories       : {len(cats)}")

# Aggregate to participant × condition level
agg = (exp.groupby(["participant_id", "condition"])
          .agg(mean_rel=("mean_relevance","mean"),
               mean_time=("task_time_s","mean"),
               mean_sus=("sus_score","mean"))
          .reset_index())

ai = agg[agg["condition"]=="AI"].set_index("participant_id")
kw = agg[agg["condition"]=="Keyword"].set_index("participant_id")
pid = ai.index

ai_rel  = ai.loc[pid,"mean_rel"].values
kw_rel  = kw.loc[pid,"mean_rel"].values
ai_time = ai.loc[pid,"mean_time"].values
kw_time = kw.loc[pid,"mean_time"].values
ai_sus  = ai.loc[pid,"mean_sus"].values
kw_sus  = kw.loc[pid,"mean_sus"].values
N = len(pid)
print(f"\n  ✓ Participant arrays built (N={N})")

agg.to_csv(f"{OUT}/A_participant_aggregated.csv", index=False)


'''
# ══════════════════════════════════════════════════════════════════════════
# B. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════
sep("B.  DESCRIPTIVE STATISTICS")

def full_desc(arr, label):
    a = np.array(arr)
    return {
        "Variable"  : label,
        "n"         : len(a),
        "Mean"      : round(float(np.mean(a)), 4),
        "SD"        : round(float(np.std(a, ddof=1)), 4),
        "SE"        : round(float(np.std(a, ddof=1) / np.sqrt(len(a))), 4),
        "Median"    : round(float(np.median(a)), 4),
        "IQR"       : round(float(np.percentile(a,75) - np.percentile(a,25)), 4),
        "Min"       : round(float(np.min(a)), 4),
        "Max"       : round(float(np.max(a)), 4),
        "Skewness"  : round(float(stats.skew(a)), 4),
        "Kurtosis"  : round(float(stats.kurtosis(a)), 4),
        "95CI_lower": round(float(np.mean(a) - 1.96 * np.std(a,ddof=1)/np.sqrt(len(a))), 4),
        "95CI_upper": round(float(np.mean(a) + 1.96 * np.std(a,ddof=1)/np.sqrt(len(a))), 4),
    }

desc_rows = [
    full_desc(ai_rel,  "Relevance — AI"),
    full_desc(kw_rel,  "Relevance — Keyword"),
    full_desc(ai_rel - kw_rel, "Relevance — difference (AI−KW)"),
    full_desc(ai_time, "Task time (s) — AI"),
    full_desc(kw_time, "Task time (s) — Keyword"),
    full_desc(ai_time - kw_time, "Task time (s) — difference (AI−KW)"),
    full_desc(ai_sus,  "SUS score — AI"),
    full_desc(kw_sus,  "SUS score — Keyword"),
    full_desc(ai_sus  - kw_sus,  "SUS score — difference (AI−KW)"),
]
desc_full_df = pd.DataFrame(desc_rows)
desc_full_df.to_csv(f"{OUT}/B_descriptive_statistics_full.csv", index=False)
print(desc_full_df[["Variable","Mean","SD","Median","IQR","Skewness"]].to_string(index=False))

# SUS grade interpretation
print(f"\n  SUS Grades:")
print(f"    AI      Mdn={np.median(ai_sus):.2f}  → {sus_grade(np.median(ai_sus))}")
print(f"    Keyword Mdn={np.median(kw_sus):.2f}  → {sus_grade(np.median(kw_sus))}")

# ══════════════════════════════════════════════════════════════════════════
# C. NORMALITY TESTING
# ══════════════════════════════════════════════════════════════════════════
sep("C.  NORMALITY TESTING (Shapiro-Wilk)")

norm_rows = []
for arr, label in [
    (ai_rel,           "Relevance — AI"),
    (kw_rel,           "Relevance — Keyword"),
    (ai_rel - kw_rel,  "Relevance — difference"),
    (ai_time,          "Task time — AI"),
    (kw_time,          "Task time — Keyword"),
    (ai_time - kw_time,"Task time — difference"),
    (ai_sus,           "SUS — AI"),
    (kw_sus,           "SUS — Keyword"),
    (ai_sus - kw_sus,  "SUS — difference"),
]:
    w, p = stats.shapiro(arr)
    norm_rows.append({"Variable": label, "W": round(w,4), "p": round(p,4),
                      "Normal (α=.05)": p > ALPHA})

norm_df = pd.DataFrame(norm_rows)
norm_df.to_csv(f"{OUT}/C_normality_tests.csv", index=False)
print(norm_df.to_string(index=False))

rel_normal  = norm_df[norm_df["Variable"]=="Relevance — difference"]["Normal (α=.05)"].values[0]
time_normal = norm_df[norm_df["Variable"]=="Task time — difference"]["Normal (α=.05)"].values[0]

# ══════════════════════════════════════════════════════════════════════════
# D. PRIMARY INFERENTIAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════
sep("D.  PRIMARY INFERENTIAL STATISTICS")

# ── D1: Search Accuracy ────────────────────────────────────────────────────
subsep("D1. Search Accuracy — Relevance Ratings")
if rel_normal:
    t_r, p_r = stats.ttest_rel(ai_rel, kw_rel)
    test_r = "Paired t-test"
    stat_r_name = f"t({N-1})"
else:
    t_r, p_r = stats.wilcoxon(ai_rel, kw_rel)
    test_r = "Wilcoxon signed-rank"
    stat_r_name = "W"

d_r     = cohens_d_paired(ai_rel, kw_rel)
sig_r   = p_r < ALPHA
ci_r    = stats.t.interval(0.95, df=N-1,
            loc=np.mean(ai_rel-kw_rel),
            scale=stats.sem(ai_rel-kw_rel))

print(f"  Test applied : {test_r}")
print(f"  AI  : M={np.mean(ai_rel):.4f}  SD={np.std(ai_rel,ddof=1):.4f}  Mdn={np.median(ai_rel):.4f}")
print(f"  KW  : M={np.mean(kw_rel):.4f}  SD={np.std(kw_rel,ddof=1):.4f}  Mdn={np.median(kw_rel):.4f}")
print(f"  Diff: M={np.mean(ai_rel-kw_rel):+.4f}  95%CI [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
print(f"  {stat_r_name} = {t_r:.4f},  {p_fmt(p_r)},  d = {d_r:.4f} ({eff_label(d_r)} effect)")
print(f"  → {'✓ SIGNIFICANT — H₁ supported' if sig_r else '✗ Not significant — H₀ retained'}")

# ── D2: Search Efficiency ──────────────────────────────────────────────────
subsep("D2. Search Efficiency — Task Completion Time")
if time_normal:
    t_t, p_t = stats.ttest_rel(ai_time, kw_time)
    test_t = "Paired t-test"
    stat_t_name = f"t({N-1})"
else:
    t_t, p_t = stats.wilcoxon(ai_time, kw_time)
    test_t = "Wilcoxon signed-rank"
    stat_t_name = "W"

d_t      = cohens_d_paired(ai_time, kw_time)
sig_t    = p_t < ALPHA
ci_t     = stats.t.interval(0.95, df=N-1,
              loc=np.mean(ai_time-kw_time),
              scale=stats.sem(ai_time-kw_time))
time_saved = np.mean(kw_time) - np.mean(ai_time)
time_pct   = time_saved / np.mean(kw_time) * 100

print(f"  Test applied : {test_t}")
print(f"  AI  : M={np.mean(ai_time):.2f}s  SD={np.std(ai_time,ddof=1):.2f}")
print(f"  KW  : M={np.mean(kw_time):.2f}s  SD={np.std(kw_time,ddof=1):.2f}")
print(f"  Diff: M={np.mean(ai_time-kw_time):+.2f}s  95%CI [{ci_t[0]:.2f}, {ci_t[1]:.2f}]")
print(f"  Time saved with AI: {time_saved:.2f}s  ({time_pct:.1f}% reduction)")
print(f"  {stat_t_name} = {t_t:.4f},  {p_fmt(p_t)},  d = {d_t:.4f} ({eff_label(d_t)} effect)")
print(f"  → {'✓ SIGNIFICANT — H₁ supported' if sig_t else '✗ Not significant — H₀ retained'}")

# ── D3: User Satisfaction ──────────────────────────────────────────────────
subsep("D3. User Satisfaction — SUS Scores")
u_stat, u_p = stats.mannwhitneyu(ai_sus, kw_sus, alternative="two-sided")
rb          = 1 - (2 * u_stat) / (N * N)
sig_s       = u_p < ALPHA

print(f"  Test applied : Mann-Whitney U")
print(f"  AI  : Mdn={np.median(ai_sus):.2f}  M={np.mean(ai_sus):.2f}  SD={np.std(ai_sus,ddof=1):.2f}  → {sus_grade(np.median(ai_sus))}")
print(f"  KW  : Mdn={np.median(kw_sus):.2f}  M={np.mean(kw_sus):.2f}  SD={np.std(kw_sus,ddof=1):.2f}  → {sus_grade(np.median(kw_sus))}")
print(f"  U = {u_stat:.1f},  {p_fmt(u_p)},  r = {rb:.4f} ({eff_label(rb)} effect)")
print(f"  → {'✓ SIGNIFICANT — H₁ supported' if sig_s else '✗ Not significant — H₀ retained'}")

# ── Save combined inferential table ───────────────────────────────────────
inf_full = pd.DataFrame([
    {"Measure":"Search accuracy","Test":test_r,
     "Statistic":round(t_r,4),"df":N-1 if rel_normal else "—",
     "p":round(p_r,4),"95CI_lower":round(ci_r[0],4),"95CI_upper":round(ci_r[1],4),
     "Cohens_d":round(d_r,4),"Effect":eff_label(d_r),"Significant":sig_r},
    {"Measure":"Search efficiency","Test":test_t,
     "Statistic":round(t_t,4),"df":N-1 if time_normal else "—",
     "p":round(p_t,4),"95CI_lower":round(ci_t[0],4),"95CI_upper":round(ci_t[1],4),
     "Cohens_d":round(d_t,4),"Effect":eff_label(d_t),"Significant":sig_t},
    {"Measure":"User satisfaction (SUS)","Test":"Mann-Whitney U",
     "Statistic":round(u_stat,1),"df":"—",
     "p":round(u_p,4),"95CI_lower":"—","95CI_upper":"—",
     "Cohens_d":round(abs(rb),4),"Effect":eff_label(rb),"Significant":sig_s},
])
inf_full.to_csv(f"{OUT}/D_inferential_statistics_full.csv", index=False)
print(f"\n  ✓ Saved D_inferential_statistics_full.csv")

# ══════════════════════════════════════════════════════════════════════════
# E. EFFECT SIZES
# ══════════════════════════════════════════════════════════════════════════
sep("E.  EFFECT SIZES")

# Eta-squared for paired t-tests
eta_r = t_r**2 / (t_r**2 + (N-1))  if rel_normal  else None
eta_t = t_t**2 / (t_t**2 + (N-1))  if time_normal else None

eff_rows = [
    {"Measure":"Search accuracy",    "Cohen_d":round(d_r,4),    "r_effect":"—",        "Eta_sq":round(eta_r,4) if eta_r else "—", "Interpretation":eff_label(d_r)},
    {"Measure":"Search efficiency",  "Cohen_d":round(d_t,4),    "r_effect":"—",        "Eta_sq":round(eta_t,4) if eta_t else "—", "Interpretation":eff_label(d_t)},
    {"Measure":"User satisfaction",  "Cohen_d":"—",             "r_effect":round(rb,4),"Eta_sq":"—",                               "Interpretation":eff_label(rb)},
]
eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv(f"{OUT}/E_effect_sizes.csv", index=False)
print(eff_df.to_string(index=False))
print("\n  Benchmarks: Small d≥0.2 | Medium d≥0.5 | Large d≥0.8")

# ══════════════════════════════════════════════════════════════════════════
# F. SUB-GROUP ANALYSIS BY QUERY TYPE
# ══════════════════════════════════════════════════════════════════════════
sep("F.  SUB-GROUP ANALYSIS BY QUERY TYPE")

subgroup_rows = []
for qt in ["specific", "vague", "browse"]:
    sub = exp[exp["task_type"] == qt]
    ai_sub = sub[sub["condition"]=="AI"]["mean_relevance"].values
    kw_sub = sub[sub["condition"]=="Keyword"]["mean_relevance"].values
    ai_time_sub = sub[sub["condition"]=="AI"]["task_time_s"].values
    kw_time_sub = sub[sub["condition"]=="Keyword"]["task_time_s"].values

    # Welch t-test (independent, different participants per task)
    t_rel,  p_rel  = stats.ttest_ind(ai_sub, kw_sub, equal_var=False)
    t_time, p_time = stats.ttest_ind(ai_time_sub, kw_time_sub, equal_var=False)

    subgroup_rows.append({
        "Query type"       : qt,
        "AI rel mean"      : round(np.mean(ai_sub),4),
        "KW rel mean"      : round(np.mean(kw_sub),4),
        "Rel diff"         : round(np.mean(ai_sub)-np.mean(kw_sub),4),
        "Rel t"            : round(t_rel,4),
        "Rel p"            : round(p_rel,4),
        "Rel sig"          : p_rel < ALPHA,
        "AI time mean"     : round(np.mean(ai_time_sub),2),
        "KW time mean"     : round(np.mean(kw_time_sub),2),
        "Time saved"       : round(np.mean(kw_time_sub)-np.mean(ai_time_sub),2),
        "Time t"           : round(t_time,4),
        "Time p"           : round(p_time,4),
    })
    print(f"  [{qt.upper()}]  Relevance: AI={np.mean(ai_sub):.3f} KW={np.mean(kw_sub):.3f}  "
          f"Δ={np.mean(ai_sub)-np.mean(kw_sub):+.3f}  {p_fmt(p_rel)}  "
          f"| Time saved: {np.mean(kw_time_sub)-np.mean(ai_time_sub):.1f}s")

sg_df = pd.DataFrame(subgroup_rows)
sg_df.to_csv(f"{OUT}/F_subgroup_query_type.csv", index=False)

# Kruskal-Wallis across query types (AI condition)
ai_sp = exp[(exp["condition"]=="AI") & (exp["task_type"]=="specific")]["mean_relevance"]
ai_vg = exp[(exp["condition"]=="AI") & (exp["task_type"]=="vague")]["mean_relevance"]
ai_br = exp[(exp["condition"]=="AI") & (exp["task_type"]=="browse")]["mean_relevance"]
kw_sp = exp[(exp["condition"]=="Keyword") & (exp["task_type"]=="specific")]["mean_relevance"]
kw_vg = exp[(exp["condition"]=="Keyword") & (exp["task_type"]=="vague")]["mean_relevance"]
kw_br = exp[(exp["condition"]=="Keyword") & (exp["task_type"]=="browse")]["mean_relevance"]

h_ai, p_ai_kw = kruskal(ai_sp, ai_vg, ai_br)
h_kw, p_kw_kw = kruskal(kw_sp, kw_vg, kw_br)
print(f"\n  Kruskal-Wallis across query types:")
print(f"    AI condition:      H={h_ai:.4f}  {p_fmt(p_ai_kw)}")
print(f"    Keyword condition: H={h_kw:.4f}  {p_fmt(p_kw_kw)}")

# ══════════════════════════════════════════════════════════════════════════
# G. PER-TASK ANALYSIS (Tasks 1–6)
# ══════════════════════════════════════════════════════════════════════════
sep("G.  PER-TASK ANALYSIS")

task_rows = []
for tid in sorted(exp["task_id"].unique()):
    te = exp[exp["task_id"]==tid]
    ai_t = te[te["condition"]=="AI"]["mean_relevance"].values
    kw_t = te[te["condition"]=="Keyword"]["mean_relevance"].values
    ai_tm= te[te["condition"]=="AI"]["task_time_s"].values
    kw_tm= te[te["condition"]=="Keyword"]["task_time_s"].values
    t_s, p_s = stats.ttest_ind(ai_t, kw_t, equal_var=False)
    tt_s,tp_s = stats.ttest_ind(ai_tm, kw_tm, equal_var=False)
    qt = te["task_type"].iloc[0]
    task_rows.append({
        "Task"            : tid,
        "Type"            : qt,
        "Query"           : te["query"].iloc[0][:50],
        "AI_rel_mean"     : round(np.mean(ai_t),4),
        "KW_rel_mean"     : round(np.mean(kw_t),4),
        "Rel_diff"        : round(np.mean(ai_t)-np.mean(kw_t),4),
        "Rel_p"           : round(p_s,4),
        "AI_time_mean"    : round(np.mean(ai_tm),2),
        "KW_time_mean"    : round(np.mean(kw_tm),2),
        "Time_saved"      : round(np.mean(kw_tm)-np.mean(ai_tm),2),
        "Time_p"          : round(tp_s,4),
    })
    print(f"  Task {tid} [{qt:8s}] | Rel: AI={np.mean(ai_t):.3f} KW={np.mean(kw_t):.3f} "
          f"({p_fmt(p_s)}) | Time saved: {np.mean(kw_tm)-np.mean(ai_tm):.1f}s ({p_fmt(tp_s)})")

task_df = pd.DataFrame(task_rows)
task_df.to_csv(f"{OUT}/G_per_task_analysis.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# H. SUS ITEM-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
sep("H.  SUS ITEM-LEVEL ANALYSIS")

SUS_ITEMS = [
    "I think I would like to use this system frequently",
    "I found the system unnecessarily complex",
    "I thought the system was easy to use",
    "I think I would need support to use this system",
    "I found the various functions well integrated",
    "I thought there was too much inconsistency",
    "I imagine most people would learn this quickly",
    "I found the system very cumbersome to use",
    "I felt very confident using this system",
    "I needed to learn a lot before getting going",
]

sus_item_rows = []
for i in range(1, 11):
    col = f"sus_{i}"
    ai_it  = exp[exp["condition"]=="AI"][col].values
    kw_it  = exp[exp["condition"]=="Keyword"][col].values
    u, pu  = stats.mannwhitneyu(ai_it, kw_it, alternative="two-sided")
    rb_it  = 1 - (2*u)/(len(ai_it)*len(kw_it))
    sus_item_rows.append({
        "Item"        : i,
        "Statement"   : SUS_ITEMS[i-1][:50],
        "AI_mean"     : round(np.mean(ai_it),3),
        "KW_mean"     : round(np.mean(kw_it),3),
        "Diff"        : round(np.mean(ai_it)-np.mean(kw_it),3),
        "U"           : round(u,1),
        "p"           : round(pu,4),
        "r"           : round(rb_it,3),
        "Significant" : pu < ALPHA,
    })
    direction = "+" if np.mean(ai_it) > np.mean(kw_it) else "-"
    print(f"  Item {i:2d} [{direction}] AI={np.mean(ai_it):.2f} KW={np.mean(kw_it):.2f}  "
          f"r={rb_it:.3f}  {p_fmt(pu)}  — {SUS_ITEMS[i-1][:45]}")

sus_item_df = pd.DataFrame(sus_item_rows)
sus_item_df.to_csv(f"{OUT}/H_sus_item_analysis.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# I. PER-RESULT-POSITION ANALYSIS (Result 1–5)
# ══════════════════════════════════════════════════════════════════════════
sep("I.  PER-RESULT-POSITION ANALYSIS")

pos_rows = []
for pos in range(1, 6):
    col = f"result_{pos}_rating"
    ai_pos = exp[exp["condition"]=="AI"][col].values
    kw_pos = exp[exp["condition"]=="Keyword"][col].values
    t_pos, p_pos = stats.ttest_ind(ai_pos, kw_pos, equal_var=False)
    pos_rows.append({
        "Position" : pos,
        "AI_mean"  : round(np.mean(ai_pos),4),
        "AI_SD"    : round(np.std(ai_pos,ddof=1),4),
        "KW_mean"  : round(np.mean(kw_pos),4),
        "KW_SD"    : round(np.std(kw_pos,ddof=1),4),
        "Diff"     : round(np.mean(ai_pos)-np.mean(kw_pos),4),
        "t"        : round(t_pos,4),
        "p"        : round(p_pos,4),
        "Sig"      : p_pos < ALPHA,
    })
    print(f"  Result #{pos}: AI={np.mean(ai_pos):.3f}({np.std(ai_pos,ddof=1):.3f}) "
          f"KW={np.mean(kw_pos):.3f}({np.std(kw_pos,ddof=1):.3f})  "
          f"Δ={np.mean(ai_pos)-np.mean(kw_pos):+.3f}  {p_fmt(p_pos)}")

pos_df = pd.DataFrame(pos_rows)
pos_df.to_csv(f"{OUT}/I_per_position_analysis.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# J. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
sep("J.  CORRELATION ANALYSIS")

corr_rows = []
pairs = [
    (ai_rel,  ai_time,  "AI: relevance vs time",    "Pearson"),
    (ai_rel,  ai_sus,   "AI: relevance vs SUS",      "Pearson"),
    (ai_time, ai_sus,   "AI: time vs SUS",           "Pearson"),
    (kw_rel,  kw_time,  "KW: relevance vs time",     "Pearson"),
    (kw_rel,  kw_sus,   "KW: relevance vs SUS",      "Pearson"),
    (kw_time, kw_sus,   "KW: time vs SUS",           "Pearson"),
    (ai_rel - kw_rel, ai_time - kw_time,
                        "Diff rel vs Diff time",     "Spearman"),
    (ai_sus  - kw_sus,  ai_rel  - kw_rel,
                        "Diff SUS vs Diff rel",      "Spearman"),
]
for x, y, label, method in pairs:
    if method == "Pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    corr_rows.append({"Pair": label, "Method": method,
                      "r": round(r,4), "p": round(p,4), "Significant": p < ALPHA})
    print(f"  {label:40s} {method} r={r:+.4f}  {p_fmt(p)}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f"{OUT}/J_correlation_analysis.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# K. PRODUCT CATALOGUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
sep("K.  PRODUCT CATALOGUE ANALYSIS (9,560 real Amazon products)")

print(f"  Products         : {len(prod):,}")
print(f"  Categories       : {prod['category'].nunique()}")
print(f"  Mean rating      : {prod['mean_rating'].mean():.4f}")
print(f"  Rating range     : {prod['mean_rating'].min():.2f} – {prod['mean_rating'].max():.2f}")
print(f"  Mean n_reviews   : {prod['n_reviews'].mean():.1f}")
print(f"  Median n_reviews : {prod['n_reviews'].median():.0f}")
print(f"  Max n_reviews    : {prod['n_reviews'].max():,}")

# Per-category stats
cat_prod = (prod.groupby("category")
                .agg(n_products=("item_id","count"),
                     mean_rating=("mean_rating","mean"),
                     median_reviews=("n_reviews","median"),
                     pct_high_rating=("mean_rating", lambda x: (x>=4.0).mean()*100))
                .reset_index().sort_values("n_products",ascending=False))
cat_prod.to_csv(f"{OUT}/K_product_catalogue_stats.csv", index=False)
print(f"\n  Per-category breakdown:")
print(cat_prod[["category","n_products","mean_rating","median_reviews","pct_high_rating"]].to_string(index=False))

# Rating distribution buckets
bins = [0,1.5,2.5,3.5,4.5,5.001]
labels_b = ["1.0","1.5–2.5","2.5–3.5","3.5–4.5","4.5–5.0"]
prod["rating_bucket"] = pd.cut(prod["mean_rating"], bins=bins, labels=labels_b)
bucket_dist = prod["rating_bucket"].value_counts().sort_index()
print(f"\n  Product rating distribution:")
for bucket, count in bucket_dist.items():
    print(f"    {bucket}: {count:,} products ({count/len(prod)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# L. CATEGORY-LEVEL ANALYSIS (real Amazon review data)
# ══════════════════════════════════════════════════════════════════════════
sep("L.  CATEGORY-LEVEL ANALYSIS (real Amazon reviews)")

cats_sorted = cats.sort_values("n_reviews", ascending=False)
cats_sorted["pct_of_total"] = (cats_sorted["n_reviews"] / cats_sorted["n_reviews"].sum() * 100).round(2)
cats_sorted.to_csv(f"{OUT}/L_category_analysis.csv", index=False)
print(cats_sorted.to_string(index=False))

# One-way ANOVA on mean ratings across categories (using category-level data)
# Compare to overall mean
f_cat, p_cat = stats.f_oneway(*[
    prod[prod["category"]==c]["mean_rating"].values
    for c in prod["category"].unique()
])
print(f"\n  ANOVA — mean rating across categories: F={f_cat:.4f}  {p_fmt(p_cat)}")

# ══════════════════════════════════════════════════════════════════════════
# M. REGRESSION — WHAT PREDICTS USER RELEVANCE RATING?
# ══════════════════════════════════════════════════════════════════════════
sep("M.  REGRESSION ANALYSIS")

# Build regression dataset: merge experiment with product catalogue
exp_with_prod = exp.merge(
    prod[["item_id","mean_rating","n_reviews","lgb_score"]] if "lgb_score" in prod.columns
    else prod[["item_id","mean_rating","n_reviews"]],
    how="left", left_on="task_id", right_on="item_id"
)

# Features: condition_enc, task_type_enc, task_id, mean_product_rating
le_cond = LabelEncoder(); le_type = LabelEncoder()
exp["condition_enc"] = le_cond.fit_transform(exp["condition"])  # AI=0, Keyword=1
exp["task_type_enc"] = le_type.fit_transform(exp["task_type"])  # browse=0, specific=1, vague=2

X_reg = exp[["condition_enc","task_type_enc","task_id"]].values
y_reg = exp["mean_relevance"].values

reg = LinearRegression().fit(X_reg, y_reg)
r2  = reg.score(X_reg, y_reg)

# Manual F-test
n_reg = len(y_reg); k_reg = X_reg.shape[1]
ss_res = np.sum((y_reg - reg.predict(X_reg))**2)
ss_tot = np.sum((y_reg - np.mean(y_reg))**2)
f_reg  = ((ss_tot - ss_res)/k_reg) / (ss_res/(n_reg-k_reg-1))
p_reg  = 1 - stats.f.cdf(f_reg, k_reg, n_reg-k_reg-1)

print(f"  Outcome variable  : Mean relevance rating")
print(f"  Predictors        : condition, query type, task ID")
print(f"  R²                : {r2:.4f}")
print(f"  Adjusted R²       : {1-(1-r2)*(n_reg-1)/(n_reg-k_reg-1):.4f}")
print(f"  F({k_reg},{n_reg-k_reg-1}) = {f_reg:.4f}  {p_fmt(p_reg)}")
print(f"  Coefficients:")
feat_names = ["Condition (AI=0, KW=1)", "Query type", "Task ID"]
for fname, coef in zip(feat_names, reg.coef_):
    print(f"    {fname:35s}: β = {coef:+.4f}")
print(f"  Intercept: {reg.intercept_:.4f}")

reg_df = pd.DataFrame({"Feature": feat_names, "Coefficient": reg.coef_})
reg_df.to_csv(f"{OUT}/M_regression_results.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# N. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════
sep("N.  GENERATING VISUALISATIONS")
print()

# ── Fig 1: Descriptive overview — 3 measures side by side ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 1: Search performance — AI vs keyword condition",
             fontsize=13, fontweight="bold", y=1.01)

for ax, (d1,d2), lbl, title, (c1,c2) in zip(axes, [
    (ai_rel,  kw_rel),
    (ai_time, kw_time),
    (ai_sus,  kw_sus),
], ["Mean relevance (1–5)","Task time (s)","SUS score (0–100)"],
   ["Search accuracy","Search efficiency","User satisfaction"],
   [(C["ai"],C["kw"]),(C["ai"],C["kw"]),(C["sus"],C["kw"])]):
    m1,m2 = np.mean(d1),np.mean(d2)
    s1,s2 = np.std(d1,ddof=1),np.std(d2,ddof=1)
    se1,se2 = s1/np.sqrt(N), s2/np.sqrt(N)
    bars = ax.bar([0,1],[m1,m2], color=[c1,c2], width=0.5, alpha=0.88,
                  yerr=[se1,se2], capsize=6,
                  error_kw={"elinewidth":1.4,"ecolor":C["dark"]})
    ax.set_xticks([0,1]); ax.set_xticklabels(["AI-based","Keyword"], fontsize=11)
    ax.set_ylabel(lbl, fontsize=10); ax.set_title(title, fontweight="bold")
    for bar, val in zip(bars,[m1,m2]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(se1,se2)*0.15,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="500")
    if title == "User satisfaction":
        ax.axhline(68, color=C["neu"], ls="--", lw=1, label="avg SUS=68")
        ax.legend(fontsize=9, framealpha=0); ax.set_ylim(0, 100)

plt.tight_layout(); save_fig("fig01_performance_overview")

# ── Fig 2: Box plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Figure 2: Score distributions — box plots", fontsize=13, fontweight="bold", y=1.01)
for ax, data, title, ylabel, cols in [
    (axes[0],[ai_rel,kw_rel],  "Search accuracy",  "Mean relevance",[C["ai"],C["kw"]]),
    (axes[1],[ai_time,kw_time],"Search efficiency", "Task time (s)", [C["ai"],C["kw"]]),
    (axes[2],[ai_sus,kw_sus],  "User satisfaction", "SUS score",     [C["sus"],C["kw"]]),
]:
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops={"color":"white","linewidth":2.5},
                    whiskerprops={"linewidth":1.2},
                    capprops={"linewidth":1.2},
                    flierprops={"marker":"o","markersize":4,"alpha":0.5})
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    ax.set_xticklabels(["AI","Keyword"]); ax.set_ylabel(ylabel); ax.set_title(title)
plt.tight_layout(); save_fig("fig02_boxplots")

# ── Fig 3: Per-task performance ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 3: Per-task analysis (Tasks 1–6)", fontsize=13, fontweight="bold")
for ax, col, ylabel, title in [
    (axes[0],"AI_rel_mean","Mean relevance (1–5)","Search accuracy per task"),
    (axes[1],"AI_time_mean","Task time (s)","Task completion time"),
]:
    x = np.arange(len(task_df)); w = 0.35
    kw_col = "KW_rel_mean" if col=="AI_rel_mean" else "KW_time_mean"
    b1 = axes[0 if col=="AI_rel_mean" else 1].bar(
        x-w/2, task_df[col], w, color=C["ai"], alpha=0.88, label="AI")
    b2 = axes[0 if col=="AI_rel_mean" else 1].bar(
        x+w/2, task_df[kw_col], w, color=C["kw"], alpha=0.88, label="Keyword")
    axes[0 if col=="AI_rel_mean" else 1].set_xticks(x)
    axes[0 if col=="AI_rel_mean" else 1].set_xticklabels(
        [f"T{row['Task']}\n{row['Type'][:3]}" for _,row in task_df.iterrows()], fontsize=9)
    axes[0 if col=="AI_rel_mean" else 1].legend(fontsize=9, framealpha=0)
    axes[0 if col=="AI_rel_mean" else 1].set_ylabel(ylabel)
    axes[0 if col=="AI_rel_mean" else 1].set_title(title)
plt.tight_layout(); save_fig("fig03_per_task")

# ── Fig 4: Query type sub-group ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 4: Performance by query type", fontsize=13, fontweight="bold")
qt_order = ["specific","vague","browse"]
sg_plot  = sg_df.set_index("Query type").loc[qt_order]
x = np.arange(3); w = 0.35
for ax_i, (ai_col, kw_col, ylabel, title) in enumerate([
    ("AI rel mean","KW rel mean","Mean relevance","Search accuracy"),
    ("AI time mean","KW time mean","Task time (s)","Task completion time"),
]):
    axes[ax_i].bar(x-w/2, sg_plot[ai_col], w, color=C["ai"], alpha=0.88, label="AI")
    axes[ax_i].bar(x+w/2, sg_plot[kw_col], w, color=C["kw"], alpha=0.88, label="Keyword")
    axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(qt_order, fontsize=11)
    axes[ax_i].set_ylabel(ylabel); axes[ax_i].set_title(title)
    axes[ax_i].legend(fontsize=9, framealpha=0)
plt.tight_layout(); save_fig("fig04_query_type")

# ── Fig 5: SUS item-level radar / bar ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 5: SUS item-level comparison", fontsize=13, fontweight="bold")
items = range(1, 11)
ai_means  = sus_item_df["AI_mean"].values
kw_means  = sus_item_df["KW_mean"].values
x = np.arange(10); w = 0.35
axes[0].bar(x-w/2, ai_means, w, color=C["sus"], alpha=0.88, label="AI")
axes[0].bar(x+w/2, kw_means, w, color=C["kw"], alpha=0.88, label="Keyword")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"Q{i}" for i in items], fontsize=10)
axes[0].set_ylabel("Mean score (1–5)"); axes[0].set_title("SUS item means by condition")
axes[0].axhline(3, color=C["neu"], ls="--", lw=0.8, label="Neutral (3)")
axes[0].legend(fontsize=9, framealpha=0)

diffs = sus_item_df["Diff"].values
bar_colors = [C["sig"] if d > 0 else C["ns"] for d in diffs]
axes[1].barh([f"Q{i}" for i in items], diffs, color=bar_colors, alpha=0.88)
axes[1].axvline(0, color=C["dark"], lw=1)
axes[1].set_xlabel("Mean difference (AI − Keyword)")
axes[1].set_title("SUS item differences (AI advantage = teal)")
plt.tight_layout(); save_fig("fig05_sus_items")

# ── Fig 6: Result position analysis ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(5); w = 0.35
b1 = ax.bar(x-w/2, pos_df["AI_mean"], w, color=C["ai"], alpha=0.88,
            yerr=pos_df["AI_SD"]/np.sqrt(180), capsize=4,
            error_kw={"elinewidth":1.2,"ecolor":C["dark"]}, label="AI")
b2 = ax.bar(x+w/2, pos_df["KW_mean"], w, color=C["kw"], alpha=0.88,
            yerr=pos_df["KW_SD"]/np.sqrt(180), capsize=4,
            error_kw={"elinewidth":1.2,"ecolor":C["dark"]}, label="Keyword")
ax.set_xticks(x); ax.set_xticklabels([f"Result #{i}" for i in range(1,6)])
ax.set_ylabel("Mean relevance rating (1–5)")
ax.set_title("Figure 6: Relevance by result position (error bars = SE)",
             fontweight="bold")
ax.legend(fontsize=10, framealpha=0); ax.set_ylim(0, 5.5)
plt.tight_layout(); save_fig("fig06_result_position")

# ── Fig 7: Effect sizes ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
lbl_e = ["Search accuracy\n(Cohen's d)","Search efficiency\n(Cohen's d)","User satisfaction\n(rank-biserial r)"]
d_e   = [d_r, d_t, abs(rb)]
col_e = [C["ai"], C["ai"], C["sus"]]
bars  = ax.barh(lbl_e, d_e, color=col_e, alpha=0.85, height=0.42)
for thresh, label_t, col_t in [
    (0.2,"Small (0.2)",C["neu"]),
    (0.5,"Medium (0.5)",C["dark"]),
    (0.8,"Large (0.8)",C["kw"]),
]:
    ax.axvline(thresh, color=col_t, ls="--", lw=1, label=label_t)
for bar, val in zip(bars, d_e):
    ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
            f"{val:.3f} ({eff_label(val)})", va="center", fontsize=10, fontweight="500")
ax.set_xlabel("Effect size magnitude")
ax.set_title("Figure 7: Effect sizes — all three dependent variables",
             fontweight="bold")
ax.legend(fontsize=9, framealpha=0, loc="lower right")
ax.set_xlim(0, max(d_e)+1.2)
plt.tight_layout(); save_fig("fig07_effect_sizes")

# ── Fig 8: Correlation heatmap ─────────────────────────────────────────────
corr_matrix_data = pd.DataFrame({
    "AI relevance" : ai_rel,
    "KW relevance" : kw_rel,
    "AI time"      : ai_time,
    "KW time"      : kw_time,
    "AI SUS"       : ai_sus,
    "KW SUS"       : kw_sus,
})
corr_mat = corr_matrix_data.corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_mat.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(corr_mat.columns, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(corr_mat.columns, fontsize=10)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr_mat.values[i,j]:.2f}", ha="center", va="center",
                fontsize=9, color="black" if abs(corr_mat.values[i,j]) < 0.6 else "white")
ax.set_title("Figure 8: Correlation matrix — all outcome variables",
             fontweight="bold", pad=12)
plt.tight_layout(); save_fig("fig08_correlation_heatmap")

# ── Fig 9: Product catalogue — rating distribution ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 9: Real Amazon product catalogue (9,560 products)",
             fontsize=13, fontweight="bold")
axes[0].hist(prod["mean_rating"], bins=30, color=C["ai"], alpha=0.85, edgecolor="white")
axes[0].axvline(prod["mean_rating"].mean(), color=C["kw"], ls="--", lw=1.5,
                label=f"Mean={prod['mean_rating'].mean():.2f}")
axes[0].set_xlabel("Mean product rating"); axes[0].set_ylabel("Number of products")
axes[0].set_title("Product rating distribution"); axes[0].legend(fontsize=9)

prod_cat = prod.groupby("category")["mean_rating"].mean().sort_values()
axes[1].barh(prod_cat.index, prod_cat.values, color=C["sus"], alpha=0.85)
for i, val in enumerate(prod_cat.values):
    axes[1].text(val+0.01, i, f"{val:.2f}", va="center", fontsize=9)
axes[1].set_xlabel("Mean rating"); axes[1].set_title("Mean rating by product category")
axes[1].set_xlim(3.5, 4.6)
plt.tight_layout(); save_fig("fig09_product_catalogue")

# ── Fig 10: Category analysis (real Amazon review data) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 10: Real Amazon Electronics — category analysis",
             fontsize=13, fontweight="bold")
cats_plot = cats.sort_values("n_reviews", ascending=True)
axes[0].barh(cats_plot["category"], cats_plot["n_reviews"]/1000,
             color=C["ai"], alpha=0.85)
axes[0].set_xlabel("Reviews (thousands)"); axes[0].set_title("Review volume by category (real data)")
axes[1].barh(cats_plot["category"], cats_plot["mean_rating"],
             color=C["sus"], alpha=0.85)
axes[1].axvline(cats["mean_rating"].mean(), color=C["kw"], ls="--", lw=1.5,
                label=f"Avg={cats['mean_rating'].mean():.2f}")
axes[1].set_xlabel("Mean star rating"); axes[1].set_title("Mean rating by category (real data)")
axes[1].legend(fontsize=9); axes[1].set_xlim(3.5, 4.4)
plt.tight_layout(); save_fig("fig10_category_analysis")

# ── Fig 11: SUS composite score comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 11: SUS score analysis", fontsize=13, fontweight="bold")

# SUS adjective rating scale benchmarks
benchmarks = [(85,"Excellent"),(72,"Good"),(52,"OK"),(38,"Poor")]
axes[0].hist(ai_sus, bins=12, color=C["sus"], alpha=0.75, label="AI", edgecolor="white")
axes[0].hist(kw_sus, bins=12, color=C["kw"], alpha=0.65, label="Keyword", edgecolor="white")
for score, label in benchmarks:
    axes[0].axvline(score, color=C["neu"], ls=":", lw=1)
    axes[0].text(score+0.5, axes[0].get_ylim()[1]*0.85, label, fontsize=8, color=C["neu"])
axes[0].set_xlabel("SUS score (0–100)"); axes[0].set_ylabel("Frequency")
axes[0].set_title("SUS score distribution"); axes[0].legend(fontsize=9)

# Violin plot
vp = axes[1].violinplot([ai_sus, kw_sus], positions=[0,1], showmedians=True,
                         showextrema=True)
for body, col in zip(vp["bodies"], [C["sus"], C["kw"]]):
    body.set_facecolor(col); body.set_alpha(0.75)
vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
for part in ["cbars","cmins","cmaxes"]:
    vp[part].set_color(C["dark"]); vp[part].set_linewidth(1.2)
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["AI-based","Keyword"])
axes[1].set_ylabel("SUS score"); axes[1].set_title("SUS distribution — violin plot")
for score, label in benchmarks:
    axes[1].axhline(score, color=C["neu"], ls=":", lw=0.8)
plt.tight_layout(); save_fig("fig11_sus_analysis")

# ── Fig 12: Regression coefficients ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
cols_r = [C["ns"] if c < 0 else C["sig"] for c in reg.coef_]
bars = ax.barh(feat_names, reg.coef_, color=cols_r, alpha=0.85, height=0.42)
ax.axvline(0, color=C["dark"], lw=1)
for bar, val in zip(bars, reg.coef_):
    ax.text(val + (0.005 if val >= 0 else -0.005),
            bar.get_y()+bar.get_height()/2,
            f"{val:+.4f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=10, fontweight="500")
ax.set_xlabel("Regression coefficient (β)")
ax.set_title(f"Figure 12: Regression — predictors of relevance rating  (R²={r2:.4f})",
             fontweight="bold")
plt.tight_layout(); save_fig("fig12_regression")

# ══════════════════════════════════════════════════════════════════════════
# O. AUTO-GENERATED CHAPTER 4 TEXT
# ══════════════════════════════════════════════════════════════════════════
sep("O.  AUTO-GENERATED THESIS CHAPTER 4 TEXT")

chapter4 = f"""
Chapter 4: Results
══════════════════════════════════════════════════════════════════

4.1 Overview
─────────────────────────────────────────────────────────────────
This chapter presents the results of the controlled user study (N={N})
comparing the AI-based product search prototype with the traditional
keyword-based system across three dependent variables: search accuracy
(mean relevance ratings), search efficiency (task completion time), and
user satisfaction (System Usability Scale scores). Quantitative data are
reported in Sections 4.2–4.4, followed by per-task and sub-group analyses
in Sections 4.5–4.6. The chapter concludes with regression findings (4.7)
and a summary of the hypothesis decisions (4.8).

Descriptive statistics for all outcome measures are presented in Table 4.1.

  Table 4.1: Descriptive statistics by condition
  ┌───────────────────────┬────────────┬────────────┬────────────┬────────────┐
  │ Measure               │ AI M (SD)  │ AI Mdn     │ KW M (SD)  │ KW Mdn     │
  ├───────────────────────┼────────────┼────────────┼────────────┼────────────┤
  │ Relevance rating (1–5)│ {np.mean(ai_rel):.3f}({np.std(ai_rel,ddof=1):.3f}) │ {np.median(ai_rel):.3f}     │ {np.mean(kw_rel):.3f}({np.std(kw_rel,ddof=1):.3f}) │ {np.median(kw_rel):.3f}     │
  │ Task time (s)         │ {np.mean(ai_time):.2f}({np.std(ai_time,ddof=1):.2f})  │ {np.median(ai_time):.2f}    │ {np.mean(kw_time):.2f}({np.std(kw_time,ddof=1):.2f}) │ {np.median(kw_time):.2f}    │
  │ SUS score (0–100)     │ {np.mean(ai_sus):.2f}({np.std(ai_sus,ddof=1):.2f})   │ {np.median(ai_sus):.2f}    │ {np.mean(kw_sus):.2f}({np.std(kw_sus,ddof=1):.2f})  │ {np.median(kw_sus):.2f}    │
  └───────────────────────┴────────────┴────────────┴────────────┴────────────┘


4.2 Normality Assessment
─────────────────────────────────────────────────────────────────
Prior to conducting the primary inferential tests, the normality of
difference scores for each dependent variable was assessed using the
Shapiro-Wilk test. For search accuracy (relevance ratings), the test
yielded W = {norm_df[norm_df['Variable']=='Relevance — difference']['W'].values[0]},
p = {norm_df[norm_df['Variable']=='Relevance — difference']['p'].values[0]:.4f},
indicating that the normality assumption {'was satisfied' if rel_normal else 'was violated'}.
For task completion time, the Shapiro-Wilk test returned
W = {norm_df[norm_df['Variable']=='Task time — difference']['W'].values[0]},
p = {norm_df[norm_df['Variable']=='Task time — difference']['p'].values[0]:.4f},
indicating that normality {'was satisfied' if time_normal else 'was violated'}.
{'Paired-samples t-tests were therefore applied for both variables.' if rel_normal and time_normal
 else 'Where normality was violated, the Wilcoxon signed-rank test was applied.'}


4.3 Search Accuracy
─────────────────────────────────────────────────────────────────
A {'paired-samples t-test' if rel_normal else 'Wilcoxon signed-rank test'} revealed a
{'statistically significant' if sig_r else 'non-significant'} difference in mean relevance
ratings between the AI-based system (M = {np.mean(ai_rel):.3f}, SD = {np.std(ai_rel,ddof=1):.3f})
and the keyword-based system (M = {np.mean(kw_rel):.3f}, SD = {np.std(kw_rel,ddof=1):.3f}),
{stat_r_name} = {t_r:.4f}, {p_fmt(p_r)},
95% CI [{ci_r[0]:.4f}, {ci_r[1]:.4f}], d = {d_r:.4f} ({eff_label(d_r)} effect).
The mean difference was {np.mean(ai_rel-kw_rel):+.4f} in favour of the
{'AI-based' if np.mean(ai_rel) > np.mean(kw_rel) else 'keyword-based'} system.
{'The null hypothesis H₀ is rejected; H₁ is supported for this dimension.' if sig_r
 else 'The result is not significant at α = .05; H₀ is retained.'}


4.4 Search Efficiency
─────────────────────────────────────────────────────────────────
A {'paired-samples t-test' if time_normal else 'Wilcoxon signed-rank test'} revealed a
{'significant' if sig_t else 'non-significant'} difference in task completion time.
Participants completed tasks {'faster' if np.mean(ai_time) < np.mean(kw_time) else 'slower'}
using the AI-based system (M = {np.mean(ai_time):.2f}s, SD = {np.std(ai_time,ddof=1):.2f})
compared to the keyword system (M = {np.mean(kw_time):.2f}s, SD = {np.std(kw_time,ddof=1):.2f}),
{stat_t_name} = {t_t:.4f}, {p_fmt(p_t)},
95% CI [{ci_t[0]:.2f}, {ci_t[1]:.2f}], d = {d_t:.4f} ({eff_label(d_t)} effect).
{'The AI system reduced task completion time by' + ' ' + f'{time_saved:.2f}s ({time_pct:.1f}%).' if sig_t else ''}
{'H₁ is supported for this dimension.' if sig_t else 'H₀ is retained.'}


4.5 User Satisfaction
─────────────────────────────────────────────────────────────────
A Mann-Whitney U test on SUS scores revealed that the AI-based system
(Mdn = {np.median(ai_sus):.2f}, M = {np.mean(ai_sus):.2f}, SD = {np.std(ai_sus,ddof=1):.2f})
received {'significantly higher' if sig_s else 'higher but not significantly different'}
usability ratings than the keyword system
(Mdn = {np.median(kw_sus):.2f}, M = {np.mean(kw_sus):.2f}, SD = {np.std(kw_sus,ddof=1):.2f}),
U = {u_stat:.1f}, {p_fmt(u_p)}, r = {rb:.4f} ({eff_label(rb)} effect).
The AI system's SUS score of {np.median(ai_sus):.1f} falls in the
'{sus_grade(np.median(ai_sus))}' range of the adjective scale,
while the keyword system's score of {np.median(kw_sus):.1f} falls in the
'{sus_grade(np.median(kw_sus))}' range (Brooke 1996).
{'H₁ is supported for this dimension.' if sig_s else 'H₀ is retained.'}


4.6 Sub-group Analysis by Query Type
─────────────────────────────────────────────────────────────────
Sub-group analysis by query type reveals differential patterns of AI
advantage across specific, vague, and browse query conditions.
{chr(10).join([f"  {row['Query type'].title()} queries: AI M={row['AI rel mean']:.3f}, KW M={row['KW rel mean']:.3f} (Δ={row['Rel diff']:+.3f}, {p_fmt(row['Rel p'])}). Time saved: {row['Time saved']:.1f}s." for _, row in sg_df.iterrows()])}


4.7 Regression Analysis
─────────────────────────────────────────────────────────────────
A multiple linear regression was conducted to identify predictors of user
relevance ratings. The model including condition (AI vs keyword), query type,
and task ID explained {r2*100:.1f}% of the variance in relevance ratings
(R² = {r2:.4f}, adjusted R² = {1-(1-r2)*(len(y_reg)-1)/(len(y_reg)-k_reg-1):.4f},
F({k_reg},{len(y_reg)-k_reg-1}) = {f_reg:.4f}, {p_fmt(p_reg)}).


4.8 Hypothesis Decision
─────────────────────────────────────────────────────────────────
H₁: AI-based product search significantly improves search accuracy
    and user satisfaction compared to traditional keyword-based systems.

  Search accuracy (relevance):   {'✓ SUPPORTED' if sig_r else '✗ NOT supported'}
  Search efficiency (task time): {'✓ SUPPORTED' if sig_t else '✗ NOT supported'}
  User satisfaction (SUS):       {'✓ SUPPORTED' if sig_s else '✗ NOT supported'}

  Overall: H₁ is {'FULLY SUPPORTED — H₀ rejected across all three measures.' if all([sig_r,sig_t,sig_s]) else 'PARTIALLY supported — see individual tests above.'}

══════════════════════════════════════════════════════════════════
"""

print(chapter4)
with open(f"{OUT}/O_chapter4_text.txt", "w") as f:
    f.write(chapter4)

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
sep("COMPLETE — ALL OUTPUTS SAVED")
outputs = sorted(os.listdir(OUT))
print(f"\n  Output directory: ./{OUT}/\n")
for o in outputs:
    size = os.path.getsize(f"{OUT}/{o}")
    print(f"  {o:50s}  {size:>8,} bytes")
print(f"\n  Total: {len(outputs)} files")
'''