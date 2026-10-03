"""
SHAP analysis of the trained loan-approval Random Forest.

Answers: "Which features contribute most to an approval decision, and in which direction?"

Run (after src/loan_approval.py has produced models/loan_approval_model.joblib):
    pip install shap
    python src/shap_analysis.py

Outputs
    reports/figures/16_shap_global_importance.png   mean |SHAP| per original feature
    reports/figures/17_shap_beeswarm.png            direction + spread of each feature's effect
    reports/figures/18_shap_dependence.png          how credit score / DTI / loan-to-income move the decision
    reports/figures/19_shap_waterfall_examples.png  why one applicant is approved and another rejected
    results/shap_feature_importance.csv             ranked table

How to read SHAP values here: the model outputs P(approved). A SHAP value is how many
probability points a feature adds (+) or removes (-) for one applicant, relative to the
average prediction (the "base value"). Values for all features sum to the final probability.
"""
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from predict import add_features          # same banking ratios used in training

warnings.filterwarnings("ignore")
SEED = 42                                  # must match loan_approval.py so the split is identical
ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "reports" / "figures", ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 1. load model and rebuild the test set
art = joblib.load(ROOT / "models" / "loan_approval_model.joblib")
pipe, threshold = art["model"], art["threshold"]
pre, rf = pipe.named_steps["pre"], pipe.named_steps["clf"]

df = pd.read_csv(ROOT / "data" / "loan_approval_data.csv").dropna(subset=["Loan_Approved"])
df = add_features(df.drop(columns=["Applicant_ID"]).reset_index(drop=True))
y = (df["Loan_Approved"] == "Yes").astype(int)
X = df.drop(columns=["Loan_Approved"])
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)   # same split as training
print(f"Explaining {len(X_te)} held-out applications (threshold {threshold:.2f})")

# ------------------------------------------------------------------ 2. transform and compute SHAP values
X_t = pd.DataFrame(pre.transform(X_te), columns=pre.get_feature_names_out(), index=X_te.index)
X_t.columns = [c.split("__", 1)[1] for c in X_t.columns]            # drop "num__" / "cat__" prefixes

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(X_t)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)   # keep the "approved" class
base = explainer.expected_value
base = float(base[1] if np.ndim(base) else base)

# Unscaled copy for readable axes / colours (numeric columns: median-imputed but NOT standardised)
num_cols = list(pre.transformers_[0][2])
num_unscaled = pre.named_transformers_["num"].named_steps["imp"].transform(X_te[num_cols])
X_disp = X_t.copy()
X_disp[num_cols] = num_unscaled
sv_df = pd.DataFrame(sv, columns=X_t.columns, index=X_t.index)

# ------------------------------------------------------------------ 3. aggregate one-hot columns back to the original feature
cat_cols = list(pre.transformers_[1][2])
def original_name(col):
    for c in cat_cols:
        if col.startswith(c + "_"):
            return c
    return col
group = {c: original_name(c) for c in sv_df.columns}
sv_orig = sv_df.T.groupby(group).sum().T                              # SHAP is additive, so summing is valid
imp = sv_orig.abs().mean().sort_values(ascending=False)
share = imp / imp.sum() * 100
table = pd.DataFrame({"mean_abs_shap": imp.round(4), "share_of_total_%": share.round(1)})
table.index.name = "feature"
table.to_csv(RES / "shap_feature_importance.csv")
print("\nMean |SHAP| (probability points) and share of total attribution:\n", table.head(10).to_string())

# ------------------------------------------------------------------ 4. global importance bar chart
top = imp.head(12)[::-1]
plt.figure(figsize=(9, 5.5))
bars = plt.barh(top.index, top.values, color="#1F3864")
for b, v in zip(bars, top.values):
    plt.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}  ({v / imp.sum() * 100:.0f}%)", va="center", fontsize=9)
plt.xlabel("Mean |SHAP value|  (average change in approval probability)")
plt.title("Which features drive loan approval? (SHAP, test set)")
plt.xlim(0, top.max() * 1.25); plt.tight_layout()
plt.savefig(FIG / "16_shap_global_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ 5. beeswarm (direction of effect)
shap.summary_plot(sv, X_disp, max_display=12, show=False, plot_size=(9.5, 6))
plt.title("SHAP beeswarm: each dot is one applicant (red = high feature value)")
plt.tight_layout(); plt.savefig(FIG / "17_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ 6. dependence plots for the top drivers
drivers = [c for c in ["Credit_Score", "DTI_Ratio", "Loan_to_Income"] if c in X_disp.columns]
fig, axes = plt.subplots(1, len(drivers), figsize=(6 * len(drivers), 4.8))
for ax, f in zip(np.atleast_1d(axes), drivers):
    shap.dependence_plot(f, sv, X_disp, interaction_index=None, ax=ax, show=False, dot_size=14, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_title(f"Effect of {f}")
    ax.set_ylabel("SHAP value (change in P(approved))")
fig.suptitle("Dependence plots: how the top features move the decision", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig(FIG / "18_shap_dependence.png", dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ 7. individual explanations (one approved, one rejected)
proba = pipe.predict_proba(X_te)[:, 1]
clear_yes = int(np.argmax(np.where(y_te.values == 1, proba, -1)))     # most confidently approved
clear_no = int(np.argmin(np.where(y_te.values == 0, proba, 2)))       # most confidently rejected
for name, i in [("approved", clear_yes), ("rejected", clear_no)]:
    exp = shap.Explanation(values=sv[i], base_values=base, data=X_disp.iloc[i].values, feature_names=list(X_disp.columns))
    shap.plots.waterfall(exp, max_display=9, show=False)
    plt.title(f"Example applicant ({name}): P(approved) = {proba[i]:.2f}", fontsize=11)
    plt.savefig(FIG / f"19_shap_waterfall_{name}.png", dpi=150, bbox_inches="tight"); plt.close()
# combine the two waterfalls into one figure
from PIL import Image
a, b = Image.open(FIG / "19_shap_waterfall_approved.png"), Image.open(FIG / "19_shap_waterfall_rejected.png")
h = max(a.height, b.height); canvas = Image.new("RGB", (a.width + b.width, h), "white")
canvas.paste(a, (0, 0)); canvas.paste(b, (a.width, 0)); canvas.save(FIG / "19_shap_waterfall_examples.png")
for n in ("approved", "rejected"):
    (FIG / f"19_shap_waterfall_{n}.png").unlink()

# ------------------------------------------------------------------ 8. plain-English summary
print("\n================ SUMMARY ================")
print(f"Base value (average predicted approval probability): {base:.3f}")
for f in imp.index[:4]:
    corr = np.corrcoef(X_disp[f] if f in X_disp else sv_orig[f], sv_orig[f])[0, 1] if f in X_disp else np.nan
    direction = "higher value -> more likely approved" if corr > 0.1 else ("higher value -> less likely approved" if corr < -0.1 else "non-linear / threshold effect")
    print(f"  {f:18s} mean|SHAP| = {imp[f]:.3f} ({share[f]:.0f}% of attribution): {direction}")
print(f"Top two features explain {share.iloc[:2].sum():.0f}% of the total attribution.")
print("Saved figures 16-19 to reports/figures and the ranking to results/shap_feature_importance.csv")
