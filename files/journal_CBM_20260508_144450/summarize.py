import pandas as pd
import numpy as np
import os
import glob
import sys

# Run this from INSIDE your journal_CBM_* folder
# python summarize.py > summary_output.txt

out = "."  # current directory is the journal_CBM folder

def section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def try_read(filepath, label):
    if os.path.exists(filepath):
        section(label)
        try:
            df = pd.read_csv(filepath)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"  ERROR reading {filepath}: {e}")
    else:
        print(f"  [MISSING] {filepath}")

# ── 1. Ablation table ─────────────────────────────────────────────────────────
try_read("ablation_table.csv", "1. ABLATION TABLE (zero-shot AUC all archs x anatomies)")

# ── 2. All improved DA results — n=500 summary ───────────────────────────────
f = "all_improved_da_results.csv"
if os.path.exists(f):
    section("2. IMPROVED DA RESULTS — n=500 (all archs, all targets)")
    df = pd.read_csv(f)
    df['balanced_acc'] = (df['sensitivity'] + df['specificity']) / 2
    sub = df[(df['n_samples'] == 500) & (df['run'] == 'avg')]
    cols = ['arch','target','method','auc','auc_std','sensitivity','specificity','balanced_acc']
    available = [c for c in cols if c in sub.columns]
    print(sub[available].sort_values(['arch','target','method']).to_string(index=False))

    section("2b. IMPROVED DA — WINNER PER CELL (best AUC at each n)")
    methods = ['vanilla_ft','ft_source_reg','mkmmd','coral','cc_dann']
    rows = []
    for arch in df['arch'].unique():
        for target in df['target'].unique():
            for n in sorted(df['n_samples'].unique()):
                sub2 = df[(df['arch']==arch) & (df['target']==target) &
                          (df['n_samples']==n) & (df['run']=='avg')]
                if sub2.empty: continue
                zs = df[(df['arch']==arch) & (df['target']==target) &
                        (df['method']=='zero-shot')]['auc']
                zs_auc = float(zs.iloc[0]) if len(zs) > 0 else np.nan
                best_row = sub2.loc[sub2['auc'].idxmax()]
                rows.append({
                    'arch': arch, 'target': target, 'n': n,
                    'zero_shot': round(zs_auc, 4),
                    'best_method': best_row['method'],
                    'best_auc': round(best_row['auc'], 4),
                    'delta': round(best_row['auc'] - zs_auc, 4),
                    'beats_zs': best_row['auc'] > zs_auc,
                })
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print(f"  [MISSING] {f}")

# ── 3. Extended DA results ────────────────────────────────────────────────────
f = "all_extended_da_results.csv"
if os.path.exists(f):
    section("3. EXTENDED DA RESULTS (FINGER/FOREARM/HUMERUS)")
    df = pd.read_csv(f)
    df['balanced_acc'] = (df['sensitivity'] + df['specificity']) / 2
    sub = df[(df['n_samples'] == 500) & (df['run'] == 'avg')]
    cols = ['arch','target','method','auc','sensitivity','specificity','balanced_acc']
    available = [c for c in cols if c in sub.columns]
    print(sub[available].sort_values(['arch','target','method']).to_string(index=False))
else:
    print(f"  [MISSING] {f}")

# ── 4. Transferability predictors ─────────────────────────────────────────────
try_read("transferability/transferability_predictors.csv",
         "4. TRANSFERABILITY PREDICTORS (dist_ratio, temperature, gap_ratio)")

# ── 5. Composite predictor correlations ──────────────────────────────────────
try_read("composite_predictor/composite_correlations.csv",
         "5. COMPOSITE PREDICTOR SPEARMAN CORRELATIONS")

# ── 6. DANN stability ablation ────────────────────────────────────────────────
try_read("dann_stability_ablation/densenet121/dann_stability_results.csv",
         "6. DANN STABILITY ABLATION (5 seeds, n=500, XR_ELBOW)")

# ── 7. GRAZ external validation ───────────────────────────────────────────────
section("7. GRAZ EXTERNAL VALIDATION")
graz_files = glob.glob("graz_validation/*/graz_results.csv")
if graz_files:
    dfs = []
    for f in sorted(graz_files):
        arch = f.split(os.sep)[1]
        df = pd.read_csv(f)
        df['arch'] = arch
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    cols = ['arch','graz_auc','graz_sensitivity','graz_specificity',
            'graz_fn_rate_pct','graz_high_conf_fn_pct','auc_degradation_pct']
    available = [c for c in cols if c in combined.columns]
    print(combined[available].to_string(index=False))
else:
    print("  [MISSING] graz_validation/*/graz_results.csv")

# ── 8. MC Dropout uncertainty ─────────────────────────────────────────────────
section("8. MC DROPOUT UNCERTAINTY QUANTIFICATION")
mc_files = glob.glob("mc_dropout/*/mc_dropout_results.csv")
if mc_files:
    dfs = []
    for f in sorted(mc_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    cols = ['arch','anatomy','auc_mc','mean_uncertainty',
            'unc_correct','unc_error','uncertainty_gap_ratio','acc_cov70']
    available = [c for c in cols if c in combined.columns]
    print(combined[available].to_string(index=False))
else:
    print("  [MISSING] mc_dropout/*/mc_dropout_results.csv")

# ── 9. Calibration results ────────────────────────────────────────────────────
section("9. CALIBRATION (ECE before/after temperature scaling)")
cal_files = glob.glob("calibration/*/calibration_results.csv")
if cal_files:
    dfs = []
    for f in sorted(cal_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    cols = ['arch','anatomy','ece_before','ece_after','temperature',
            'ece_reduction_pct','high_conf_error_rate','auc']
    available = [c for c in cols if c in combined.columns]
    print(combined[available].to_string(index=False))
else:
    print("  [MISSING] calibration/*/calibration_results.csv")

# ── 10. Post-DA calibration ───────────────────────────────────────────────────
section("10. POST-DA CALIBRATION (ECE after adaptation, before/after TS)")
post_cal_files = glob.glob("domain_adaptation_improved/*/*/post_da_calibration.csv")
if post_cal_files:
    dfs = []
    for f in sorted(post_cal_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    cols = ['arch','target','method','n_samples',
            'ece_post_da','ece_post_da_after_ts','post_da_temperature','ece_reduction_pct']
    available = [c for c in cols if c in combined.columns]
    print(combined[available].sort_values(
        ['arch','target','n_samples','method']).to_string(index=False))
else:
    print("  [MISSING] domain_adaptation_improved/*/*/post_da_calibration.csv")

# ── 11. Distance predictor ────────────────────────────────────────────────────
try_read("distance_predictor/predictor_table.csv",
         "11. DISTANCE PREDICTOR TABLE (anatomy tiers + best method)")

try_read("distance_predictor/distance_predictor_data.csv",
         "11b. DISTANCE PREDICTOR FULL DATA")

# ── 12. Winner consistency ────────────────────────────────────────────────────
try_read("multiarch_da_comparison/winner_consistency.csv",
         "12. MULTI-ARCH WINNER CONSISTENCY")

# ── 13. CC-DANN pseudo-label quality ─────────────────────────────────────────
section("13. CC-DANN PSEUDO-LABEL QUALITY")
pl_files = glob.glob("domain_adaptation_improved/*/cc_dann_pseudo_label_quality.csv")
if pl_files:
    dfs = []
    for f in sorted(pl_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    print(combined.to_string(index=False))
else:
    print("  [MISSING] cc_dann_pseudo_label_quality.csv files")

# ── 14. Feature distance ratios ───────────────────────────────────────────────
section("14. FEATURE DISTANCE RATIOS (per arch x anatomy)")
dist_files = glob.glob("feature_analysis/*/distance_ratios.csv")
if dist_files:
    dfs = []
    for f in sorted(dist_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    print(combined.to_string(index=False))
else:
    print("  [MISSING] feature_analysis/*/distance_ratios.csv")

# ── 15. Grad-CAM stats ────────────────────────────────────────────────────────
section("15. GRAD-CAM ATTENTION STATS")
gc_files = glob.glob("gradcam/*/gradcam_stats.csv")
if gc_files:
    dfs = []
    for f in sorted(gc_files):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    print(combined.to_string(index=False))
else:
    print("  [MISSING] gradcam/*/gradcam_stats.csv")

print("\n" + "="*70)
print("SUMMARY COMPLETE")
print("="*70)
print(f"\nOutput saved — upload summary_output.txt to Claude for analysis.")