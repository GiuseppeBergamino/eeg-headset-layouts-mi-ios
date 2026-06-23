"""MOABB benchmark script for the headset-layout motor imagery study."""

import moabb
import mne
import sklearn
import math
import pandas as pd

from moabb.datasets import Cho2017
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.metrics import make_scorer, f1_score

from datetime import datetime
import time
from pathlib import Path

from headset_profiles import HEADSETS, make_no_occipital_mask, print_catalog

import sys


# ------------------ UTILITIES ------------------

# Select the number of CSP components as a function of channel count. 
def choose_even_csp_components(n_ch: int, cap: int = 8, min_comp: int = 2) -> int:
    """
    n_components = round(sqrt(n_ch)), forced to an even value,
    with constraints: >= min_comp, <= n_ch, <= cap.
    This preserves the MOABB reference choice of 8 CSP components for 64 channels.
    """
    n = int(round(math.sqrt(n_ch)))

    if n % 2 == 1: # if odd
        n += 1.    # make it even

    n = max(min_comp, n) # enforce minimum number of components
    n = min(n, cap, n_ch) # enforce upper bounds

    return int(n)

def mean_std_str(scores: pd.Series) -> str: # mean±std formatting for summary tables
    m = scores.mean() * 100
    s = scores.std(ddof=1) * 100
    return f"{m:.2f} ± {s:.2f}"

# MOABB does not expose macro-F1 as the default MI scoring metric.
# We subclass the paradigm and override the scoring property.
F1_MACRO = make_scorer(f1_score, average = "macro") # compute F1 per class and then average
class MotorImageryF1(MotorImagery):
    @property
    def scoring(self):
        return F1_MACRO
    
# Reduce logging verbosity.
moabb.set_log_level("ERROR")     # alternatively: "WARNING"
mne.set_log_level("CRITICAL")    # most restrictive MNE logging level

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Pipeline instance is not fitted yet.*",
)




# ------------------ OUTPUT DIRECTORIES ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR_CSV = PROJECT_ROOT / "results" / "within_session"
DIR_CSV.mkdir(parents=True, exist_ok=True)

csv_summary_path = DIR_CSV / "__summary.csv"


# ------------------ EXP DEFINES ------------------
dataset = Cho2017()
print("--------------------------")
print("\n\nDataset:", dataset.code)
# Load the first subject only to retrieve the available EEG channel names.
subject = dataset.subject_list[0]
data1 = dataset.get_data(subjects=[subject])
subj_id = next(iter(data1))
sess_id = next(iter(data1[subj_id]))
run_id  = next(iter(data1[subj_id][sess_id]))

rawEEG = data1[subj_id][sess_id][run_id].copy().pick('eeg') # keep EEG channels only
available_channels = rawEEG.ch_names

# ------------------ HEADSET MASK SELECTION ------------------
print_headset_catalog = False
if print_headset_catalog:
    print(print_catalog())
# FULL_CHO, 
# EMOTIV_FLEX, EMOTIV_EPOCH_X, EMOTIV_INSIGHT, EMOTIV_MN8, 
# ULTRACORTEX_16_MILIMB, ULTRACORTEX_16, ULTRACORTEX_8, 
# GTEC_UNICORN_HYBRID, 
# BITBRAIN_DIADEM, BITBRAIN_AIR, BITBRAIN_HERO, BITBRAIN_IKON, 
# MUSE_HEADBAND, 
# NEUROSITY_CROWN,
# CGX_QUICK_20R, CGX_QUICK_32R,
# MINDROVE_VISION, MINDROVE_LUCID, MINDROVE_ARC, MINDROVE_BRIGHT,
# BRAINBIT_DRAGON, BRAINBIT_HEADPHONES, BRAINBIT_HEADBAND_PRO, BRAINBIT_HEADBAND
# NEEURO_SENZEBAND
# ABM_B_ALERT_X24, ABM_B_ALERT_X10
# NEUROSKY_MINDWAVE_2
# WS_DSI_24, WS_DSI_7, WS_DSI_VR300, WS_DSI_VRVEP

NO_OCCIPITAL = False
headset = HEADSETS["WS_DSI_VRVEP"]  # <-- default headset choice 
mask_name = headset.name     
channel_mask = headset.make_mask(available_channels, verbose = True)
n_channels = len(channel_mask)  # compute before CSP component selection

if NO_OCCIPITAL:
    channels_before = len(channel_mask)
    channel_mask = make_no_occipital_mask(channel_mask)
    channels_after = len(channel_mask)
    if channels_before == channels_after:
        print("\nNo occipital channels found in the selected mask.")
        print("Stopping execution.")
        sys.exit(0) # stop because the NO_OCCIPITAL variant is identical to the original mask
    else:
        mask_name = f"{mask_name}_NO_OCC"
        print("Mask after removing occipital channels:", channels_after)
    

#####################------------- PIPELINES ----------------################
seed = 23
moabb.setup_seed(seed)
# Select n_components: even number with cap=8.
n_csp_components = choose_even_csp_components(n_channels) 

pipelines = {} # Dictionary of pipelines passed to WithinSessionEvaluation.process.

#Common Spatial Pattern and Linear Discriminant Analysis
pipelines["CSP+LDA"] = make_pipeline(
    CSP(n_components = n_csp_components,
        reg = None, 
        log = True, 
        cov_est = "concat"),
    LinearDiscriminantAnalysis(solver = "svd")
)
# Common Spatial Patterns and Support Vector Machine classifier.
pipelines["CSP+SVM"] = make_pipeline(
    CSP(n_components = n_csp_components, 
        reg = None, 
        log = True, 
        cov_est = "concat"),
    SVC(kernel = "linear", # linear support vector classifier
        C = 1.0 #regularisation parameter
        ) 
)

# Riemannian tangent-space pipelines (Covariances -> TangentSpace -> classifier).
# Support Vector Machine.
pipelines["TS+SVM"] = make_pipeline(
        Covariances(estimator = "oas"),
        TangentSpace(metric = "riemann"),
        SVC(kernel = "linear", 
            C = 1.0
            )
    )
# Logistic Regression.
pipelines["TS+LR"] =  make_pipeline(
        Covariances(estimator = "oas"),
        TangentSpace(metric = "riemann"),
        LogisticRegression(max_iter = 2000),
    )
# Elastic net.
pipelines["TS+EL"] = make_pipeline(
    Covariances(estimator = "oas"),
    TangentSpace(metric = "riemann"),
    LogisticRegression(
            solver = "saga",
            penalty = "elasticnet",
            l1_ratio = 0.5,  
            max_iter = 4000
             )
    )

pipeline_name = "multi_pipes"        # used only for the experiment identifier

####################---------------- START --------------#################
start_time = datetime.now()
t0 = time.perf_counter()

exp_id = f"{start_time:%Y%m%d_%H%M}_{mask_name}_{pipeline_name}"
print("\nSTART experiment:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("EXP_ID:", exp_id)
print("Number of CSP components:", n_csp_components)
print(" ")

# ------------------ PARADIGMS with ROC/AUC (default) + F1 macro (custom) ------------------
paradigm_auc = MotorImagery(
    n_classes = 2, #with 2 classes default ROC-AUC 
    events = ["left_hand", "right_hand"],
    fmin = 8.0, 
    fmax = 32.0,
    tmin = 0.0,
    tmax = 3.0,
    baseline = None, 
    channels = channel_mask,  # <-- mask enters here
    resample = None
)

paradigm_f1 = MotorImageryF1( #custom f1 macro scoring
    n_classes = 2, 
    events = ["left_hand", "right_hand"],
    fmin = 8.0,
    fmax = 32.0,
    tmin = 0.0,
    tmax = 3.0,
    baseline = None,
    channels = channel_mask,   # <-- same mask here
    resample = None
)

# ------------------ Evaluations ------------------#
#------- RUN ROC-AUC (default)
eval_auc = WithinSessionEvaluation(
    paradigm = paradigm_auc,
    datasets = [dataset],
    random_state = seed,
    n_jobs = 1,
    overwrite = True,  # overwrite existing HDF5 file
    suffix = f"{exp_id}_AUC", # goes in MNE_DATA/results/id_AUC.hdf5"
)

res_auc = eval_auc.process(pipelines) #evaluation

#-------- RUN F1 (custom scoring)
eval_f1 = WithinSessionEvaluation(
    paradigm = paradigm_f1,
    datasets = [dataset],
    random_state = seed,
    n_jobs = 1, 
    overwrite = True,
    suffix = f"{exp_id}_F1",
)

res_f1 = eval_f1.process(pipelines) #evaluation
# -------------------- SAVE RAW RESULT FILES
auc_raw_path = DIR_CSV / f"{exp_id}_AUC.csv"
res_auc.to_csv(auc_raw_path, index = False, float_format = "%.6f")
print("AUC results saved!\n")

f1_raw_path = DIR_CSV / f"{exp_id}_F1.csv"
res_f1.to_csv(f1_raw_path, index = False, float_format = "%.6f")
print("F1 results saved!")


# -------------------- UPDATE AGGREGATED SUMMARY CSV
summary_row = {
    "Exp_id": exp_id,
    "Mask": mask_name,
    "Price": headset.price,
    "N_ch": len(channel_mask),
}

# Column fixed order for summary
wanted_pipes = ["CSP+LDA", "CSP+SVM", "TS+LR", "TS+SVM", "TS+EL"] #for summary

# col order: AUC pipeline1, F1 pipeline1, AUC pipeline2, F1 pipeline2, ...
metric_cols = []
for p in wanted_pipes:
    col_auc = f"{p}_AUC"
    col_f1  = f"{p}_F1"
    metric_cols += [col_auc, col_f1]

    sub_auc = res_auc[(res_auc["dataset"] == dataset.code) & (res_auc["pipeline"] == p)]
    sub_f1  = res_f1[(res_f1["dataset"] == dataset.code) & (res_f1["pipeline"] == p)]

    summary_row[col_auc] = mean_std_str(sub_auc["score"]) if len(sub_auc) else ""
    summary_row[col_f1]  = mean_std_str(sub_f1["score"])  if len(sub_f1)  else ""

new_row = pd.DataFrame([summary_row])

if csv_summary_path.exists():
    df = pd.read_csv(csv_summary_path)
    df = pd.concat([df, new_row], ignore_index = True)
    df = df.drop_duplicates(subset = ["Exp_id"], keep = "last")
else:
    df = new_row

# general column ordering
col_order = ["Exp_id", "Mask", "Price", "N_ch"] + metric_cols
for c in col_order:
    if c not in df.columns:
        df[c] = ""
df = df[col_order]

df.to_csv(csv_summary_path, index = False)
print("\nSummary table updated!")


# ------------------ TIMING ------------------
elapsed_s = time.perf_counter() - t0
print(f"\nElapsed time: {elapsed_s:.2f} s ({elapsed_s/60:.2f} min)")
