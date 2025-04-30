from typing import Dict, List
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# load once at import‐time
_cancer = load_breast_cancer()

# # Paths
# DATA_PATH = "../data/DataSet_Combined_SI_SNI_Baseline_FE.csv"

# Configuration
DEPTHS = list(range(3, 13))
MIN_SAMPLES = [3, 5, 10, 30, 50]
NUM_BOOTSTRAPS = 20


# Feature sets for different tasks
FEATURE_SETS: Dict[str, List[str]] = {
    "suicidea": [
        "age", "gender", "sexori", "raceall", "trauma_sum", "cesd_score", "harddrug_life", "school", "degree", "job", "sex", "concurrent", "exchange", "children", "weapon", "fight", "fighthurt", "ipv", "ptsd_score", "alcfirst", "potfirst", "staycurrent", "homelage", "time_homeless_month", "jail", "jailkid", "gettherapy", "sum_alter", "sum_family", "sum_home_friends", "sum_street_friends", "sum_unknown_alter", "sum_talk_once_week", "sum_alter3close", "prop_family_harddrug", "prop_friends_harddrug", "prop_friends_home_harddrug", "prop_friends_street_harddrug", "prop_alter_all_harddrug", "prop_enc_badbehave", "prop_alter_homeless", "prop_family_emosup", "prop_friends_emosup", "prop_friends_home_emosup", "prop_friends_street_emosup", "prop_alter_all_emosup", "prop_family_othersupport", "prop_friends_othersupport", "prop_friends_home_othersupport", "prop_friends_street_othersupport", "prop_alter_all_othersupport", "sum_alter_staff", "prop_object_badbehave", "prop_enc_goodbehave", "prop_alter_school_job", "sum_alter_borrow"],
    "suicattempt": [
        "age", "gender", "sexori", "raceall", "trauma_sum", "cesd_score", "harddrug_life", "school", "degree", "job", "sex", "concurrent", "exchange", "children", "weapon", "fight", "fighthurt", "ipv", "ptsd_score", "alcfirst", "potfirst", "staycurrent", "homelage", "time_homeless_month", "jail", "jailkid", "gettherapy", "sum_alter", "prop_family", "prop_home_friends", "prop_street_friends", "prop_unknown_alter", "sum_talk_once_week", "sum_alter3close", "prop_family_harddrug", "prop_friends_harddrug", "prop_friends_home_harddrug", "prop_friends_street_harddrug", "prop_alter_all_harddrug", "prop_enc_badbehave", "prop_alter_homeless", "prop_family_emosup", "prop_friends_emosup", "prop_friends_home_emosup", "prop_friends_street_emosup", "prop_alter_all_emosup", "prop_family_othersupport", "prop_friends_othersupport", "prop_friends_home_othersupport", "prop_friends_street_othersupport", "prop_alter_all_othersupport", "sum_alter_staff", "prop_object_badbehave", "prop_enc_goodbehave", "prop_alter_school_job", "sum_alter_borrow"],
    
    "target": list(_cancer.feature_names)
}

#data directory:
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"

# Model parameters for different tasks
MODEL_PARAMS = {
    "suicidea": dict(min_samples_leaf=10, min_samples_split=20, max_depth=4),
    "suicattempt": dict(min_samples_leaf=10, min_samples_split=30, max_depth=4),
    "target": dict(min_samples_leaf=10, min_samples_split=30, max_depth=4),
}

# Labels
LABELS = ["suicidea", "suicattempt", "target"]

# Random seed
RANDOM_SEED = 42