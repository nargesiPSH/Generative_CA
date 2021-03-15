#!/usr/bin/env python
# coding: utf-8

# # OCSVM Baseline Model
# *updated by Narges Pourshahrokhi, 2021*
# 
# **Description**   
# 
# Reimplemenation of a OCSVM approach to Continuous Authentication described by [1] and baseline model implmenattion by [2]. This code is modified version of [2] for perpuse of using geneartive model in CA system 
# 
# **Purpose**
# 
# - Get basic idea about authentication performance using raw data
# - Verify results of [1]
# 
# **Data Sources**   
# 
# - [H-MOG Dataset](http://www.cs.wm.edu/~qyang/hmog.html)  
#   (Downloaded beforehand using  [./src/data/make_dataset.py](./src/data/make_dataset.py), stored in [./data/external/hmog_dataset/](./data/external/hmog_dataset/) and converted to [./data/processed/hmog_dataset.hdf5](./data/processed/hmog_dataset.hdf5))
# 
# **References**   
# 
# [1] Centeno, M. P. et al. (2018): Mobile Based Continuous Authentication Using Deep Features. Proceedings of the 2^nd International Workshop on Embedded and Mobile Deep Learning (EMDL), 2018, 19-24.
# [2]  Holger Buech, https://github.com/dynobo/ContinAuth
# **Table of Contents**
# 
# **1 - [Preparations](#1)**  
# 1.1 - [Imports](#1.1)  
# 1.2 - [Configuration](#1.2)  
# 1.3 - [Experiment Parameters](#1.3)  
# 1.4 - [Select Approach](#1.4)  
# 
# **2 - [Data Preparations](#2)**  
# 2.1 - [Load Dataset](#2.1)  
# 2.2 - [Normalize Features (if global)](#2.2)  
# 2.3 - [Split Dataset for Valid/Test](#2.3)  
# 2.4 - [Check Splits](#2.4)  
# 2.5 - [Reshape Features](#2.5)  



# Standard
from pathlib import Path
import os
import sys
import dataclasses
import math
import warnings

# Extra
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import statsmodels.stats.api as sms
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from utils import utils_set_output_style
from utils import utils_custom_scale
from utils import utils_split_report
from utils import utils_reshape_features
from utils import utils_save_plot
from utils import utils_ppp
from utils import utils_eer
from utils import utils_plot_distance_hist
from utils import utils_plot_training_loss
from utils import utils_generate_cv_scenarios
from utils import utils_create_cv_splits
from utils import utils_cv_report
from utils import utils_plot_randomsearch_results
from utils import utils_plot_acc_eer_dist
from utils import utils_plot_training_delay
from utils import utils_plot_detect_delay
from IPython import get_ipython
ipy = get_ipython()

# Custom `DatasetLoader`class for easier loading and subsetting data from the datasets.
parrents_path_str = "/vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/src/utility/"
 
module_path = Path(parrents_path_str ) #os.path.abspath(os.path.join(".."))  # supposed to be parent folder
#if module_path not in sys.path:
#    print("not in sys path :(")
#    sys.path.append(module_path)
#from src.utility.dataset_loader_hdf  import DatasetLoader
#sys.path.insert(1, parrents_path_str)
from srcc.dataset_loader_hdf5 import DatasetLoader
#"/vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/src/utility/dataset_loader_hdf5.py"
# src.utility.dataset_loader_hdf5
# Global utitlity functions are in separate notebook
#get_ipython().run_line_magic('run', 'utils.ipynb')


# ### 1.2 Configuration <a id='1.2'>&nbsp;</a>

# In[10]:


# Various Settings
SEED = 712  # Used for every random function
hmog_path_str = "/vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/data/processed/hmog_dataset.hdf5"
HMOG_HDF5 = Path(hmog_path_str)
EXCLUDE_COLS = ["sys_time"]
CORES = -1

# For plots and CSVs

out_path_str = "/vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/notebooks/output/chapter-6-1-3-ocsvm"
OUTPUT_PATH = Path(out_path_str)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


report_path_str = "/vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/reports/figures"
REPORT_PATH = Path(report_path_str) # Figures for thesis

# Plotting
#get_ipython().run_line_magic('matplotlib', 'inline')
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
utils_set_output_style()


# In[11]:


# Workaround to remove ugly spacing between progress bars
#HTML("<style>.p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty{padding: 0;border: 0;} div.output_subarea{padding:0;}</style>")


# In[12]:


### 1.3 Experiment Parameters <a id='1.3'>&nbsp;</a> 
#Selection of parameters set that had been tested in this notebook. Select one of them to reproduce results.


# In[13]:


@dataclasses.dataclass
class ExperimentParameters:
    """Contains all relevant parameters to run an experiment."""

    name: str  # Name of Parameter set. Used as identifier for charts etc.
    frequency: int
    max_subjects: int
    max_test_subjects: int
    seconds_per_subject_train: float
    seconds_per_subject_test: float
    task_types: list  # Limit scenarios to [1, 3, 5] for sitting or [2, 4, 6] for walking, or don't limit (None)
    window_size: int  # After resampling
    step_width: int  # After resampling
    scaler: str  # {"std", "robust", "minmax"}
    scaler_scope: str  # {"subject", "session"}
    scaler_global: bool  # fit transform scale on all data (True) or fit on training only (False)
    ocsvm_nu: float  # Best value found in random search, used for final model
    ocsvm_gamma: float  # Best value found in random search, used for final model
    feature_cols: list  # Columns used as features
    exclude_subjects: list  # Don't load data from those users
        
    # Calculated values
    def __post_init__(self):
        # HDF key of table:
        self.table_name = f"sensors_{self.frequency}hz"

        # Number of samples per _session_ used for training:
        self.samples_per_subject_train = math.ceil(
            (self.seconds_per_subject_train * 100)
            / (100 / self.frequency)
            / self.window_size
        )

        # Number of samples per _session_ used for testing:
        self.samples_per_subject_test = math.ceil(
            (self.seconds_per_subject_test * 100)
            / (100 / self.frequency)
            / self.window_size
        )

        

# INSTANCES
# ===========================================================

# NAIVE_APPROACH
# -----------------------------------------------------------
NAIVE_MINMAX_OCSVM = ExperimentParameters(
    name="NAIVE-MINMAX_OCSVM",
    frequency=100,
    max_subjects=90,
    max_test_subjects=30,
    seconds_per_subject_train=67.5,
    seconds_per_subject_test=67.5,    
    task_types=None,
    window_size=50,
    step_width=50,
    scaler="minmax",
    scaler_scope="subject",
    scaler_global=True,
    ocsvm_nu=0.086,
    ocsvm_gamma=0.091,
    feature_cols=[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ],
    exclude_subjects=[
        "733162",  # No 24 sessions
        "526319",  # ^
        "796581",  # ^
        "539502",  # Least amount of sensor values
        "219303",  # ^
        "737973",  # ^
        "986737",  # ^
        "256487",  # Most amount of sensor values
        "389015",  # ^
        "856401",  # ^
    ],
)

# VALID_APPROACH
# -----------------------------------------------------------
VALID_MINMAX_OCSVM = dataclasses.replace(
    NAIVE_MINMAX_OCSVM,
    name="VALID-MINMAX-OCSVM",
    scaler_global=False,
    ocsvm_nu=0.165,
    ocsvm_gamma=0.039,
)

# NAIVE_ROBUST_APPROACH
# -----------------------------------------------------------
NAIVE_ROBUST_OCSVM = dataclasses.replace(
    NAIVE_MINMAX_OCSVM,
    name="NAIVE-ROBUST-OCSVM",
    scaler="robust",
    scaler_global=True,
    ocsvm_nu=0.153,
    ocsvm_gamma=0.091,  # below median, selected by chart
)

# ROBUST_APPROACH (VALID)
# -----------------------------------------------------------
VALID_ROBUST_OCSVM = dataclasses.replace(
    NAIVE_MINMAX_OCSVM,
    name="VALID-ROBUST-OCSVM",
    scaler="robust",
    scaler_global=False,
    ocsvm_nu=0.098,
    ocsvm_gamma=0.003,
)


# ### 1.4 Select approach <a id='1.4'>&nbsp;</a> 
# Select the parameters to use for current notebook execution here!

# In[14]:


P = VALID_ROBUST_OCSVM


# **Overview of current Experiment Parameters:**

# In[15]:


#utils_ppp(P)


# ## 2. Data Preparation <a id='2'>&nbsp;</a> 

# ### 2.1 Load Dataset <a id='2.1'>&nbsp;</a> 

# In[16]:


hmog = DatasetLoader(
    hdf5_file=HMOG_HDF5,
    table_name=P.table_name,
    max_subjects=P.max_subjects,
    task_types=P.task_types,
    exclude_subjects=P.exclude_subjects,
    exclude_cols=EXCLUDE_COLS,
    seed=SEED,
)

hmog.data_summary()


# ### 2.2 Normalize features (if global) <a id='2.2'>&nbsp;</a> 
# Used here for naive approach (before splitting into test and training sets). Otherwise it's used during generate_pairs() and respects train vs. test borders.

# In[17]:


if P.scaler_global:
    print("Normalize all data before splitting into train and test sets...")
    hmog.all, _ = utils_custom_scale(
        hmog.all,
        scale_cols=P.feature_cols,        
        feature_cols=P.feature_cols,
        scaler_name=P.scaler,
        scope=P.scaler_scope,
        plot=True,
    )
else:
    print("Skipped, normalize after splitting.")


# ### 2.3 Split Dataset for Valid/Test <a id='2.3'>&nbsp;</a> 
# In two splits: one used during hyperparameter optimization, and one used during testing.
# 
# The split is done along the subjects: All sessions of a single subject will either be in the validation split or in the testing split, never in both.

# In[18]:


hmog.split_train_test(n_test_subjects=P.max_test_subjects)
hmog.data_summary()


# ### 2.4 Check Splits <a id='2.4'>&nbsp;</a> 

# In[ ]:


utils_split_report(hmog.train)


# In[ ]:


utils_split_report(hmog.test)


# ### 2.5 Reshape Features  <a id='2.5'>&nbsp;</a> 

# **Reshape & store Set for Validation:**

# In[ ]:


df_train_valid = utils_reshape_features(
    hmog.train,
    feature_cols=P.feature_cols,
    window_size=P.window_size,
    step_width=P.step_width,
)

# Clean memory
del hmog.train
if ipy is not None:
    ipy.run_line_magic('reset_selective', '-f hmog.train')
#get_ipython().run_line_magic('reset_selective', '-f hmog.train')

print("Validation data after reshaping:")
#display(df_train_valid.head())

# Store iterim data
df_train_valid.to_msgpack(OUTPUT_PATH / "df_train_valid.msg")

# Clean memory
if ipy is not None:
    ipy.run_line_magic('reset_selective', '-f df_train_valid')
#get_ipython().run_line_magic('reset_selective', '-f df_train_valid')


# **Reshape & store Set for Testing:**

# In[ ]:


df_train_test = utils_reshape_features(
    hmog.test,
    feature_cols=P.feature_cols,
    window_size=P.window_size,
    step_width=P.step_width,
)

del hmog.test
if ipy is not None:
    ipy.run_line_magic('reset_selective', '-f hmog.test')
#get_ipython().run_line_magic('reset_selective', '-f hmog.test')

print("Testing data after reshaping:")
#display(df_train_test.head())

# Store iterim data
df_train_test.to_msgpack(OUTPUT_PATH / "df_train_test.msg")

# Clean memory
if ipy is not None:
    ipy.run_line_magic('reset_selective', '-f df_train_test')
#get_ipython().run_line_magic('reset_selective', '-f df_train_test')


# In[ ]:


# Clean Memory
if ipy is not None:
    ipy.run_line_magic('reset_selective', '-f df_')
