# Reproduction package for the paper "CodeFlowLM: Incremental Just-In-Time Defect Prediction with Pretrained Language Models and Exploratory Insights into Defect Localization"

## Replication steps

### CodeFlowLM - Within-Project (WP) Setting

Hardware requirements: A100 or other GPU with at least 24GB OF VRAM for CodeT5+ with batch size = 16. This notebook was tested with A100 GPU on Google Colab environment. So, the instructions in this replication package are well suited to a Colab env.


1. Upload or open in Colab [continual_jitsdp/CodeFlowLM-WP.ipynb](continual_jitsdp/CodeFlowLM-WP.ipynb). This file runs a single execution for each of 19 projects with a default/fixed random seed. 
2. Change execution environment to A100 GPU or equivalent, if you wish ro train CodeFlowLM with CodeT5+ base learner.  If you prefer UniXCoder, T4 GPU is sufficient.  Notice: it may be necessary to buy credits.
3. Run the notebook in Colab environment.  It will download the necessary repositories with code and data, including this repository.

   
