# CodeFlowLM: Incremental Just-In-Time Defect Prediction with Pretrained Language Models and Exploratory Insights into Defect Localization

## Replication steps

### CodeFlowLM (RQ 1)

Hardware requirements: A100 or other GPU with at least 24GB OF VRAM for CodeT5+ with batch size = 16. This notebook was tested with A100 GPU on Google Colab environment. So, the instructions in this replication package are well suited to a Colab env.

1. Upload or open in Colab [continual_jitsdp/CodeFlowLM-WP.ipynb](continual_jitsdp/CodeFlowLM-WP.ipynb) for within-project (WP) learning, and [continual_jitsdp/CodeFlowLM-CP.ipynb](continual_jitsdp/CodeFlowLM-CP.ipynb) for cross-project (CP) learning. Each file runs a single execution for each of 19 projects with a default/fixed random seed and different combinations of base learner (CodeT5+/UniXCoder) + fine-tuning technique (LoRA/Preffix fine tuning - PreT). In current version, the tested combinations include CodeT5+ + LoRA, UniXCoder + PreT and UniXCoder + LoRA (tested only in WP setting).
2. Change execution environment to A100 GPU or equivalent, if you wish ro train CodeFlowLM with CodeT5+ base learner.  If you prefer UniXCoder, T4 GPU is sufficient.  Notice: it may be necessary to buy credits.
3. Run the notebook in Colab environment.  It will download the necessary repositories with code and data, including this repository.

The following random seeds were used to run the experiments repetitions: [33 (default), 23, 99, 72, 22].  To fix the seed, add seed parameter to codeflowlm.train.train_project_with_lat_ver() function.

#### Inference time measurements

[continual_jitsdp/PredictionTimesTest.ipynb](continual_jitsdp/PredictionTimesTest.ipynb): tests CodeFlowLM inference time for a single prediction with both base learners (CodeT5+ and UniXCoder).

### Defect Localization (RQs 2 and 3)

- [jit-dl-llms/JIT-DL-GPT-5.ipynb](jit-dl-llms/JIT-DL-GPT-5.ipynb): Quantitative and qualitative experiments with GPT-5, the LLM with the best observed metrics in our experiments.
- response-gpt-5_exec-*.pkl files: Different runs for the same LLM

#### False-positives analysis

- Prompt: "I am attaching a .json file containing a list of false positives, i.e., lines classified as defective when they are actually non-defective. Analyze the file and return a list of the most common types of false positives, explicitly indicating the corresponding JSON excerpts and their respective projects."
- File: [jit-dl-llms/fps.json](jit-dl-llms/fps.json) 

#### False-negatives analysis

- Prompt: "The attached .txt file contains a list of false negatives, i.e., lines classified as clean when they are actually defective. Analyze the file and return a list of the most common types of false negatives."
- File: [jit-dl-llms/fns.txt](jit-dl-llms/fns.txt) 
   
