from codeflowlm.data import get_changes_from_features
from codeflowlm.train import execute_command
import pickle
import os

def test(batch_classifier_dir, path, full_changes_train_file, full_changed_valid_file, full_changes_test_file, project, features_test, model_path, th, pretrained_model, 
         calculate_metrics=True, peft_alg="lora", eval_metric='f1', batch_size=16):
  changes_test = get_changes_from_features(full_changes_train_file, full_changed_valid_file, full_changes_test_file, features_test, do_test=True)
  with open(f"{path}/changes_test_online_{project}.pkl", "wb") as f:
    pickle.dump(changes_test, f)

  with open(f"{path}/features_test_online_{project}.pkl", "wb") as f:
    pickle.dump(features_test, f)

  print(f"Testing with recent data with th = {th}...")
  command = f"""
  python {batch_classifier_dir}PEFT4CC/just-in-time/run_{peft_alg}.py \
   --test_data_file {path}/changes_test_online_{project}.pkl {path}/features_test_online_{project}.pkl \
   --output_dir {model_path} \
   --pretrained_model {pretrained_model} \
   --batch_size {batch_size} \
   --do_test \
   --threshold {th} \
   --eval_metric {eval_metric} \
   """

  if peft_alg == "lora":
    command += "--use_lora "

  if pretrained_model == 'codet5p-770m':
    command += "--hidden_size 1024 "

  if calculate_metrics:
    command += """--calculate_metrics """

  execute_command(command)

  results = None

  if os.path.exists("results.pkl"):
    with open(f"results.pkl", "rb") as f:
      results = pickle.load(f)

    os.remove("results.pkl")

  with open(f"predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

  os.remove("predictions.pkl")

  return results, predictions