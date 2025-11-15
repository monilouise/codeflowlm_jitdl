import pandas as pd
import pickle
import os
from datetime import datetime
import time
import traceback
import shutil
from codeflowlm.command import execute_command
from codeflowlm.data import get_ord_cross_changes_full, get_df_features_full
from codeflowlm.latency_verification import add_first_fix_date, do_latency_verification, do_real_latency_verification, process_buggy_commit
from codeflowlm.prequential_metrics import calculate_prequential_mean_and_std
from codeflowlm.plots import plot
from codeflowlm.test import test
from codeflowlm.threshold import calculate_th_from_test
from time import time

projects_with_real_lat_ver = ['ant-ivy','commons-bcel','commons-beanutils',
                                'commons-codec','commons-collections',
                                'commons-compress','commons-configuration',
                                'commons-digester',
                                'commons-jcs','commons-lang','commons-math',
                                'commons-net','commons-scxml',
                                'commons-validator','commons-vfs','gora',
                                'parquet-mr', 'giraph', 'opennlp']

batches = []

#From a given df_train (a segment from the stream), a training pool, a training
#queue and a buggy pool, move rows to/from training queue/training pool/buggy
#pool.
def prepare_train_data(df_train, training_pool, training_queue,
                       map_commit_to_row, buggy_pool, do_real_lat_ver=False):
  last_timestamp = 0

  for _, row in df_train.iterrows():
    assert row['author_date_unix_timestamp'] >= last_timestamp
    last_timestamp = row['author_date_unix_timestamp']

    if row['is_buggy_commit'] == 1:
      if do_real_lat_ver:
        process_buggy_commit(row, training_queue, map_commit_to_row, buggy_pool)
      else:
        training_pool.append(row)
    else:
      training_queue.append((row['commit_hash'], row['author_date_unix_timestamp']))
      map_commit_to_row[row['commit_hash']] = row

    if do_real_lat_ver:
      do_real_latency_verification(row, training_pool, training_queue,
                                  map_commit_to_row, buggy_pool)
    else:
      do_latency_verification(row, training_pool, training_queue,
                            map_commit_to_row)

  return last_timestamp

def is_valid_training_data(training_pool):
  df = pd.DataFrame(training_pool)

  #Mudança 27/06/2025 -> relaxando restrição para split de validação
  #07/07/2025: pelo menos um exemplo positivo e um exemplo negativo
  return df.shape[0] > 0 and df['is_buggy_commit'].sum() >= 1 and df['is_buggy_commit'].sum() < df.shape[0]

#Mudança 12/08/2025: Setei do_eval_with_all_negative=True para que a validação seja feita com a loss de validação caso não haja exemplo
#de validação positivo
def prepare_training_data(path, full_changes_train_file, full_changed_valid_file, full_changes_test_file, project, df, 
                          do_eval_with_all_negative=True):
  if 'first_fix_date' in df.columns and 'fixes' in df.columns:
    df = df.drop(columns=['first_fix_date', 'fixes'])

  df = df.reset_index()
  commits = []
  labels = []
  commit_messages = []
  codes = []
  
  ord_cross_changes_full = get_ord_cross_changes_full(full_changes_train_file, full_changed_valid_file, full_changes_test_file)

  for _, row in df.iterrows():
    commit_id = row['commit_hash']
    idx = ord_cross_changes_full[0].index(commit_id)
    label = row['is_buggy_commit']
    commit_message = ord_cross_changes_full[2][idx]
    code = ord_cross_changes_full[3][idx]
    commits.append(commit_id)
    labels.append(label)
    commit_messages.append(commit_message)
    codes.append(code)

  assert len(commits) == len(labels) == len(commit_messages) == len(codes) == df.shape[0]
  train_size = int(0.9 * df.shape[0])
  print("Training size = ", train_size)
  val_size = df.shape[0] - train_size
  print("Validation size = ", val_size)
  train_df = df.iloc[:train_size]
  train_bug_ratio = train_df["is_buggy_commit"].sum()/train_df.shape[0]
  print("Training bug ratio = ", train_bug_ratio)
  print("Negative instances newest date:")
  print(train_df[train_df["is_buggy_commit"] == 0]["author_date"].max())

  val_df = df.iloc[train_size:]
  val_bug_ratio = val_df["is_buggy_commit"].sum()/val_df.shape[0]
  print("Validation bug ratio = ", val_bug_ratio)

  print("df['is_buggy_commit'].sum() = ", df['is_buggy_commit'].sum())

  if df['is_buggy_commit'].sum() >= 2 and (train_bug_ratio == 0 or val_bug_ratio == 0):
    #Muda a divisão dos dados de forma que o conjuntp de validação contenha ao menos o último exemplo positivo
    condition = (df['is_buggy_commit'] == 1) | (df['is_buggy_commit'] == 1.0)
    indices = df[condition].index.tolist()
    val_start = indices[-1]
    val_end = val_start + val_size
    train_df = pd.concat([df.iloc[:val_start], df.iloc[val_end:]])
    val_df = df.iloc[val_start:val_end]

    train_commits = commits[:val_start] + commits[val_end:]
    train_labels = labels[:val_start] + labels[val_end:]
    train_commit_messages = commit_messages[:val_start] + commit_messages[val_end:]
    train_codes = codes[:val_start] + codes[val_end:]
    val_commits = commits[val_start:val_end]
    val_labels = labels[val_start:val_end]
    val_commit_messages = commit_messages[val_start:val_end]
    val_codes = codes[val_start:val_end]
    assert train_df.shape[0] == len(train_commits) == len(train_labels) == len(train_commit_messages) == len(train_codes)
    assert val_df.shape[0] == len(val_commits) == len(val_labels) == len(val_commit_messages) == len(val_codes)
    changes_train = (train_commits, train_labels, train_commit_messages, train_codes)
    changes_valid = (val_commits, val_labels, val_commit_messages, val_codes)

    train_bug_ratio = train_df["is_buggy_commit"].sum()/train_df.shape[0]
    print("Adjusted training bug ratio = ", train_bug_ratio)

    val_bug_ratio = val_df["is_buggy_commit"].sum()/val_df.shape[0]
    print("Adjusted validation bug ratio = ", val_bug_ratio)

    val_clean_ratio = val_df[val_df["is_buggy_commit"] == 0]["is_buggy_commit"].count()/val_df.shape[0]
    print("Adjusted validation clean ratio = ", val_clean_ratio)

    #Mudança 27/06/2025 -> relaxando restrição para split de validação
    #if val_clean_ratio == 0:
    #  raise Exception("Not enough validation data.")
  elif df['is_buggy_commit'].sum() == 1 and train_bug_ratio == 0:
    #Muda a divisão dos dados de forma que o conjuntp de validação contenha ao menos o segundo exemplo positivo
    #Mudança 27/06/2025 -> relaxando restrição para split de validação
    #Garante que o conjunto de treinamento tenha o único exemplo positivo
    condition = (df['is_buggy_commit'] == 1) | (df['is_buggy_commit'] == 1.0)
    indices = df[condition].index.tolist()

    if indices[0] >= train_size:
      train_size = indices[0] + 1

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_bug_ratio = train_df["is_buggy_commit"].sum()/train_df.shape[0]
    print("Adjusted training bug ratio = ", train_bug_ratio)

    val_bug_ratio = val_df["is_buggy_commit"].sum()/val_df.shape[0]
    print("Adjusted validation bug ratio = ", val_bug_ratio)

    val_clean_ratio = val_df[val_df["is_buggy_commit"] == 0]["is_buggy_commit"].count()/val_df.shape[0]
    print("Adjusted validation clean ratio = ", val_clean_ratio)

    train_commits = commits[:train_size]
    train_labels = labels[:train_size]
    train_commit_messages = commit_messages[:train_size]
    train_codes = codes[:train_size]
    val_commits = commits[train_size:]
    val_labels = labels[train_size:]
    val_commit_messages = commit_messages[train_size:]
    val_codes = codes[train_size:]
    assert train_df.shape[0] == len(train_commits) == len(train_labels) == len(train_commit_messages) == len(train_codes)
    assert val_df.shape[0] == len(val_commits) == len(val_labels) == len(val_commit_messages) == len(val_codes)
    changes_train = (train_commits, train_labels, train_commit_messages, train_codes)
    changes_valid = (val_commits, val_labels, val_commit_messages, val_codes)

  elif df['is_buggy_commit'].sum() == 1 and val_bug_ratio == 0 and not do_eval_with_all_negative:
      #Mudança 06/07 -> flag do_eval_with_all_negative, que indica se deve fazer validação mesmo sem nenhum exemplo de validação positivo.
      #Caso seja igual a false, não faz validação, ou seja, seta o split de validação para ser igual ao split de treino.
      print("Not enough positive samples for validation -> using same split for training...")
      train_df = val_df = df
      train_commits = val_commits = commits
      train_labels = val_labels = labels
      train_commit_messages = val_commit_messages = commit_messages
      train_codes = val_codes = codes
      assert train_df.shape[0] == len(train_commits) == len(train_labels) == len(train_commit_messages) == len(train_codes)
      assert val_df.shape[0] == len(val_commits) == len(val_labels) == len(val_commit_messages) == len(val_codes)
      changes_train = (train_commits, train_labels, train_commit_messages, train_codes)
      changes_valid = (val_commits, val_labels, val_commit_messages, val_codes)

  else:
    train_commits = commits[:train_size]
    train_labels = labels[:train_size]
    train_commit_messages = commit_messages[:train_size]
    train_codes = codes[:train_size]
    val_commits = commits[train_size:]
    val_labels = labels[train_size:]
    val_commit_messages = commit_messages[train_size:]
    val_codes = codes[train_size:]
    assert train_df.shape[0] == len(train_commits) == len(train_labels) == len(train_commit_messages) == len(train_codes)
    assert val_df.shape[0] == len(val_commits) == len(val_labels) == len(val_commit_messages) == len(val_codes)
    changes_train = (train_commits, train_labels, train_commit_messages, train_codes)
    changes_valid = (val_commits, val_labels, val_commit_messages, val_codes)

  with open(f"{path}/changes_train_online_{project}.pkl", "wb") as f:
    pickle.dump(changes_train, f)

  with open(f"{path}/changes_valid_online_{project}.pkl", "wb") as f:
    pickle.dump(changes_valid, f)

  with open(f"{path}/features_train_online_{project}.pkl", "wb") as f:
    pickle.dump(train_df, f)

  with open(f"{path}/features_valid_online_{project}.pkl", "wb") as f:
    pickle.dump(val_df, f)

  return f"{path}/changes_train_online_{project}.pkl", f"{path}/features_train_online_{project}.pkl", f"{path}/changes_valid_online_{project}.pkl", f"{path}/features_valid_online_{project}.pkl"

def add_to_cumulative_training_pool(row, global_training_pool):
  for training_example in global_training_pool:
    if training_example['commit_hash'] == row['commit_hash']:
      training_example['is_buggy_commit'] = row['is_buggy_commit']
      return

  global_training_pool.append(row)

def train(batch_classifier_dir, path, full_changes_train_file, full_changed_valid_file, full_changes_test_file, project, model_path, 
          training_pool, use_only_new_data=True, th=0.5, eval_metric="f1", do_oversample=False, do_undersample=False, 
          pretrained_model='codet5p-770m', trained=0, skewed_oversample=False, peft_alg="lora", seed=33, window_size=100, 
          target_th=0.5, l0=10, l1=12, m=1.5, batch_size=16, cross_project=False, do_eval_with_all_negative=False):

  if os.path.exists(os.path.join(model_path, "training_status.txt")):
    os.remove(os.path.join(model_path, "training_status.txt"))

  df = pd.DataFrame(training_pool)

  print("Training pool size = ", df.shape[0])

  changes_train_file, features_train_file, changes_valid_file, features_valid_file = prepare_training_data(path, 
                                                                                                           full_changes_train_file, 
                                                                                                           full_changed_valid_file, 
                                                                                                           full_changes_test_file, 
                                                                                                           project, df, 
                                                                                                           do_eval_with_all_negative=do_eval_with_all_negative)

  batches.append(df)

  #Training command:
  model_name = pretrained_model

  action = "do_train"

  #If there's already a trained model, continue training with the model referenced by output_dir
  if os.path.exists(f"{model_path}/checkpoint-best-{eval_metric}/model.bin"):
    action = "do_resume_training"

  command = f"""
  python {batch_classifier_dir}PEFT4CC/just-in-time/run_{peft_alg}.py \
   --train_data_file {changes_train_file} {features_train_file} \
   --eval_data_file {changes_valid_file} {features_valid_file} \
   --output_dir {model_path} \
   --pretrained_model {model_name} \
   --learning_rate 1e-4 \
   --epochs 10 \
   --batch_size {batch_size} \
   --{action} \
   --threshold {th} \
   --seed {seed} \
   --window_size {window_size} \
   --target_th {target_th} \
   --l0 {l0} \
   --l1 {l1} \
   --m {m} \
   --activation relu \
   """

  if eval_metric == "gmean":
    command += """--eval_metric gmean """

  if do_oversample:
    command += """--oversample """
  elif skewed_oversample:
    command += """--skewed_oversample """

  if do_undersample:
    command += """--undersample """

  if pretrained_model == 'codet5p-770m':
    command += """--hidden_size 1024 """

  if peft_alg == "lora":
    command += """--use_lora """
    
  if cross_project:
    command += """--cross_project """

  """
  try:
    file_to_monitor = f'{model_path}/model.bin'
  except FileNotFoundError:
    print(f"File '{file_to_monitor}' not found.")
    """

  print(f"Training with th={th}...")
  execute_command(command)

  if use_only_new_data:
    while not os.path.exists(os.path.join(model_path, "training_status.txt")):
      print("Waiting for traning_status.txt")
      time.sleep(5)

    with open(os.path.join(model_path, "training_status.txt"), "r") as file:
      file_contents = file.read()

      if file_contents == 'changed':
        print(f"Model file has changed!")
        #Clear training pool
        training_pool.clear()
        trained += len(set([sample['commit_hash'] for sample in training_pool]))
      else:
        print(f"Model file has not changed.  Keeping training data.")

  return th, trained

def check_df_project_sorted(df_project):
    if 'author_date_unix_timestamp' not in df_project.columns:
      raise ValueError("Column 'author_date_unix_timestamp' not found.")
    timestamps = pd.to_numeric(df_project['author_date_unix_timestamp'], errors='coerce')
    if not timestamps.is_monotonic_increasing:
      raise ValueError("df_project is NOT increasingly sorted by 'author_date_unix_timestamp'.")
    print("df_project IS increasingly sorted by 'author_date_unix_timestamp'")

def merge_cross_project_data(df_features_full, df_train, project, initial_cp_timestamp=0, current_timestamp=0):
  # For cross-project JIT-SDP, adds other projects data.  
  max_timestamp = 0
  if len(df_train) == 0:
    max_timestamp = current_timestamp
  else:
    max_timestamp = pd.to_numeric(df_train['author_date_unix_timestamp'], errors='coerce').max()
  
  print(f"Cross-project training enabled. Merging data from other projects with timestamps between {datetime.fromtimestamp(initial_cp_timestamp)} and {datetime.fromtimestamp(max_timestamp)}")
  if df_features_full is not None:
    df_others = df_features_full[df_features_full['project'] != project]
    df_others['author_date_unix_timestamp'] = pd.to_numeric(df_others['author_date_unix_timestamp'], errors='coerce')
    max_timestamp = float(max_timestamp)
    initial_cp_timestamp = float(initial_cp_timestamp)
    df_others = df_others[df_others['author_date_unix_timestamp'] < max_timestamp]
    df_others = df_others[df_others['author_date_unix_timestamp'] >= initial_cp_timestamp]
  else:
    df_others = pd.DataFrame()
  
  # Concatenate df_train and df_others, maintaining ascending order by author_date_unix_timestamp
  if not df_others.empty:
    df_train = pd.concat([df_train, df_others], ignore_index=True)
    df_train['author_date_unix_timestamp'] = pd.to_numeric(df_train['author_date_unix_timestamp'], errors='coerce')
    df_train = df_train.sort_values(by='author_date_unix_timestamp', ascending=True).reset_index(drop=True)
  
  return df_train, max_timestamp

def save_execution_status(model_path, current, step, list_of_predictions, list_of_results, max_timestamp_for_cp):
  with open(os.path.join(model_path, "training_status.pickle"), "wb") as f:
        pickle.dump({"current":current + step, "list_of_predictions":list_of_predictions, "max_timestamp_for_cp": max_timestamp_for_cp}, f)
  return list_of_results, list_of_predictions, False


def train_on_line_with_new_data(batch_classifier_dir, path, full_changes_train_file, full_changes_valid_file, 
                                full_changes_test_file, project, df_project, model_path, training_pool, training_queue, 
                                map_commit_to_row, buggy_pool=[], training_examples=50, th=0.5, adjust_th=False,
                                eval_metric="f1", do_oversample=False, do_undersample=False, 
                                pretrained_model='codet5p-770m', do_real_lat_ver=False, skewed_oversample=False,
                                adjust_th_on_test=False, seed=33, window_size=100, target_th=0.5, l0=10, l1=12,
                                m=1.5, train_from_scratch=True, batch_size=16, df_features_full=None, cross_project=False, 
                                do_eval_with_all_negative=False):
  list_of_results = []
  list_of_predictions = []

  print('len(training_pool) = ', len(training_pool))
  print('len(training_queue) = ', len(training_queue))

  #Deletes previous project global data
  if os.path.exists('global_training_pool.pkl'):
    os.remove('global_training_pool.pkl')

  df_project = df_project.reset_index(drop=True)
  trained = 0

  start = 0
  max_timestamp_for_cp = 0

  if os.path.exists(os.path.join(model_path, "training_status.pickle")):
    with open(os.path.join(model_path, "training_status.pickle"), "rb") as f:
      training_status = pickle.load(f)
      if 'current' in training_status:
        start = training_status['current']
        print("Resuming from current = ", start)
        list_of_predictions = training_status.get('list_of_predictions', [])
        print("Resuming with list_of_predictions of size = ", len(list_of_predictions))
        max_timestamp_for_cp = training_status.get('max_timestamp_for_cp', 0)
      else:
        print("Starting from scratch.")

  step = training_examples
  end = df_project.shape[0]
  
  check_df_project_sorted(df_project)
  execution_start = time()
  #max_exec_time = 20 * 60 * 60 #20 hours
  max_exec_time = 40 * 60 * 60 #40 hours

  print(df_project.head())

  for current in range(start, end, step):
    try:
      print('current = ', current)
      current_timestamp = df_project['author_date_unix_timestamp'].iloc[current]
      print(f"current_timestamp = {datetime.fromtimestamp(int(float(current_timestamp)))}")

      if train_from_scratch:
        df_train = df_project[:current].copy() #all data
        training_queue.clear()
        buggy_pool.clear()
      else:
        df_train = df_project[max(current - step, 0):current].copy() #only recent data

      project = df_project['project'].iloc[current]
      df_train.to_csv(f'df_train_{project}_{current}.csv')
      df_test = df_project[current:min(current + step, end)].copy()
      df_test.to_csv(f'df_test_{project}_{current}.csv')

      df_train, max_timestamp_for_cp = adjust_train_data(project, df_features_full, cross_project, df_train, 
                                                        initial_cp_timestamp=max_timestamp_for_cp, current_timestamp=current_timestamp)

      # Builds training queue and training pool based on latency verification and buggy commit detection.  
      _ = prepare_train_data(df_train, training_pool, training_queue, map_commit_to_row, buggy_pool, do_real_lat_ver=do_real_lat_ver)
      
      print("Training pool size = ", len(training_pool))

      if is_valid_training_data(training_pool):
        #Train
        try:
          th, trained = train(batch_classifier_dir, path, full_changes_train_file, full_changes_valid_file, 
                              full_changes_test_file, project, model_path, training_pool, th=th, eval_metric=eval_metric, 
                              do_oversample=do_oversample and (not skewed_oversample), do_undersample=do_undersample,
                              pretrained_model=pretrained_model, trained=trained, skewed_oversample=skewed_oversample, seed=seed, 
                              window_size=window_size, target_th=target_th, l0=l0, l1=l1, m=m, batch_size=batch_size, 
                              cross_project=cross_project, do_eval_with_all_negative=do_eval_with_all_negative)
        except Exception as e:
          print("Not enough labeled training/validation data.  Delaying training...")
          print(f"Ocorreu um erro: {str(e)}")
          traceback.print_exc()

      #Test with current model (or no model)
      calculate_metrics = df_test['is_buggy_commit'].sum() > 0

      if os.path.exists(f"{model_path}/checkpoint-best-{eval_metric}/model.bin"):
        if adjust_th_on_test:
          print("Calculating new th...")
          _, predictions = test(batch_classifier_dir, path, full_changes_train_file, full_changes_valid_file, 
                                full_changes_test_file, project, df_test[-window_size:], model_path, th=th, 
                                pretrained_model=pretrained_model, calculate_metrics=calculate_metrics, 
                                eval_metric=eval_metric, batch_size=batch_size)
          th = calculate_th_from_test(predictions, target_th=target_th)

        results, predictions = test(batch_classifier_dir, path, full_changes_train_file, full_changes_valid_file, 
                                    full_changes_test_file, project, df_test, model_path, th=th, 
                                    pretrained_model=pretrained_model, calculate_metrics=calculate_metrics,
                                    eval_metric=eval_metric, batch_size=batch_size)
        list_of_results.append(results)
      else:
        #file_to_monitor = f'{model_path}/model.bin'
        print(f"No trained model found on {model_path}/checkpoint-best-{eval_metric}/model.bin")
        print(f"{model_path}/checkpoint-best-{eval_metric}/model.bin")
        print("df_test.shape[0] = ", df_test.shape[0])
        pred_label = [0] * df_test.shape[0]
        pred_prob = [0] * df_test.shape[0]
        true_label = df_test['is_buggy_commit'].to_list()
        predictions = {'pred_label': pred_label, 'true_label': true_label,
                      'pred_prob':pred_prob}

      list_of_predictions.append(predictions)
    except:
      print("Error during training/testing.  Saving intermediate results and aborting processing...")
      return save_execution_status(model_path, current, step, list_of_predictions, list_of_results, max_timestamp_for_cp)

    if time() > execution_start + max_exec_time:
      #pauses execution
      print("Maximum execution time reached.  Saving current state...") 
      return save_execution_status(model_path, current, step, list_of_predictions, list_of_results, max_timestamp_for_cp)
    """
      with open(os.path.join(model_path, "training_status.pickle"), "wb") as f:
        pickle.dump({"current":current + step, "list_of_predictions":list_of_predictions}, f)
      return list_of_results, list_of_predictions, False
      """

  print("Final training pool size = ", len(training_pool))

  return list_of_results, list_of_predictions, True

def adjust_train_data(project, df_features_full, cross_project, df_train, initial_cp_timestamp=0, current_timestamp=0):
    if cross_project:
      df_train, max_timestamp = merge_cross_project_data(df_features_full, df_train, project, initial_cp_timestamp, current_timestamp)
      return df_train, max_timestamp
    else:
      return df_train, 0

def train_on_line_with_new_data_with_early_stop(batch_classifier_dir, path, full_changes_train_file, 
                                                full_changes_valid_file, full_changes_test_file, project, df_project, 
                                                model_path, early_stop_metric='f1', do_real_lat_ver=False, 
                                                adjust_th=False, do_oversample=True, skewed_oversample=False,
                                                adjust_th_on_test=False, seed=33, window_size=100, target_th=0.5, l0=10, l1=12,
                                                m=1.5, pretrained_model="codet5p-770m", train_from_scratch=True, batch_size=16, 
                                                df_features_full=None, cross_project=False, do_eval_with_all_negative=False):
  batches = []
  training_pool = []
  training_queue = []
  buggy_pool = []
  map_commit_to_row = dict()
  print('len(batches) in train_on_line_with_new_data_with_early_stop(): ',
        len(batches))
  return train_on_line_with_new_data(batch_classifier_dir, path, full_changes_train_file, full_changes_valid_file, 
                                     full_changes_test_file, project, df_project, model_path, training_pool, 
                                     training_queue, map_commit_to_row, buggy_pool, eval_metric=early_stop_metric,
                                     do_oversample=do_oversample, do_real_lat_ver=do_real_lat_ver, adjust_th=adjust_th, 
                                     skewed_oversample=skewed_oversample, adjust_th_on_test=adjust_th_on_test, seed=seed, window_size=window_size,
                                     target_th=target_th, l0=l0, l1=l1, m=m, pretrained_model=pretrained_model,
                                     train_from_scratch=train_from_scratch, batch_size=batch_size, df_features_full=df_features_full, 
                                     cross_project=cross_project, do_eval_with_all_negative=do_eval_with_all_negative)

def train_project(batch_classifier_dir, path, model_root, commit_guru_path, full_features_train_file, 
                  full_features_valid_file, full_features_test_file, full_changes_train_file, full_changed_valid_file, 
                  full_changes_test_file, project, early_stop_metric="gmean", do_real_lat_ver=False, adjust_th=False, 
                  do_oversample=True, model_path=None, skewed_oversample=False, adjust_th_on_test=False, seed=33, 
                  window_size=100, target_th=0.5, l0=10, l1=12 , m=1.5, start=0, end=None, 
                  pretrained_model="codet5p-770m", train_from_scratch=True, batch_size=16, cross_project=False, 
                  do_eval_with_all_negative=False):
  
  df_features_full = get_df_features_full(full_features_train_file, full_features_valid_file, full_features_test_file)
  df_project = df_features_full[df_features_full['project'] == project]
  rows1 = df_project.shape[0]
  print('df_project.shape before: ', df_project.shape)
  df_project = add_first_fix_date(commit_guru_path, df_project, project)
  rows2 = df_project.shape[0]
  print('df_project.shape after: ', df_project.shape)
  assert rows1 == rows2

  df_features_full = adjust_df_features_full(commit_guru_path, cross_project, df_features_full)

  if not end:
    end = df_project.shape[0]

  df_project = df_project[start:end]
  print('Final df_project shape = ', df_project.shape)

  if not model_path:
    model_path = model_root + pretrained_model + f"/concat/online/baseline/{project}_best_{early_stop_metric}/checkpoints"

  results, list_of_predictions, finished = train_on_line_with_new_data_with_early_stop(batch_classifier_dir, path, 
                                                                             full_changes_train_file, 
                                                                             full_changed_valid_file, 
                                                                             full_changes_test_file, project, 
                                                                             df_project, model_path, 
                                                                             early_stop_metric=early_stop_metric, 
                                                                             do_real_lat_ver=do_real_lat_ver, 
                                                                             adjust_th=adjust_th, 
                                                                             do_oversample=do_oversample, 
                                                                             skewed_oversample=skewed_oversample, 
                                                                             adjust_th_on_test=adjust_th_on_test, 
                                                                             seed=seed, window_size=window_size, 
                                                                             target_th=target_th, l0=l0, l1=l1, m=m, 
                                                                             pretrained_model=pretrained_model, 
                                                                             train_from_scratch=train_from_scratch,
                                                                             batch_size=batch_size, df_features_full=df_features_full, 
                                                                             cross_project=cross_project, 
                                                                             do_eval_with_all_negative=do_eval_with_all_negative)

  true_labels = []
  pred_labels = []
  pred_probs = []

  for prediction in list_of_predictions:
    tl = prediction['true_label']
    pl = prediction['pred_label']
    pp = prediction['pred_prob']
    true_labels.extend(tl)
    pred_labels.extend(pl)
    pred_probs.extend(pp)

  predictions = {'true_labels': df_project['is_buggy_commit'].tolist(),
                 'pred_labels': pred_labels, 'pred_probs': pred_probs}

  return results, predictions, model_path, finished

def adjust_df_features_full(commit_guru_path, cross_project, df_features_full):
    if cross_project:
      projects = df_features_full['project'].unique()
      df_features_full_with_first_fix_date = pd.DataFrame()
    
      for project in projects:
        df_cross = df_features_full[df_features_full['project'] == project]
        df_cross = add_first_fix_date(commit_guru_path, df_cross, project)
        df_features_full_with_first_fix_date = pd.concat([df_features_full_with_first_fix_date, df_cross], ignore_index=True)
    
      df_features_full_with_first_fix_date = df_features_full_with_first_fix_date.sort_values(by='author_date_unix_timestamp', 
                                                                                              ascending=True).reset_index(drop=True)
      return df_features_full_with_first_fix_date
    
    return df_features_full

def train_project_with_lat_ver(batch_classifier_dir, path, model_root, commit_guru_path, full_features_train_file, full_features_valid_file, full_features_test_file, 
                               full_changes_train_file, full_changed_valid_file, full_changes_test_file, project, 
                               early_stop_metric="gmean", adjust_th=False, do_oversample=True, skewed_oversample=False, 
                               adjust_th_on_test=False, seed=33, decay_factor=0.99, window_size=100, target_th=0.5, l0=10, l1=12 , 
                               m=1.5, results_folder='', start=0, end=None, pretrained_model="codet5p-770m", train_from_scratch=True, 
                               batch_size=16, cross_project=False, do_eval_with_all_negative=False):

    columns = ['project', 'g_mean', 'f1', 'precision', 'recall', 'R0', 'R1',
             '|R0-R1|', 'std_g_mean', 'std_f1', 'std_precision', 'std_recall',
             'std_R0', 'std_R1', 'std_|R0-R1|']

    print(f"Project: {project}")
    df = pd.DataFrame(columns=columns)

    _, predictions, model_path, finished = train_project(batch_classifier_dir, path, model_root, commit_guru_path, 
                                                         full_features_train_file, full_features_valid_file, full_features_test_file, 
                                                         full_changes_train_file, full_changed_valid_file, full_changes_test_file, 
                                                         project, early_stop_metric=early_stop_metric, do_real_lat_ver=True, 
                                                         adjust_th=adjust_th, do_oversample=do_oversample, 
                                                         skewed_oversample=skewed_oversample, adjust_th_on_test=adjust_th_on_test, 
                                                         seed=seed, window_size=window_size, target_th=target_th, l0=l0, l1=l1, m=m, 
                                                         start=start, end=end, pretrained_model=pretrained_model, 
                                                         train_from_scratch=train_from_scratch, batch_size=batch_size, 
                                                         cross_project=cross_project, 
                                                         do_eval_with_all_negative=do_eval_with_all_negative)

    if finished:
      print("Training finished successfully.")
      
      with open(f'{model_path}/{project}_predictions_wp.pkl', 'wb') as f:
        pickle.dump(predictions, f)
      
      calculate_metrics_and_plot(model_root, project, early_stop_metric, decay_factor, results_folder, pretrained_model, df, predictions, model_path)

def calculate_metrics_and_plot(model_root, project, early_stop_metric, decay_factor, results_folder, pretrained_model, 
                               df, predictions, model_path):
    mean_g_mean, std_g_mean, mean_r_diff, std_r_diff, mean_f1, std_f1, mean_precision, std_precision, mean_recall, std_recall, mean_r0, std_r0, mean_r1, std_r1, roc_auc, metrics = calculate_prequential_mean_and_std(predictions, decay_factor=decay_factor)

    plot(metrics, model_path)

    df.loc[len(df)] = [project, mean_g_mean, mean_f1, mean_precision,
                        mean_recall, mean_r0, mean_r1, mean_r_diff, std_g_mean,
                        std_f1, std_precision, std_recall, std_r0, std_r1,
                        std_r_diff]
    base_path = model_root + pretrained_model + '/concat/online/baseline'
    path = base_path + f'/{results_folder}'
    df.to_csv(f'{path}/{project}_results_wp.csv', index=False)

    with open(f'{path}/{project}_predictions_wp.pkl', 'wb') as f:
      pickle.dump(predictions, f)

    metrics.to_csv(f'{path}/{project}_metrics_wp.csv', index=False)

    if len(results_folder) > 0:
      shutil.move(f'{base_path}/{project}_best_{early_stop_metric}', path)