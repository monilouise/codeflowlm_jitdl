from datetime import datetime
import pandas as pd


import math
import os

from codeflowlm.date_util import get_difference


def add_first_fix_date(commit_guru_path, df, project):
    csv = commit_guru_path + project + '.csv'

    if os.path.exists(csv) == False:
      return df

    df_csv = pd.read_csv(csv)
    result = pd.merge(df, df_csv[['commit_hash', 'fixes']], on=['commit_hash'], how='left')

    result = result.fillna('')
    print("result.columns = ", result.columns)

    for index, row in result.iterrows():
        if row['is_buggy_commit'] == 1:
          fixes = row['fixes']
          fixes = fixes[1:-1]
          fixes = fixes.split()
          timestamp = math.inf

          for fix in fixes:
              last_pos = fix.rfind('"')
              fix = fix[1:last_pos]
              fix_commit = result[result['commit_hash'] == fix]
              if fix_commit.shape[0] > 0:
                  author_date_unix_timestamp = fix_commit['author_date_unix_timestamp']
                  author_date_unix_timestamp = int(author_date_unix_timestamp.iloc[0])

                  if author_date_unix_timestamp < timestamp:
                      timestamp = author_date_unix_timestamp

          if timestamp < math.inf:
              result.loc[index, 'first_fix_date'] = int(timestamp)

    result = result.fillna(0)
    result['first_fix_date'] = result['first_fix_date'].astype(int)
    return result


def add_to_training_pool(row, training_pool):
  #Verifica se o id já existe no training pool.  Se já existir, checar o label.
  #Se for 0 e o novo for 1, setar o label no registro atual e ignorar o novo
  #registro.  Nas demais combinações de labels atual e novo, ignorar o novo
  #registro e deixar o atual. Se não existir, adicionar o novo.
  for training_example in training_pool:
    if training_example['commit_hash'] == row['commit_hash']:
      if training_example['is_buggy_commit'] == 0 and row['is_buggy_commit'] == 1:
        training_example['is_buggy_commit'] = 1.0
        assert training_example['is_buggy_commit'] == row['is_buggy_commit']
      return

  training_pool.append(row)


waiting_time = 90


def do_real_latency_verification(row, training_pool, training_queue,
                            map_commit_to_row, buggy_pool):

  #olhar os commits que têm o atributo first_fix_date != 0 e checar se essa data
  #é anterior à data do commit atual
  for example in training_queue:
    example_row = map_commit_to_row[example[0]]
    
    if 'first_fix_date' not in example_row:
      print(f"Example {example[0]} does not have 'first_fix_date' column!!!!!!!!!")
      print(example_row.shape)
      continue
    
    if example_row['first_fix_date'] != 0 and example_row['first_fix_date'] < row['author_date_unix_timestamp']:
      print(f"Current date: {row['author_date']}.  Promoting example from {example_row['project']} fixed on {datetime.fromtimestamp(example_row['first_fix_date'])} to training pool.")
      #volta o label para 1
      example_row['is_buggy_commit'] = 1.0
      #training_pool.append(map_commit_to_row[example[0]])
      add_to_training_pool(example_row, training_pool)
      assert example_row['is_buggy_commit'] == map_commit_to_row[example[0]]['is_buggy_commit']
      training_queue.remove(example)
      buggy_pool.remove(example)

  #Checks for examples older than waiting time to promote them to training pool
  for example in training_queue:
    #timestamp1 = example['author_date_unix_timestamp']
    example_row = map_commit_to_row[example[0]]
    timestamp1 = example[1]
    timestamp2 = row['author_date_unix_timestamp']

    if get_difference(timestamp1, timestamp2) >= waiting_time:
      print(f"Current date: {row['author_date']}.  Promoting example from {example_row['project']} commited on {datetime.fromtimestamp(timestamp1)} to training pool.")
      #training_pool.append(map_commit_to_row[example[0]])
      add_to_training_pool(map_commit_to_row[example[0]], training_pool)
      training_queue.remove(example)
    #else: #COMENTADO EM 22/06/25!!!
    #  break
    else: #DESCOMENTADO EM 24/06/25!!!
      break

  #Olhar também para o pool de commits defeituosos para checar se tem algum cujo
  #atributo first_fix_date seja menor que a data do commit atual: esses terão
  #que ser promovidos para dados  de treinamento, sendo reapresentados como
  #dados positivos.
  for example in buggy_pool:
    example_row = map_commit_to_row[example[0]]
    if example_row['first_fix_date'] != 0 and example_row['first_fix_date'] < row['author_date_unix_timestamp']:
      print(f"Current date: {row['author_date']}.  Promoting example from {example_row['project']} fixed on {datetime.fromtimestamp(example_row['first_fix_date'])} to training pool.")
      example_row['is_buggy_commit'] = 1.0
      #training_pool.append(example_row)
      add_to_training_pool(example_row, training_pool)
      assert example_row['is_buggy_commit'] == map_commit_to_row[example[0]]['is_buggy_commit']

      if example in training_queue:
        training_queue.remove(example)

      buggy_pool.remove(example)


def do_latency_verification(row, training_pool, training_queue,
                            map_commit_to_row):
  #Checks for examples older than waiting time to promote them to training pool
  for example in training_queue:
    #timestamp1 = example['author_date_unix_timestamp']
    timestamp1 = example[1]
    timestamp2 = row['author_date_unix_timestamp']

    if get_difference(timestamp1, timestamp2) >= waiting_time:
      training_pool.append(map_commit_to_row[example[0]])
      training_queue.remove(example)
    else:
      break


def process_buggy_commit(row, training_queue, map_commit_to_row, buggy_pool):
  #Ao encontrar no dataset ordenado um commit rotulado como buggy, setar o label
  #como 0, colocar o commit no pool do waiting time e guardar o commit também em
  #um pool de commits defeituosos até que chegue um outro commit qualquer com
  #data posterior à data de detecção dele.
  row['is_buggy_commit'] = 0.0
  training_queue.append((row['commit_hash'], row['author_date_unix_timestamp']))
  map_commit_to_row[row['commit_hash']] = row
  buggy_pool.append((row['commit_hash'], row['author_date_unix_timestamp']))