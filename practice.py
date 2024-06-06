from collections import Counter
def schedule_tasks(tasks, n):
  leng = len(tasks)
  task_counters = Counter(tasks)
  seq_len = 0
  most_frequent = task_counters.most_common(1)
  (task, count) = most_frequent[0]
  while True:
    tc = 0
    for task, count in task_counters.items():
      if count > 0:
        seq_len += 1
        task_counters[task] = count-1
        tc += 1
      if count == 0 and tc == len(task_counters):
        break
    while seq_len % n != 0:
      seq_len += 1
  return seq_len


tasks = ["A", "A", "A", "B", "B", "B"]
n = 3
schedule_tasks(tasks, n)

tasks = ["A", "C", "A", "B", "D", "B"]
n = 1
