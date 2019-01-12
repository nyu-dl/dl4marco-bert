import numpy as np


def average_precision(gt, pred):
  """
  Computes the average precision.

  This function computes the average prescision at k between two lists of
  items.

  Parameters
  ----------
  gt: set
       A set of ground-truth elements (order doesn't matter)
  pred: list
        A list of predicted elements (order does matter)

  Returns
  -------
  score: double
      The average precision over the input lists
  """

  if not gt:
    return 0.0

  score = 0.0
  num_hits = 0.0
  for i,p in enumerate(pred):
    if p in gt and p not in pred[:i]:
      num_hits += 1.0
      score += num_hits / (i + 1.0)

  return score / max(1.0, len(gt))


def NDCG(gt, pred, use_graded_scores=False):
  score = 0.0
  for rank, item in enumerate(pred):
    if item in gt:
      if use_graded_scores:
        grade = 1.0 / (gt.index(item) + 1)
      else:
        grade = 1.0
      score += grade / np.log2(rank + 2)

  norm = 0.0
  for rank in range(len(gt)):
    if use_graded_scores:
      grade = 1.0 / (rank + 1)
    else:
      grade = 1.0
    norm += grade / np.log2(rank + 2)
  return score / max(0.3, norm)


def metrics(gt, pred, metrics_map):
  '''
  Returns a numpy array containing metrics specified by metrics_map.
  gt: ground-truth items
  pred: predicted items
  '''
  out = np.zeros((len(metrics_map),), np.float32)

  if ('MAP' in metrics_map):
    avg_precision = average_precision(gt=gt, pred=pred)
    out[metrics_map.index('MAP')] = avg_precision

  if ('RPrec' in metrics_map):
    intersec = len(gt & set(pred[:len(gt)]))
    out[metrics_map.index('RPrec')] = intersec / max(1., float(len(gt)))

  if 'MRR' in metrics_map:
    score = 0.0
    for rank, item in enumerate(pred):
      if item in gt:
        score = 1.0 / (rank + 1.0)
        break
    out[metrics_map.index('MRR')] = score

  if 'MRR@10' in metrics_map:
    score = 0.0
    for rank, item in enumerate(pred[:10]):
      if item in gt:
        score = 1.0 / (rank + 1.0)
        break
    out[metrics_map.index('MRR@10')] = score

  if ('NDCG' in metrics_map):
    out[metrics_map.index('NDCG')] = NDCG(gt, pred)

  return out

