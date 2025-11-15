from datetime import datetime


def get_difference(timestamp1, timestamp2):
  dt1 = datetime.fromtimestamp(timestamp1)
  dt2 = datetime.fromtimestamp(timestamp2)
  difference = dt2 - dt1
  days_difference = difference.days
  return days_difference