# random cities based on number of flights
import random
from collections import defaultdict
## given a dictionary:
input = {'NY': 100, 'LA': 80, "SEA": 75}
## you cannot use random.choices. Instead, use random. how to do it.
def random_cities(flights):
  flights_idxes = range(len(flights))
  return random.choices(flights_idxes, flights, k=1)

flights = [80, 100, 75]
output = defaultdict(int)
for i in range(100):
  city = random_cities(flights)[0]
  output[city] += 1
print(output)