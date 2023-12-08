#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from typing import List

# A State object for use in branch and bound
# contains a bound property and a matrix property
class State:
	def __init__(self, bound, matrix, path):
		self.bound = bound
		self.matrix = matrix
		self.path = path

	# when comparing states, the longer the path length, the more likely that that state should be prioritized. However, still consider the bound of each state
	def __lt__(self, other):
		return (self.bound / len(self.path)) < (other.bound / len(other.path))



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''


	def greedy( self,time_allowance=60.0 ):
		# start at an arbitrary city
		# take the shortest path to an unvisited city until a complete tour is formed
		# If it gets stuck (no untraveled, non-infinite paths left), restart at a different city

		cities: List[City] = self._scenario.getCities()
		
		distances = np.zeros((len(cities), len(cities)))
		for i in range(len(cities)):
			for j in range(len(cities)):
				distances[i][j] = cities[i].costTo(cities[j])

		soln = None

		while soln is None:
			# start at random city
			curr_city = random.randint(0, len(cities) - 1)
			visited_cities = {curr_city}
			found_new_city = True

			while found_new_city:
				found_new_city = False
				min_dist = np.inf
				min_dist_city = None
				possible_destinations = distances[curr_city, :]
				for i in range(len(possible_destinations)):
					if possible_destinations[i] == np.inf or i in visited_cities:
						continue
					if distances[curr_city][i] < min_dist:
						min_dist = distances[curr_city][i]
						min_dist_city = i
						found_new_city = True
				
				if found_new_city:
					visited_cities.add(min_dist_city)
					curr_city = min_dist_city
			
			if len(visited_cities) == len(cities):
				soln = TSPSolution([cities[i] for i in visited_cities])
		
		res = {}
		res['soln'] = soln
		res['time'] = 0
		res['count'] = 0
		res['cost'] = soln.cost
		res['max'] = None
		res['total'] = None
		res['pruned'] = None

		return res
		


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''


	def branchAndBound( self, time_allowance=60.0 ):
		numStates = 0
		def reduce_matrix(matrix, prior_bound, cost=0):
			new_matrix = matrix.copy()
			bound = prior_bound
			# reduce rows
			for i in range(len(new_matrix)):
				min = np.min(new_matrix[i])
				if min == np.inf:
					continue
				bound += min
				for j in range(len(new_matrix[i])):
					if new_matrix[i][j] != np.inf:
						new_matrix[i][j] -= min

			# reduce columns
			for i in range(len(new_matrix)):
				min = np.min(new_matrix[:, i])
				if min == np.inf:
					continue
				bound += min
				for j in range(len(new_matrix[i])):
					if new_matrix[j][i] != np.inf:
						new_matrix[j][i] -= min
				
			bound += cost
			return (bound, new_matrix)
		
		def is_solution(state: State):
			if len(state.path) == len(self._scenario.getCities()):
				return True
			return False
		
		def expand(state: State) -> List[State]:
			expanded_states = []
			curr_city = state.path[-1]
			
			possible_destinations = state.matrix[curr_city, :]
			for i in range(len(possible_destinations)):
				# don't visit cities that cant be reached or the initial city
				if possible_destinations[i] == np.inf or i == 0:
					continue
				new_matrix = state.matrix.copy()
				new_matrix[curr_city, :] = np.inf
				new_matrix[:, i] = np.inf
				new_matrix[i, curr_city] = np.inf

				cost = possible_destinations[i]
				new_bound, new_reduced_matrix = reduce_matrix(new_matrix, state.bound, cost)
				new_path = state.path.copy()
				new_path.append(i)
				new_state = State(new_bound, new_reduced_matrix, new_path)
				expanded_states.append(new_state)
			return expanded_states


		start_time = time.time()
		pq = []
		bssf: TSPSolution = self.greedy()['soln']

		scenario: Scenario = self._scenario
		cities: List[City] = scenario.getCities()

		# convert cities to a matrix where the value at [i][j] is the distance from city i to city j
		distances = np.zeros((len(cities), len(cities)))
		for i in range(len(cities)):
			for j in range(len(cities)):
				distances[i][j] = cities[i].costTo(cities[j])

		bound, initial_reduced_matrix = reduce_matrix(distances, prior_bound=0, cost=0)
		initial_state = State(bound, initial_reduced_matrix, [0])

		heapq.heappush(pq, initial_state)

		num_solns = 0
		num_states = 1
		num_pruned = 0
		max_queue_size = 0

		while len(pq) > 0 and time.time() - start_time < time_allowance:
			max_queue_size = max(max_queue_size, len(pq))
			state: State = heapq.heappop(pq)

			if state.bound >= bssf.cost:
				num_pruned += 1
				continue
			
			if is_solution(state):
				new_bssf = TSPSolution([cities[i] for i in state.path])
				if new_bssf.cost < bssf.cost:
					num_solns += 1
					bssf = new_bssf
			
			new_states = expand(state)
			num_states += len(new_states)
			num_pushed = 0
			for new_state in new_states:
				if new_state.bound < bssf.cost:
					heapq.heappush(pq, new_state)
					num_pushed += 1
			num_pruned += len(new_states) - num_pushed
			

		end_time = time.time()
		res = {}

		res['soln'] = bssf
		res['time'] = end_time - start_time
		res['count'] = num_solns
		res['cost'] = bssf.cost
		res['max'] = max_queue_size
		res['total'] = num_states
		res['pruned'] = num_pruned


		return res