#define LIBBOT_PRESENT 0

#include <ctime>
#include <fstream>
#include <iostream>

#include <bot_core/bot_core.h>

#include <lcm/lcm.h>
#include <random>
#include <lcmtypes/lcmtypes.h>
#include <google/cloud/storage/client.h>

#include "rrts.hpp"
#include "system_single_integrator.h"
#include <list>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>

using namespace RRTstar;
using namespace SingleIntegrator;

using namespace std;

// int sw=1;
// int algo=0;
typedef Planner<State, Trajectory, System> planner_t;
typedef Vertex<State, Trajectory, System> vertex_t;

class obst {
public:
  double center[3];
  double size[3];
  double radius;
};

bool check(double *first, double *second) {

	if (first[0] == second[0] && first[1] == second[1])
	{
		return true;
	}
	else
	{
		return false;
	}
}
namespace gcs = google::cloud::storage;

int size = 50000;

int publishTree(lcm_t *lcm, planner_t &planner, System &system);

int publishPC(lcm_t *lcm, double nodes[8000][2], int sze, System &system);

int publishTraj(lcm_t *lcm,
                planner_t &planner,
                System &system,
                int num,
                string fod,
                google::cloud::StatusOr <gcs::Client> client);
// lcm_t *lcm, region& regionOperating, region& regionGoal,list<region*>& obstacles
// int publishEnvironment (lcm_t *lcm);

int publishEnvironment(lcm_t *lcm, region &regionOperating, region &regionGoal, list<region *> &obstacles);

pair<float, float> getFreeSpacePoint(vector <pair<float, float>> perm);

bool validPoint(float x, float y, vector <pair<float, float>> perm, float obsSize);

void printEnv(vector <pair<float, float>> perm);

void printStartGoal(pair<float, float> start, pair<float, float> goal);

pair<float, float> getFreeSpacePoint(vector <pair<float, float>> perm) {
	float mapSize = 40;
	float obsSize = 5;

	random_device rd{};
	mt19937 generator(rd());

	float x, y = 0;
	uniform_int_distribution<int> distribution(-mapSize * 50, mapSize * 50);
	do
	{
		x = distribution(generator) / 100.0;
		y = distribution(generator) / 100.0;
	}
	while (not validPoint(x, y, perm, obsSize));
	return make_pair(x, y);
}

bool validPoint(float x, float y, vector <pair<float, float>> perm, float obsSize) {
	pair<float, float> dist;
	for (auto obstacle = perm.begin(); obstacle != perm.end(); obstacle++)
	{
		dist.first = abs(x - obstacle->first);
		dist.second = abs(y - obstacle->second);
		if (dist.first < obsSize / 2 and dist.second < obsSize / 2)
		{
			return false;
		}
	}
	return true;
}

auto readCsv(gcs::ObjectReadStream *myFile) {
	// Reads a CSV file into a vector of <string, vector<int>> pairs where
	// each pair represents <column name, column values>

	// Create a vector of <string, int vector> pairs to store the result
	vector < vector < pair < float, float>>> perm;
	vector <pair<float, float>> obstacles;

	// Helper vars
	std::string line;
	std::string val;

	char delim = ',';
	// Read data, line by line
	while (std::getline(*myFile, line))
	{
		// Create a stringstream of the current line
		obstacles.clear();
		int current = 0;

		stringstream sstream(line);
		pair<float, float> data;
		while (std::getline(sstream, val, delim))
		{
			if (!current)
			{
				current = 1;
				data.first = stof(val);
			}
			else
			{
				current = 0;
				data.second = stof(val);
				obstacles.push_back(data);
			}
		}

		perm.push_back(obstacles);
	}
	obstacles.clear();

	// Close file
	myFile->Close();

	return perm;
}

void printEnv(vector <pair<float, float>> perm) {
	cout << "\nEnvironment:\n[";
	for (auto x = perm.begin(); x != perm.end(); x++)
	{
		cout << "[" << x->first << ", " << x->second << "]\t";
	}
	cout << "]\n";
}

void printStartGoal(pair<float, float> start, pair<float, float> goal) {
	cout << "\n\n Start -> [" << get<0>(start) << ", " << get<1>(start) << "]\n";
	cout << " Goal -> [" << get<0>(goal) << ", " << get<1>(goal) << "]\n\n";
}

int main(int argc, char **argv) {

	google::cloud::StatusOr <gcs::Client> client = gcs::Client::CreateDefaultClient();
	if (!client)
	{
		std::cerr << "Failed to create Storage Client, status=" << client.status()
		          << "\n";
		return 1;
	}
	int startId = 0;
	int numRuns = 5;
	switch (argc)
	{
	case 2: startId = atoi(argv[1]);
		break;
	case 3: startId = atoi(argv[1]);
		numRuns = atoi(argv[2]);
		break;
	}

	srand(time(0));

	gcs::ObjectReadStream reader = client->ReadObject("mpnet-bucket", "obs/perm.csv");

	if (!reader)
	{
		std::cerr << "Error reading object: " << reader.status() << "\n";
		return 1;
	}

	// We drop 7 obstacle blocks in the workspace to generate random environments using 20P7=77520
	// permutations. Note that we can have now 77520 different environments but we use 110 envs only
	vector < vector < pair < float, float>>>  perm = readCsv(&reader);

	// start and goal region
	/*
	We also generted a random set of nodes from obstacle-free space, denoted as graph. These nodes are used as
	start and goal pairs
	*/
	pair<float, float> goalPoint;
	pair<float, float> startPoint;

	for (int env_no = startId; env_no < startId + numRuns; env_no++)
	{
		for (int idx = 0; idx < 4000; idx++)
		{
			cout << "idx " << idx << endl;

			planner_t rrts;

			cout << "RRTstar is alive" << endl;

			// Get lcm
			lcm_t *lcm = bot_lcm_get_global(NULL);

			goalPoint = getFreeSpacePoint(perm[env_no]);
			startPoint = getFreeSpacePoint(perm[env_no]);

			printEnv(perm[idx]);
			printStartGoal(startPoint, goalPoint);

			// Create the dynamical system
			System system;

			// Three dimensional configuration space
			system.setNumDimensions(2);
			// Define the operating region
			system.regionOperating.setNumDimensions(2);
			system.regionOperating.center[0] = 0.0;
			system.regionOperating.center[1] = 0.0;
//		system.regionOperating.center[2] = 0.0;
			system.regionOperating.size[0] = 40.0;
			system.regionOperating.size[1] = 40.0;
//		system.regionOperating.size[2] = 0.0;
			// Define the goal region
			system.regionGoal.setNumDimensions(2);
			system.regionGoal.center[0] = goalPoint.first;
			system.regionGoal.center[1] = goalPoint.second;
//		system.regionGoal.center[2] = 0.0;
			system.regionGoal.size[0] = 1.0;
			system.regionGoal.size[1] = 1.0;
//		system.regionGoal.size[2] = 0.0;
			region *obstacle, *obstacle1, *obstacle2, *obstacle3, *obstacle4, *obstacle5, *obstacle6;
			obstacle = new region;
			obstacle1 = new region;
			obstacle2 = new region;
			obstacle3 = new region;
			obstacle4 = new region;
			obstacle5 = new region;
			obstacle6 = new region;

			obstacle->setNumDimensions(2);
			obstacle->center[0] = perm[env_no][0].first;
			obstacle->center[1] = perm[env_no][0].second;
//		obstacle->center[2] = 0.0;
			obstacle->size[0] = 5.0;
			obstacle->size[1] = 5.0;
//		obstacle->size[2] = 0.0;

			obstacle1->setNumDimensions(2);
			obstacle1->center[0] = perm[env_no][1].first;
			obstacle1->center[1] = perm[env_no][1].second;
//		obstacle1->center[2] = 0.0;
			obstacle1->size[0] = 5.0;
			obstacle1->size[1] = 5.0;
//		obstacle1->size[2] = 0.0;

			obstacle2->setNumDimensions(2);
			obstacle2->center[0] = perm[env_no][2].first;
			obstacle2->center[1] = perm[env_no][2].second;
//		obstacle2->center[2] = 0.0;
			obstacle2->size[0] = 5.0;
			obstacle2->size[1] = 5.0;
//		obstacle2->size[2] = 0.0;

			obstacle3->setNumDimensions(2);
			obstacle3->center[0] = perm[env_no][3].first;
			obstacle3->center[1] = perm[env_no][3].second;
//		obstacle3->center[2] = 0.0;
			obstacle3->size[0] = 5.0;
			obstacle3->size[1] = 5.0;
//		obstacle3->size[2] = 0.0;

			obstacle4->setNumDimensions(2);
			obstacle4->center[0] = perm[env_no][4].first;
			obstacle4->center[1] = perm[env_no][4].second;
//		obstacle4->center[2] = 0.0;
			obstacle4->size[0] = 5.0;
			obstacle4->size[1] = 5.0;
//		obstacle4->size[2] = 0.0;

			obstacle5->setNumDimensions(2);
			obstacle5->center[0] = perm[env_no][5].first;
			obstacle5->center[1] = perm[env_no][5].second;
//		obstacle5->center[2] = 0.0;
			obstacle5->size[0] = 5.0;
			obstacle5->size[1] = 5.0;
//		obstacle5->size[2] = 0.0;

			obstacle6->setNumDimensions(2);
			obstacle6->center[0] = perm[env_no][6].first;
			obstacle6->center[1] = perm[env_no][6].second;
//		obstacle6->center[2] = 0.0;
			obstacle6->size[0] = 5.0;
			obstacle6->size[1] = 5.0;
//		obstacle6->size[2] = 0.0;

			system.obstacles.push_front(obstacle);  // Add the obstacle to the list
			system.obstacles.push_front(obstacle1); // Add the obstacle to the list
			system.obstacles.push_front(obstacle2); // Add the obstacle to the list
			system.obstacles.push_front(obstacle3); // Add the obstacle to the list
			system.obstacles.push_front(obstacle4); // Add the obstacle to the list
			system.obstacles.push_front(obstacle5);
			system.obstacles.push_front(obstacle6);

			// Add the system to the planner
			rrts.setSystem(system);

//		publishEnvironment(lcm, system.regionOperating, system.regionGoal, system.obstacles);
			// Set up the root vertex
			vertex_t &root = rrts.getRootVertex();
			State &rootState = root.getState();

			// Define start state
			rootState[0] = startPoint.first;
			rootState[1] = startPoint.second;

			// Initialize the planner
			rrts.initialize();

			// This parameter should be larger than 1.5 for asymptotic
			//   optimality. Larger values will weigh on optimization
			//   rather than exploration in the RRT* algorithm. Lower
			//   values, such as 0.1, should recover the RRT.
			rrts.setGamma(1.5);

			clock_t start = clock();
			int j = 0;
			double node[2];

			// random obstacle-free nodes generation. These nodes were generated to form random start and goal
			// pairs.


			// p-rrt* path generation
			double cost = 1000;
			int k = 0;
			int c = 0, cp = 0;
			for (int j = 0; j <= 100000; j += 2000)
			{

				int limit = 5000 + j;

				while (k < limit)
				{

					rrts.iteration(node, -1, -1);
					k++;
				}
				vertex_t &vertexBest = rrts.getBestVertex();
				if (&vertexBest != NULL)
				{
					if (vertexBest.costFromRoot < cost)
					{
						cost = vertexBest.costFromRoot;
						c++;
					}
				}
				if (cp != c)
				{
					cp = c;
				}
				else
				{
					break;
				}
			}

			cout << "iterations:" << k << endl;

			// Run the algorithm for 10000 iteartions
			clock_t finish = clock();
			cout << "Time : " << ((double) (finish - start)) / CLOCKS_PER_SEC << endl;

			// publishTree (lcm, rrts, system);
			// stores path in the folder env_no
			auto result = publishTraj(lcm, rrts, system, idx, to_string(env_no), client);
			if (!result)
			{
				idx--;
			}

		}
	}

	return 1;
}

//int publishEnvironment(lcm_t *lcm, region &regionOperating, region &regionGoal, list<region *> &obstacles) {
//
//	// Publish the environment
//	lcmtypes_environment_t *environment = (lcmtypes_environment_t *) malloc(sizeof(lcmtypes_environment_t));
//
//	environment->operating.center[0] = regionOperating.center[0];
//	environment->operating.center[1] = regionOperating.center[1];
//	environment->operating.center[2] = 0.0;
//	environment->operating.size[0] = regionOperating.size[0];
//	environment->operating.size[1] = regionOperating.size[1];
//	environment->operating.size[2] = 0.0;
//
//	environment->goal.center[0] = regionGoal.center[0];
//	environment->goal.center[1] = regionGoal.center[1];
//	environment->goal.center[2] = 0.0;
//	environment->goal.size[0] = regionGoal.size[0];
//	environment->goal.size[1] = regionGoal.size[1];
//	environment->goal.size[2] = 0.0;
//
//	environment->num_obstacles = obstacles.size();
//
//	if (environment->num_obstacles > 0)
//	{
//		environment->obstacles = (lcmtypes_region_3d_t *) malloc(sizeof(lcmtypes_region_3d_t));
//	}
//
//	int idx_obstacles = 0;
//	for (list<region *>::iterator iter = obstacles.begin(); iter != obstacles.end(); iter++)
//	{
//
//		region *obstacleCurr = *iter;
//
//		environment->obstacles[idx_obstacles].center[0] = obstacleCurr->center[0];
//		environment->obstacles[idx_obstacles].center[1] = obstacleCurr->center[1];
//		environment->obstacles[idx_obstacles].center[2] = 0.0;
//		environment->obstacles[idx_obstacles].size[0] = obstacleCurr->size[0];
//		environment->obstacles[idx_obstacles].size[1] = obstacleCurr->size[1];
//		environment->obstacles[idx_obstacles].size[2] = 0.0;
//
//		idx_obstacles++;
//	}
//
//	lcmtypes_environment_t_publish(lcm, "ENVIRONMENT", environment);
//	lcmtypes_environment_t_destroy(environment);
//
//	return 1;
//}

int publishTraj(lcm_t *lcm,
                planner_t &planner,
                System &system,
                int num,
                string fod,
                google::cloud::StatusOr <gcs::Client> client) {

	auto writer = client->WriteObject("mpnet-bucket", "env/e" + fod + "/path" + to_string(num) + ".dat");

	cout << "Publishing trajectory -- start" << endl;

	vertex_t &vertexBest = planner.getBestVertex();

	if (&vertexBest == NULL)
	{
		cout << "No best vertex" << endl;
		writer.Close();
		return 0;
	}

	cout << "Cost From root " << vertexBest.costFromRoot << endl;
	list<double *> stateList;
	planner.getBestTrajectory(stateList);
	lcmtypes_trajectory_t *opttraj = (lcmtypes_trajectory_t *) malloc(sizeof(lcmtypes_trajectory_t));
	opttraj->num_states = stateList.size();
	opttraj->states = (lcmtypes_state_t *) malloc(opttraj->num_states * sizeof(lcmtypes_state_t));
	int psize = (stateList.size() - 1) / 2 + 1;
	int pindex = 0;
	double path[psize][2];
	int stateIndex = 0;

	for (list<double *>::iterator iter = stateList.begin(); iter != stateList.end(); iter++)
	{

		double *stateRef = *iter;
		opttraj->states[stateIndex].x = stateRef[0];
		opttraj->states[stateIndex].y = stateRef[1];
		if (pindex > 0)
		{

			if (path[pindex - 1][0] != stateRef[0])
			{
				path[pindex][0] = stateRef[0];
				path[pindex][1] = stateRef[1];
				pindex++;
			}
		}
		else
		{
			path[pindex][0] = stateRef[0];
			path[pindex][1] = stateRef[1];
			pindex++;
		}

		if (system.getNumDimensions() > 2)
		{
			opttraj->states[stateIndex].z = stateRef[2];
		}
		else
		{
			opttraj->states[stateIndex].z = 0.0;
		}

		free(stateRef);

		stateIndex++;
	}

	writer.write((char *) &path, sizeof path);

	if (writer.metadata())
	{
		std::cout << "Successfully created object: " << *writer.metadata() << "\n";
	}
	else
	{
		std::cerr << "Error creating object: " << writer.metadata().status() << "\n";
		return 1;
	}
	writer.Close();

	lcmtypes_trajectory_t_publish(lcm, "TRAJECTORY", opttraj);

	lcmtypes_trajectory_t_destroy(opttraj);

	cout << "Publishing trajectory -- end" << endl;

	return 1;
}

int publishTree(lcm_t *lcm, planner_t &planner, System &system) {

	cout << "Publishing the tree -- start" << endl;

	bool plot3d = (system.getNumDimensions() > 2);

	lcmtypes_graph_t *graph = (lcmtypes_graph_t *) malloc(sizeof(lcmtypes_graph_t));
	graph->num_vertices = planner.numVertices;
	cout << "num_Vertices: " << graph->num_vertices << endl;

	if (graph->num_vertices > 0)
	{

		graph->vertices = (lcmtypes_vertex_t *) malloc(graph->num_vertices * sizeof(lcmtypes_vertex_t));

		int vertexIndex = 0;
		for (list<vertex_t *>::iterator iter = planner.listVertices.begin();
		     iter != planner.listVertices.end(); iter++)
		{

			vertex_t &vertexCurr = **iter;
			State &stateCurr = vertexCurr.getState();

			graph->vertices[vertexIndex].state.x = stateCurr[0];
			graph->vertices[vertexIndex].state.y = stateCurr[1];
			if (plot3d)
			{
				graph->vertices[vertexIndex].state.z = stateCurr[2];
			}
			else
			{
				graph->vertices[vertexIndex].state.z = 0.0;
			}

			vertexIndex++;
		}
	}
	else
	{
		graph->vertices = NULL;
	}

	if (graph->num_vertices > 1)
	{

		graph->num_edges = graph->num_vertices - 1;
		graph->edges = (lcmtypes_edge_t *) malloc(graph->num_edges * sizeof(lcmtypes_edge_t));

		int edgeIndex = 0;
		for (list<vertex_t *>::iterator iter = planner.listVertices.begin();
		     iter != planner.listVertices.end(); iter++)
		{

			vertex_t &vertexCurr = **iter;

			vertex_t &vertexParent = vertexCurr.getParent();

			if (&vertexParent == NULL)
			{
				continue;
			}

			State &stateCurr = vertexCurr.getState();
			State &stateParent = vertexParent.getState();

			graph->edges[edgeIndex].vertex_src.state.x = stateParent[0];
			graph->edges[edgeIndex].vertex_src.state.y = stateParent[1];
			if (plot3d)
			{
				graph->edges[edgeIndex].vertex_src.state.z = stateParent[2];
			}
			else
			{
				graph->edges[edgeIndex].vertex_src.state.z = 0.0;
			}

			graph->edges[edgeIndex].vertex_dst.state.x = stateCurr[0];
			graph->edges[edgeIndex].vertex_dst.state.y = stateCurr[1];
			if (plot3d)
			{
				graph->edges[edgeIndex].vertex_dst.state.z = stateCurr[2];
			}
			else
			{
				graph->edges[edgeIndex].vertex_dst.state.z = 0.0;
			}

			graph->edges[edgeIndex].trajectory.num_states = 0;
			graph->edges[edgeIndex].trajectory.states = NULL;

			edgeIndex++;
		}

	}
	else
	{
		graph->num_edges = 0;
		graph->edges = NULL;
	}

	lcmtypes_graph_t_publish(lcm, "GRAPH", graph);

	lcmtypes_graph_t_destroy(graph);

	cout << "Publishing the tree -- end" << endl;

	return 1;
}
