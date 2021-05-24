#include <bot_core/bot_core.h>
#include "lcm/lcm.h"
#include "rrts.hpp"
#include "system_single_integrator.h"
#include "string"
#include "time.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace RRTstar;
using namespace SingleIntegrator;

using json = nlohmann::json;

using namespace std;

typedef Planner<State, Trajectory, System> planner_t;
typedef Vertex<State, Trajectory, System> vertex_t;

class obst {
public:
  double center[3];
  double size[3];
  double radius;
};
int publishTraj(lcm_t *lcm, planner_t &planner, System &system, int num, string fod, string filepath, string envFolder);

auto readCsv(ifstream *myFile) {
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
	myFile->close();

	return perm;
}
int main(int argc, char **argv) {
	string filepath = __FILE__;
	size_t strIdx = filepath.find("mpnet");
	filepath = filepath.substr(0, strIdx + 5);

	ifstream points(filepath + "/data/selected_points.json");

	json envJson;
	envJson = json::parse(points);

	ifstream reader(filepath + "/obs/perm.csv");

	vector < vector < pair < float, float>>>  perm = readCsv(&reader);
	vector<float> timing;

	json timingsJson;
	try
	{
		ifstream timings(filepath + "/data/rrt_timings.json");
		timingsJson = json::parse(timings);
	}
	catch (exception)
	{
	}

	for (auto &el : envJson.items())
	{
		int envId = stoi(el.key());
		cout << "\n\t ENV " << envId << "\n\n";

		cout << "Size: " << timingsJson[to_string(envId)].size() << endl;
		size_t startIdx = timingsJson[to_string(envId)].size();

		vector < vector < vector < float>>> values = el.value();

		values = vector < vector < vector < float>>>(values.begin() + startIdx, values.end());

		for (auto &trajectory: values)
		{
			planner_t rrts;

			vector<float> startPoint = trajectory[0];
			vector<float> goalPoint = trajectory[1];

			System system;

			system.setNumDimensions(2);
			system.regionOperating.setNumDimensions(2);
			system.regionOperating.center[0] = 0.0;
			system.regionOperating.center[1] = 0.0;
//		system.regionOperating.center[2] = 0.0;
			system.regionOperating.size[0] = 40.0;
			system.regionOperating.size[1] = 40.0;
//		system.regionOperating.size[2] = 0.0;
			// Define the goal region
			system.regionGoal.setNumDimensions(2);
			system.regionGoal.center[0] = goalPoint[0];
			system.regionGoal.center[1] = goalPoint[1];
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
			obstacle->center[0] = perm[envId][0].first;
			obstacle->center[1] = perm[envId][0].second;
//		obstacle->center[2] = 0.0;
			obstacle->size[0] = 5.0;
			obstacle->size[1] = 5.0;
//		obstacle->size[2] = 0.0;

			obstacle1->setNumDimensions(2);
			obstacle1->center[0] = perm[envId][1].first;
			obstacle1->center[1] = perm[envId][1].second;
//		obstacle1->center[2] = 0.0;
			obstacle1->size[0] = 5.0;
			obstacle1->size[1] = 5.0;
//		obstacle1->size[2] = 0.0;

			obstacle2->setNumDimensions(2);
			obstacle2->center[0] = perm[envId][2].first;
			obstacle2->center[1] = perm[envId][2].second;
//		obstacle2->center[2] = 0.0;
			obstacle2->size[0] = 5.0;
			obstacle2->size[1] = 5.0;
//		obstacle2->size[2] = 0.0;

			obstacle3->setNumDimensions(2);
			obstacle3->center[0] = perm[envId][3].first;
			obstacle3->center[1] = perm[envId][3].second;
//		obstacle3->center[2] = 0.0;
			obstacle3->size[0] = 5.0;
			obstacle3->size[1] = 5.0;
//		obstacle3->size[2] = 0.0;

			obstacle4->setNumDimensions(2);
			obstacle4->center[0] = perm[envId][4].first;
			obstacle4->center[1] = perm[envId][4].second;
//		obstacle4->center[2] = 0.0;
			obstacle4->size[0] = 5.0;
			obstacle4->size[1] = 5.0;
//		obstacle4->size[2] = 0.0;

			obstacle5->setNumDimensions(2);
			obstacle5->center[0] = perm[envId][5].first;
			obstacle5->center[1] = perm[envId][5].second;
//		obstacle5->center[2] = 0.0;
			obstacle5->size[0] = 5.0;
			obstacle5->size[1] = 5.0;
//		obstacle5->size[2] = 0.0;

			obstacle6->setNumDimensions(2);
			obstacle6->center[0] = perm[envId][6].first;
			obstacle6->center[1] = perm[envId][6].second;
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
			rootState[0] = startPoint[0];
			rootState[1] = startPoint[1];

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


			// Run the algorithm for 10000 iteartions
			list<double *> stateList;
			rrts.getBestTrajectory(stateList);
			clock_t finish = clock();
			timing.push_back(((float) (finish - start)) / CLOCKS_PER_SEC);

			// publishTree (lcm, rrts, system);
			// stores path in the folder env_no
//			auto result = publishTraj(lcm, rrts, system, idx, to_string(envId), filepath, envFolder);
//			if (!result)
//			{
//				idx--;
//			}
		}
		cout << "\n";
		timingsJson[to_string(envId)] = timing;
		timing.clear();

		ofstream output(filepath + "/data/rrt_timings.json");
		output << timingsJson;

		output.close();
	}

	return 0;

}