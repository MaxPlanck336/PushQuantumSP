import math
import time
import numpy as np


class QuboSPBinary1:
    def __init__(self, gra, P1=1, P2=2, P3=2, process=False) -> None:
        self.gra = gra
        self.radar1 = []
        self.radar0 = []     
        self.usedLidars = []
        self.mandatoryLidars = []
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        
        #print("Nodes in the graph:", list(self.gra.G.nodes))
        #print("Adjacency list keys:", list(self.gra.G.adj.keys()))
        if process:
            self.solve_preprocessing(P1, P2, P3)
        self.model = self.__compute_QUBO_Matrix_binary(P1, P2, P3)
        #print("Matrix: ", self.model)
        

    def __inverter_matrix(self, sample):
        solution_dict = {
            f"x_{self.usedLidars[i][0]}_{self.usedLidars[i][1]}_{self.usedLidars[i][2]}_{self.usedLidars[i][3]}_{self.usedLidars[i][4]}": sample[
                i
            ]
            for i in range(len(self.usedLidars))
        }
        return solution_dict

    def solve(self, solve_func, **config):

        start_time = time.time()
        answer = solve_func(Q=self.model, **config)
        solve_time = time.time() - start_time

        solution = self.__inverter_matrix(answer.first.sample)
        info = answer.info

        return {
            "solution": solution,
            "energy": answer.first.energy,
            "runtime": solve_time,
            "info": info,
        }

    def __needed_bitnum(self, decnum):
        if decnum == 0:
            return 0
        return int(math.ceil(math.log2(decnum)))

    def __is_in_list(self, mylist, target):
        for i in mylist:
            if target == i:
                return True
        return False

    def reduce_q(self, lidar):
        #self.usedLidars.remove(lidar)
        #self.mandatoryLidars.remove(lidar)
        self.gra.G.remove_node(lidar)

    def remove_slack_zero(self):
        to_delete = []
        for s in self.gra.G.nodes:
            if not s in to_delete:
                #if street point 
                if len(s) == 3:
                    #calc number of bits needed for slack with log2
                    slackbits = self.__needed_bitnum(len(self.gra.G.adj[s].items()))
                    if slackbits == 0:
                        #get first lidar
                        lidar_s = next(iter(self.gra.G.adj[s].items()))[0]
                        all_sp = self.gra.G.adj[lidar_s]
                        for sp in all_sp:
                            to_delete.append(sp)
                        to_delete.append(lidar_s)
                        self.radar1.append(lidar_s)
                        print("j'ai append !!!!")
        for d in list(set(to_delete)):
            self.gra.G.remove_node(d)

    def find_similar_lidar(self):
        ret = []
        liders = []
        for lidar1 in self.gra.G.nodes:
            if len(lidar1) != 3:
                liders.append(lidar1)
        for lidar1 in liders:
            for lidar2 in liders:
                if lidar1 != lidar2:
                    if lidar1 in self.gra.G.nodes and lidar2 in self.gra.G.nodes:
                        adj1 = self.gra.G.adj[lidar1]
                        adj2 = self.gra.G.adj[lidar2]
                        #look if subset exists
                        if adj1.items() <= adj2.items():
                            self.reduce_q(lidar1)
                            self.radar0.append(lidar1)
                        elif adj2.items() < adj1.items():
                            self.reduce_q(lidar2)
                            self.radar0.append(lidar2)

    def solve_preprocessing(self, P1, P2, P3):
        self.find_similar_lidar()
        self.remove_slack_zero()
        print("#nodes: ", len(self.gra.G.nodes))
        print(self.gra.G.nodes)

    def __compute_QUBO_Matrix_binary(self, P1, P2, P3):
        #initialize variables
        slacksize = 0
        slacklist = []
        #iterate over all nodes in G
        for s in self.gra.G.nodes:
            #if street point
            if len(s) == 3:
                #calc number of bits needed for slack with log2
                slackbits = self.__needed_bitnum(len(self.gra.G.adj[s].items()))

                lidar_per_SP = []
                #iterate over all lidars that are connected to the street point
                for ls in self.gra.G.adj[s].items():
                    
                    self.usedLidars.append(ls[0])
                    lidar_per_SP.append(ls[0])
                    #only one connection to lidar, therefore lidar must be activated
                    if slackbits == 0:
                        self.mandatoryLidars.append(ls[0])

                slacklist.append(
                    [lidar_per_SP, {slacksize + i + 1: 2**i for i in range(slackbits)}]
                )
                slacksize += slackbits

        self.usedLidars = list(set(self.usedLidars))
        ilist = list(range(len(self.usedLidars)))
        usedLidars_index = dict(zip(self.usedLidars, ilist))
        for s in slacklist:
            if s[1]:
                s[1] = {
                    key + len(self.usedLidars) - 1: -value
                    for key, value in s[1].items()
                }

        myQUBOsize = len(self.usedLidars) + slacksize
        myQUBOMatrix = np.zeros([myQUBOsize, myQUBOsize], dtype=float)

        for i in range(0, len(self.usedLidars)):
            myQUBOMatrix[i, i] = P1
            if self.__is_in_list(self.mandatoryLidars, self.usedLidars[i]):
                #space for improvement here!!!
                #p2 random penalty added to ensure that the lidar is activated
                myQUBOMatrix[i, i] -= P2

        for s in slacklist:
            #
            if s[1]:
                sdict = s[1]
                ldict = {}
                for l in s[0]:
                    ldict[usedLidars_index[l]] = 1
                ldict.update(sdict)

                for i in ldict:
                    myQUBOMatrix[i, i] -= 2 * P3 * ldict[i]
                    for j in ldict:
                        myQUBOMatrix[i, j] += P3 * ldict[i] * ldict[j]

        return myQUBOMatrix
