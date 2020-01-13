from aferl.dataset import Dataset
from aferl.transformation_graph import TransformationGraph
from aferl.transformations import TransformationFactory
import numpy as np
import random
import operator
from graphviz import Digraph
from aferl.utils import diff
import time
import json
import codecs
import copy
import os

class FeatureEngineer:
    def __init__(self, estimator, max_iter=100, learning_rate=0.1, discount_factor=0.99, epsilon=0.15, h_max = 8, cv=5, w = None, datetime_format = None, w_init = np.ones(14), random_state = 123, scoring = 'f1_micro', transformations = None):
        self.max_iter = max_iter
        self.estimator = estimator
        self.cv = cv   
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.h_max = np.inf if h_max == None else h_max
        self.datetime_format = datetime_format
        self.random_state = random_state
        self.scoring = scoring
        self.transformation_factory = TransformationFactory(transformations)
        if w is None:
            self.w = {} 
            for transformation in self.transformation_factory.transformations:
                self.w.update({transformation.name: w_init.copy()})
        else:
            self.w = copy.deepcopy(w)

    def fit(self, X, y, data_info=None, missing_value_mark=None):                
        if type(self.max_iter) is not list:
            self.max_iter = [self.max_iter]
        
        for i in range (0, len(self.max_iter)):
            print("Max iter: " + str(self.max_iter[i]))

            if data_info == None:
                data_info = X.shape[1]*['numeric']
            dataset = Dataset(X.copy(), y.copy(), data_info.copy(), missing_value_mark=missing_value_mark, is_initial=True)
            self.graph = TransformationGraph(dataset, self.transformation_factory, self.estimator, self.cv, self.max_iter[i], self.h_max, self.w, self.random_state, self.scoring)    

            self._fit(self.max_iter[i])
        return self.w     
    
    def transform(self, X, y, missing_value_mark=None):
        transformations = self._get_best_transformations()
        dataset = Dataset(X, y, self.graph.root.dataset.data_info, missing_value_mark=missing_value_mark, is_initial=True)
        for transformation in transformations:
            dataset, _ = transformation.transform(dataset)
        return dataset.X

    def save_transformation_graph(self, path, filename):
        dot = Digraph(filename=os.path.join(path, filename), format='pdf')
        for node in self.graph.nodes:
            dot.node(str(node.id), label=(str(node.id) + ": " + "{0:.3f}".format(node.score)))
        for edge in self.graph.edges:
            dot.edge(str(edge.start_node.id), str(edge.end_node.id), label=edge.transformation.name[0:7] + " - " + edge.exp_type)
        dot.render()

    def save_weights(self, path, filename):
        w = self.w.copy()
        for key in w.keys():
            w[key] = w[key].tolist()
        
        json.dump(w, codecs.open(os.path.join(path, filename + '.json'), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    def _fit(self, max_iter):
        for i in range(0, max_iter):
            print("Iteration: " + str(i+1) + "/" + str(max_iter) + ":")
            if self._take_step() == False:
                break

    def _get_best_transformations(self):
        node = self.graph.get_best_node()
        transformations = []
        while node.is_root() == False:
            transformation = node.in_edges[0].transformation
            transformations.append(transformation)
            node = node.get_parent_node()
        transformations.reverse()
        return transformations

    def _take_step(self):
        if random.uniform(0, 1) < 1 - self.epsilon:
            node, transformation, Q, reward = self._take_policy_step()
        else:
            node, transformation, Q, reward = self._take_random_step()        

        if None not in [node, transformation, Q, reward]:
            self._learn(reward, Q, node, transformation) 
            return True

        return False    

    def _take_random_step(self):
        while True:
            (nodes_transformations) = self._get_transformable_nodes()           
            if(len(nodes_transformations) == 0):
                return (None, None, None, None)

            (node, transformation) = nodes_transformations[random.randint(0, len(nodes_transformations) - 1)]    
            reward = self.graph.apply_transformation(node, transformation, 'r')

            if reward is not None:
                Q = self._get_Q_value(node, transformation)
                return (node, transformation, Q, reward)       

    def _take_policy_step(self):
        proposals = self._get_all_Q_values()

        for (node, transformation, Q) in proposals:
            reward = self.graph.apply_transformation(node, transformation, 'p')
            if reward is not None:
                return (node, transformation, Q, reward)
        
        return (None, None, None, None)

    def _learn(self, reward, Q, node, transformation):      
        learn_factor = self.learning_rate * (reward + self.discount_factor * self._get_Q_max() - Q)
        self.w[transformation.name] = self.w[transformation.name] + np.dot(learn_factor, node.get_state(transformation)) 

    def _get_Q_max(self):
        proporsals = self._get_all_Q_values()

        if len(proporsals) > 0:
            return proporsals[0][2]
        
        return 0

    def _get_all_Q_values(self):
        proposals = self._get_transformable_nodes_with_Q()   
        random.shuffle(proposals) 
        proposals.sort(key = operator.itemgetter(2), reverse=True)
        return proposals

    def _get_Q_value(self, node, transformation):
        return np.dot(self.w[transformation.name], node.get_state(transformation))

    def _get_transformable_nodes_with_Q(self):
        return [(n, t, self._get_Q_value(n, t)) for n in self.graph.nodes if n.depth < self.h_max for t in n.get_possible_transformations()]

    def _get_transformable_nodes(self):
        return [(n, t) for n in self.graph.nodes if n.depth < self.h_max for t in n.get_possible_transformations()]
        
