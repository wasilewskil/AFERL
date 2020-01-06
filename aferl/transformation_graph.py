from sklearn.model_selection import cross_val_score, StratifiedKFold
from operator import attrgetter
import numpy as np
import operator
from aferl.utils import diff, RunWithTimeout
import time

class TransformationGraph:
    def __init__(self, dataset, transformation_factory, estimator, cv, budget, h_max, w, random_state, scoring):
        self.transformation_factory = transformation_factory        
        self.step = 0        
        self.random_state = random_state
        self.cv = cv       
        self.estimator = estimator
        self.nodes = []
        self.edges = [] 
        self.budget = budget
        self.b_ratio = 0
        self.h_max = h_max   
        self.max_depth = 1
        self.max_feature_count_ratio = 1  
        self.scoring = scoring
        self.root = Node(self, dataset, 1)      
        self.max_score = self.root.get_score()
        self.nodes.append(self.root)
        self.transformation_factory.create_transformations()

    def apply_transformation(self, node, transformation, exp_type):
        new_dataset, tc = self._transform_dataset_if_possible(node, transformation)

        if new_dataset is None: 
            node.remove_possible_transformation(transformation.name)
            transformation.add_usage(0)
            return 0

        if new_dataset is False:        
            node.remove_possible_transformation(transformation.name)
            return None

        new_node = Node(self, new_dataset, node.depth + 1)

        if new_node.score is None:        
            node.remove_possible_transformation(transformation.name)
            return None

        new_edge = Edge(node, new_node, tc, exp_type)
        transformation_name = transformation.name
        
        new_node.transformations_on_path = node.transformations_on_path.copy()
        new_node.transformations_on_path.update( {transformation_name: new_node.transformations_on_path.setdefault(transformation_name, 0) + 1} )

        new_node.in_edges.append(new_edge)  

        new_edge.start_node.out_edges.append(new_edge)
        self.nodes.append(new_edge.end_node)
        new_edge.end_node.id = len(self.nodes) - 1
        self.edges.append(new_edge)

        new_max = max([self.max_score, new_node.get_score()])
        reward = new_max - self.max_score
        self.max_score = new_max
        self.max_depth = max([self.max_depth, new_node.depth])
        self.max_feature_count_ratio = max([self.max_feature_count_ratio, new_node.feature_count_ratio])

        self.step = self.step + 1
        self.b_ratio = self.step / self.budget   
                   
        transformation.add_usage(reward)
        node.remove_possible_transformation(transformation.name)

        return reward

    def get_best_node(self):
        return max(np.array(self.nodes)[1:,], key=(lambda x: (round(x.get_score(), 3), -x.depth)))

    def _transform_dataset_if_possible(self, node, transformation):
        try:
            print(str(node.id) + " " + transformation.name)
            if transformation not in node.get_possible_transformations():
                return False, False

            new_dataset, tc = node.transform(transformation)

            if new_dataset is None:
                return None, None

            if next((True for node in self.nodes if node.dataset is not None and new_dataset.X.shape == node.dataset.X.shape and np.allclose(new_dataset.X, node.dataset.X, equal_nan = True)), False):
                node.remove_possible_transformation(transformation.name)
                return False, False

            return new_dataset, tc
        except Exception as e:
            print(str(node.id) + " " + transformation.name)
            print(e)
            return None, None

class Node:
    def __init__(self, graph, dataset, depth, in_edges=None, out_edges=None):
        self.dataset = dataset
        self.in_edges = in_edges if in_edges is not None else []
        self.out_edges = out_edges if out_edges is not None else []
        self.transformations_on_path = {}
        self.depth = depth
        self._graph = graph        
        self.id = graph.step              
        self.feature_count_ratio = self.dataset.X.shape[1]/self._graph.root.dataset.X.shape[1] if depth > 1 else 1                
        self._parent_score = None
        self._grandparent_score = None
        self._possible_transformations = set()

        self._evaluate_score()        
        self._build_possible_transformations() 
        if self._graph.h_max == self.depth or self.score >= 1:
            #Saving RAM, since this node will not be transformed in the future
            self.dataset = None
            self._possible_transformations = set()  

    def is_root(self):
        return len(self.in_edges) == 0

    def transform(self, transformation):
        return transformation.transform(self.dataset)

    def get_parent_node(self):
        return self.in_edges[0].start_node if len(self.in_edges) > 0 else None    

    def get_score(self):
        return self.score if self.score is not None else 0
    
    def get_state(self, transformation):
        state = []
        trans_on_path = self.transformations_on_path.setdefault(transformation.name, 0)
        d_max = self._graph.h_max if self._graph.h_max > 0 and self._graph.h_max is not None else self._graph.max_depth

        state.append(self.get_score())                                                                      #0. nodes's score
        state.append(transformation.get_avg_reward())                                                       #1. Transformation's avg. imm. reward
        state.append((trans_on_path / self.depth - 1) if self.depth > 1 else 0)                             #2. No. times transformation has been used
        state.append(self.get_score() - self._get_parent_score())                                           #3. Score gain from parent
        state.append(self.get_score() - self._get_grandparent_score())                                      #4. Score gain from grand-parent
        state.append(-self.depth / d_max)                                                                   #5. nodes's depth
        state.append(self._graph.b_ratio)                                                                   #6. Budget used
        state.append(self.feature_count_ratio / self._graph.max_feature_count_ratio)                        #7. Ratio of feature counts compared to initial dataset
        state.append(transformation.is_feature_selection())                                                 #8. Is feature selection
        state.append(self.dataset.contains_type('numeric'))                                                 #9. Dataset contains numeric
        state.append(self.dataset.contains_type('datetime') or self.dataset.contains_type('timestamp'))     #10. Dataset contains datetime
        state.append(self.dataset.contains_type('nominal'))                                                 #11. Dataset contains string
        state.append(self.dataset.contains_missing_values())                                                #12. Dataset contains missing values
        state.append(transformation.is_imputing())                                                          #13. Is imputing
        
        return [x/len(state) for x in state]
    
    def get_possible_transformations(self):  
        return [t for t in self._graph.transformation_factory.transformations if t.name in self._possible_transformations]

    def remove_possible_transformation(self, transformation_name):
        if transformation_name in self._possible_transformations:
            self._possible_transformations.remove(transformation_name)

    def _evaluate_score(self):
        try:
            print("Calculating score...")
            cv = StratifiedKFold(n_splits=self._graph.cv, random_state=self._graph.random_state) 
            cv_scores = cross_val_score(self._graph.estimator, self.dataset.X, self.dataset.y, cv=cv, error_score=0, scoring=self._graph.scoring, n_jobs = -1)
            self.score = cv_scores.mean()
        except Exception as e:
            print(e)
            self.score = 0

    def _build_possible_transformations(self):
        if round(self.score, 3) >= 1:
            self._possible_transformations = set()
            return

        pt = self._graph.transformation_factory.transformations

        if self.dataset.contains_missing_values() == False:
            pt = [t for t in pt if t.is_imputing() == False]

        if self.dataset.contains_type('nominal') == False:
            pt = [t for t in pt if t.is_encoding() == False]
        
        for transformation in pt:
            self._possible_transformations.add(transformation.name)

    def _get_parent_score(self):
        if self._parent_score is None:
            parent = self.get_parent_node()
            score = parent.get_score() if parent != None else self.get_score()
            self._parent_score = score

        return self._parent_score
    
    def _get_grandparent_score(self):
        if self._grandparent_score is None:
            parent = self.get_parent_node()
            grandParent = parent.get_parent_node() if parent != None else None
            score = grandParent.get_score() if grandParent != None else self.get_score()
            self._grandparent_score = score

        return self._grandparent_score

class Edge:
    def __init__(self, start_node, end_node, transformation, exp_type):
        self.transformation = transformation
        self.start_node = start_node
        self.end_node = end_node
        self.exp_type = exp_type   
