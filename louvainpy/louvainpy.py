#!/usr/bin/env python3

# Lucas Seninge (lseninge)
# Author: Lucas Seninge


"""
From-scratch implementation of louvain community detection. This implementation doesnt rely on any graph package
(networkx or igraph) and can be used on any adjacency numpy array (weighted or not) directly, at the cost of speed.
The goal is to provided a readable OOP implementation for education purposes. To use it:

>> import louvainpy
>> best_partition = louvainpy.run_louvain(adjacency_matrix, matrix_type='sparse', all_levels=False)
>> assignments = best_partition.assignment

The implementation relies on reporting a Partition object, containing the modularity and node assignment to each
community at each level if all_levels=True, only the best level else.
"""

# Import modules
import numpy as np
import random


# Classes
# Class for nodes
class Node:
    def __init__(self, node_id):
        """Constructor for class node. In scRNA-seq, each node corresponds to one cell."""
        # This is the cell index in the matrix, and not its name, although it could easily be added
        self.node_id = node_id
        self.community = node_id  # init community to each node is its own community
        # Edge dict
        self.edges = {}

    # These functions are used to get in and out degrees of Node object
    @property
    def degree(self):
        return len(self.edges)

    def incident_sum_weight(self):
        """k_i in Blondel's paper modularity formulation"""
        k_i = 0
        for edge in self.edges:
            k_i += self.edges[edge].weight
        return k_i

    def incident_community_weight(self, community_node_list):
        """k_i_in in Blondel's paper modularity formulation.
        Check for each node j in C if an edge exist between i and j, and sum for all."""
        k_i_in = 0
        inter_node = set(community_node_list).intersection((self.edges.keys()))
        for node_j in inter_node:
            if node_j != self.node_id:
                k_i_in += self.edges[node_j].weight
        return k_i_in

    # Function to add edge
    def add_edge(self, neighbor_id, weight):
        # Add edge
        self._add_edge_check(self.edges, neighbor_id, weight)

    def _add_edge_check(self, edge_set, neighbor_id, weight):
        """
        Add edges given 2 nodes.
        """
        # If neighbour already in edges, update label
        if neighbor_id in edge_set:
            return "Edge already exist!"
        else:
            edge = Edge(in_node=self.node_id, out_node=neighbor_id, weight=weight)
            # Add to dict
            edge_set[neighbor_id] = edge


# Class edge
class Edge:
    def __init__(self, in_node=-1, out_node=-1, weight=None):
        """Constructor for class Edge. Each edge has an incoming and outgoing node, as well as
        a weight corresponding to the distance between the 2 nodes."""
        self.edge_nodes = [in_node, out_node]
        self.weight = weight


# Class community
class Community:
    def __init__(self, node_list=None):
        """
        Constructor for class Community. To be modified
        """
        # If a single node ID is passed put in list
        if type(node_list)!= list:
            node_list = [node_list]
        self.node_list = node_list
        self.weight_in = 0
        self.weight_tot = 0


# Class graph
class LouvainGraph:
    def __init__(self, adjacency_matrix=None, distance_matrix=None, input_type='dense', node_labels=None):
        """
        Constructor for class LouvainGraph.
        Builds a graph based on an adjacency_matrix and distance_matrix. If input_type is 'Dense',
        we assume that the data has the shape (n_nodes x n_neighbors). A sparse implementation will be done later.

        The variable node_labels is used if the nodes shouldn't be labeled by their indices in the adjacency matrix.
        If None, indices are used to label the nodes. Only valid option for sparse matrix input.
        """
        # Type of input and name for nodes
        self.input_type = input_type
        self.node_labels = node_labels

        # Graph matrices storing data
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix

        # Attributes
        self.nnodes = 0
        self.nedges = 0
        self.ncommunities = 0
        self.node_dict = {}
        self.nodeID_list = []
        self.modularity = 0
        self.community_dict = {}
        # Total weight of link in the network
        self.m = 0

        # If input is dense
        if self.input_type == 'dense':
            self.build_graph_from_dense(self.adjacency_matrix, self.distance_matrix)
        elif self.input_type == 'sparse_connectivities':
            # Check dim to see if sparse
            if self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1]:
                self.build_graph_from_sparse(self.adjacency_matrix)

    # Function to build graph from sparse
    def build_graph_from_sparse(self, adj_mat):
        """
        Create graph for louvain clustering from matrix of connectivities.
        """
        #print("Building graph object...")
        # Iterate over matrices rows to create nodes and edges object
        for cell in range(adj_mat.shape[0]):
            # Check if node labels are to be used
            if self.node_labels is not None:
                node_name = self.node_labels[cell]
            else:
                node_name = cell
            # Create node if doesn't exist yet
            if node_name not in self.node_dict:
                self.add_node(node_name)
            # Get non zero value indices for neighbors - use index here
            if np.sum(adj_mat[cell][:]) > 0:
                if adj_mat[cell][:].ndim > 1:
                    neighbors = np.nonzero(adj_mat[cell][:])[1]
                else:
                    neighbors = np.nonzero(adj_mat[cell][:])[0]
                for n_j in neighbors:
                    # Here, also check for node name and rel-label of necessary
                    if self.node_labels is not None:
                        neighbor_name = self.node_labels[n_j]
                    else:
                        neighbor_name = n_j
                    # Create node if first time we see it
                    if neighbor_name not in self.node_dict:
                        self.add_node(neighbor_name)
                    # Add edge now that we are sure that the node exist in the graph
                    self.add_edge(node_name, neighbor_name, weight=adj_mat[cell, n_j])

        self.ncommunities = len(self.community_dict)
        self.m = np.sum(adj_mat) / 2.
        #print("DONE")
        return

    # Function to build graph from dense
    def build_graph_from_dense(self, adj_mat, dist_mat):
        """
        Fill attributes of class LouvainGraph based on passed data.
        """
        print("Building graph object...")
        if self.node_labels is not None:
            print('Node labels only available for sparse matrix input. Indices will be used to label nodes.')
        # Iterate over matrices rows to create nodes and edges object
        for cell in range(adj_mat.shape[0]):
            # Create node
            self.add_node(cell)
            # Get neighbors, create node if doesnt exist yet and add edges
            for i in range(len(adj_mat[cell][:])):
                # Create node if first time we see it
                if adj_mat[cell][i] not in self.node_dict:
                    self.add_node(adj_mat[cell][i])
                # Add edge now that we are sure that the node exist in the graph
                self.add_edge(cell, adj_mat[cell][i], weight=dist_mat[cell][i])

        self.ncommunities = len(self.community_dict)
        self.m = np.sum(dist_mat) / 2.
        print("DONE")
        return

    def add_node(self, node_id):
        """
        Add node to graph
        """
        new = Node(node_id)
        self.node_dict[node_id] = new
        self.nodeID_list.append(node_id)
        self.community_dict[node_id] = Community(node_id)
        self.nnodes += 1
        return

    def add_edge(self, start, end, weight):
        """
        Add edge to graph between 2 nodes.
        """
        # Some error check
        if start is None or end is None:
            return "Need in/out node to create edge!"

        if start not in self.node_dict or end not in self.node_dict:
            return str(start) + " or " + str(end) + " is not in node dictionary."

        # Now create edge
        old = self.node_dict[start].degree + self.node_dict[end].degree

        self.node_dict[start].add_edge(end, weight)
        self.node_dict[end].add_edge(start, weight)

        new = self.node_dict[start].degree + self.node_dict[end].degree

        if new != old:
            # We added an edge if True
            self.nedges += 1

        return

    def do_louvain(self):
        """Perform the phase 1 of the louvain algorithm as described by Blondel et al., 2008."""
        # Get communities
        community_set = self.community_dict
        # Initialize communities and get weights of links within / outside
        for community in community_set:
            sig_c_in = self._weight_community_in(community_set[community])
            sig_c_tot = self._weight_community_out(community_set[community])
            # Update community weight attributes
            community_set[community].weight_in = sig_c_in
            community_set[community].weight_tot = sig_c_tot

        # Iteration until communities don't evolve anymore
        self._louvain_phase1()

        # update modularity
        self.modularity = self._get_modularity()
        # update number of communities
        self.ncommunities = len(self.community_dict)

        # Done
        return "Phase 1 done"

    def _louvain_phase1(self):
        """
        Performs the inner loop as described by Blondel et al.
        """
        # Stopping heuristic and current modularity
        thrsh = 0.0000001
        curr_mod = self._get_modularity()
        new_mod = curr_mod
        improvement = True
        while improvement:
            curr_mod = new_mod
            nb_move = 0
            # Shuffle node list
            random.shuffle(self.nodeID_list)
            # Iterate over all node of the graph and try putting it neighbor communities
            for node_i in self.nodeID_list:
                loop_weight = self.node_dict[node_i].edges[node_i].weight if node_i in self.node_dict[
                    node_i].edges else 0
                old_c = self.node_dict[node_i].community
                k_i = self.node_dict[node_i].incident_sum_weight()
                # Get communities of neighbors dict with community: k_i_in
                neighbor_com = self._neighbor_communities(node_i)
                # Now remove node from its current community - use get in case it is still in its own community
                self._remove(node_i, old_c, neighbor_com.get(old_c, 0), loop_weight)

                best_c = old_c
                best_incr = 0
                k_i_best = 0
                for neigh_c in neighbor_com:
                    incr = self._gain_modularity(neigh_c, neighbor_com.get(neigh_c, 0), k_i)
                    if incr > best_incr:
                        best_c = neigh_c
                        best_incr = incr
                        k_i_best = neighbor_com.get(neigh_c, 0)

                # Now insert node in best community
                self._insert(node_i, best_c, k_i_best, loop_weight)
                # Destroy old community if empty
                if len(self.community_dict[old_c].node_list) == 0:
                    self.community_dict.pop(old_c)
                # Check if we put it back in old community
                if best_c != old_c:
                    nb_move += 1

            # Now that we passed all node of the graph, check if we moved anything
            new_mod = self._get_modularity()
            if nb_move > 0 and new_mod - curr_mod > thrsh:
                improvement = True
            else:
                improvement = False

    def _neighbor_communities(self, node):
        """
        Access neighboring communities of node_i and get the weight of connections.
        """
        weights = {}
        for neighbor, edge_obj in self.node_dict[node].edges.items():
            # Exclude self loops from ki_in as it will be added later
            if neighbor != node:
                edge_weight = edge_obj.weight
                neighborcom = self.node_dict[neighbor].community
                weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

        return weights

    def _insert(self, node, community, k_i_in, loop_weight):
        """
        Insert node in new community. Loop weight was computed prior to this
        operation.
        """
        # get k_i
        k_i = self.node_dict[node].incident_sum_weight()
        # Moved to best_c
        self.community_dict[community].weight_tot += k_i
        self.community_dict[community].weight_in += (2 * k_i_in + loop_weight)
        # Update
        self.node_dict[node].community = community
        self.community_dict[community].node_list.append(node)
        return

    def _remove(self, node, community, k_i_in, loop_weight):
        """
        Remove node from its old community. Loop weight is computed prior to
        this operation.
        """
        # get k_i
        k_i = self.node_dict[node].incident_sum_weight()
        # Moved to best_c
        self.community_dict[community].weight_tot -= k_i
        self.community_dict[community].weight_in -= (2 * k_i_in + loop_weight)
        # Update
        self.node_dict[node].community = -1
        self.community_dict[community].node_list.remove(node)
        return

    def _weight_community_in(self, community):
        """
        Get weight of nodes within a community
        """
        # Check all node in community
        candidate_to_check = []
        node_in = set(community.node_list)
        for node in node_in:
            for target in self.node_dict[node].edges:
                if target in node_in:
                    candidate_to_check.append((node, target))
        # Now compute total weigth based on retained edges between nodes that are both in community
        weight_in = 0
        for pair in candidate_to_check:
            weight_in += self.node_dict[pair[0]].edges[pair[1]].weight

        return weight_in

    def _weight_community_out(self, community):
        """
        Get weight of all nodes incident to a community
        """
        # Get node of the community
        node_in = set(community.node_list)
        weight_tot = 0
        for node in node_in:
            for target in self.node_dict[node].edges:
                weight_tot += self.node_dict[node].edges[target].weight
        return weight_tot

    def _get_modularity(self):
        """
        Get current modularity of the graph.
        """
        modularity = 0
        # Update modularity of the graph and community weights (sigmas)
        community_set = self.community_dict
        for community in community_set:
            # Update modularity of the graph
            sig_c_in = community_set[community].weight_in
            sig_c_tot = community_set[community].weight_tot
            modularity = modularity + (sig_c_in / (2 * self.m)) - (sig_c_tot / (2 * self.m)) ** 2
        return modularity

    def _gain_modularity(self, community, k_i_in, k_i):
        """
        Get gain of modularity of inserting node in new community as described
        in louvain cpp code.
        """
        sig_tot = self.community_dict[community].weight_tot
        m2 = 2 * self.m

        return k_i_in - (sig_tot * k_i / m2)

    def _weight_between_communities(self, community1, community2):
        """
        Gets weight of links between nodes of 2 communities.
        """
        node_com1 = self.community_dict[community1].node_list
        node_com2 = self.community_dict[community2].node_list
        # Arbitrarily starts with community 1's nodes - check if link exist
        weight_between = 0
        for node in node_com1:
            link_set = set(node_com2).intersection(set(self.node_dict[node].edges.keys()))
            # Sum weight of all links between both communities
            for edge in link_set:
                weight_between += self.node_dict[node].edges[edge].weight

        return weight_between

    # Community to graph function
    def community_to_graph(self):
        """
        Initialize phase 2 of the algorithm by converting communities to a new graph.
        Return an adjacency matrix to initialize a new graph.
        """
        # Initialize empty adjacency matrix of size n_commu x n_commu
        # Initialize a community tracker to get which community corresponds to which index in the
        # adjacency matrix, as well as node inside it to keep track of merged communities in later pass
        new_adj = np.zeros((len(self.community_dict), len(self.community_dict)))
        community_tracker = {}

        # Get an ordering of community to consider
        ordered_community = list(self.community_dict.keys())
        for i in range(len(ordered_community)):
            if i not in community_tracker:
                community_tracker[i] = ordered_community[i]
            for j in range(len(ordered_community)):
                if j not in community_tracker:
                    community_tracker[j] = ordered_community[j]
                # Get self edge as weight of links within node of the same community
                if i == j:
                    new_adj[i][j] = self.community_dict[ordered_community[i]].weight_in
                # Else, sum of links between nodes of communities
                else:
                    weight_ij = self._weight_between_communities(ordered_community[i], ordered_community[j])
                    new_adj[i][j] = weight_ij
        return new_adj, community_tracker

    def community_assignment(self):
        """Returns a dict of node to community assignment."""
        assignment_dict = {}
        for node in self.node_dict:
            assignment_dict[node] = self.node_dict[node].community
        return assignment_dict

    # Summary function
    def summary(self):
        """
        Useful function to quickly display important graph attributes.
        """
        print("Graph summary")
        print(self.nnodes, " nodes")
        print(self.ncommunities, " communities")
        print("Modularity:", self.modularity)
        return


# Class partition
class Partition:
    def __init__(self, n_communities=0, modularity=0, assignment=None):
        """Constructor for class partition. Aims at producing a succint reporter
        object for result of 1 louvain algorithm pass"""
        self.ncommunities = n_communities
        self.modularity = modularity
        self.assignment = assignment

    def get_assignment(self):
        """Simply return the assignment list of the object"""
        return self.assignment


# This function performs all phase of louvain algorithm
def run_louvain(input_adjacency, matrix_type='sparse', all_levels=False):
    """
    Function to run the louvain algorithm. Initialize first graph from data given adjacency matrix type (sparse or dense)
    Run as many pass as necessary until community number doesn't change.
    Args:
        input_adjacency (np.array): adjacency matrix of a graph with value (i,j) being the distance/weight between nodes
        (i,j).
        matrix_type (str): Type of format for the matrix. Accept 'sparse' or 'dense'. By dense we mean a matrix with
        no 0 values (typically , a KNN matrix of size (n,k).
        all_levels (bool): Option to returns all partition of the graph or only the best partition.
    Returns:
        dict_part (dict): Dictionary of partition objects for all levels passed in the louvain algorithm
        (if all_levels is True)
        best_part (obj): Partition object containing the best partitioning for the input graph. Node assignment to
        community are found under Partition.assignment. (if all_levels is False)
    """
    # Determine input type
    if matrix_type is 'sparse':
        matrix_type = 'sparse_connectivities'

    # Initialize variables for louvain clustering
    old_part = -1
    n_pass = 0
    dict_part = {}
    # Init first LouvainGraph
    louvain = LouvainGraph(adjacency_matrix=input_adjacency,
                           input_type=matrix_type, node_labels=None)
    new_part = louvain.ncommunities
    original_assignment = {node: item.community for node, item in louvain.node_dict.items()}
    while new_part != old_part:
        # Run louvain phase 1 for current level
        louvain.do_louvain()
        old_part = new_part
        new_part = louvain.ncommunities
        assignment_dict = louvain.community_assignment()
        original_assignment = {node: assignment_dict[com] for node, com in original_assignment.items()}
        # Mapping cluster numbers to partition original id in order of biggest to smallest
        list_assign = [original_assignment[i] for i in sorted(original_assignment.keys())]
        vals, counts = np.unique(list_assign, return_counts=True)
        order = np.argsort(counts)[::-1]
        map_dict = dict(zip(vals[order], np.arange(len(vals))))
        list_assign = [map_dict[j] for j in list_assign]
        # Export and init next graph
        dict_part[n_pass] = Partition(new_part, louvain.modularity, np.array(list_assign))
        adjacency, tracker = louvain.community_to_graph()
        louvain = LouvainGraph(adjacency_matrix=adjacency,
                               input_type='sparse_connectivities', node_labels=tracker)
        n_pass += 1

    if all_levels:
        return dict_part
    else:
        return dict_part[n_pass-1]
