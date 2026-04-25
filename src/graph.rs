//! Graph Theory and Network Analysis
//!
//! Provides algorithms for pathfinding, network centrality, and spanning trees.
//! 
//! # Features
//! - **Shortest Path**: Dijkstra, A*, Bellman-Ford
//! - **Minimum Spanning Tree**: Prim's Algorithm
//! - **Network Analysis**: PageRank

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use crate::errors::{SciError, SciResult};

#[derive(Debug, Clone)]
pub struct Graph {
    pub directed: bool,
    pub adjacency_list: HashMap<usize, Vec<(usize, f64)>>,
}

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    position: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Notice that we flip the ordering on costs.
        // In case of a tie we compare positions - this step is necessary
        // to make implementations of `PartialEq` and `Ord` consistent.
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, PartialEq)]
struct EdgeState {
    cost: f64,
    u: usize,
    v: usize,
}

impl Eq for EdgeState {}

impl Ord for EdgeState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for EdgeState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Graph {
    pub fn new(directed: bool) -> Self {
        Self {
            directed,
            adjacency_list: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: usize) {
        self.adjacency_list.entry(node).or_insert_with(Vec::new);
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adjacency_list.entry(u).or_insert_with(Vec::new).push((v, weight));
        if !self.directed && u != v {
            self.adjacency_list.entry(v).or_insert_with(Vec::new).push((u, weight));
        }
    }

    pub fn nodes(&self) -> Vec<usize> {
        self.adjacency_list.keys().copied().collect()
    }

    /// Dijkstra's Algorithm for shortest paths from a source node.
    /// Returns a tuple of (distances, predecessors).
    pub fn dijkstra(&self, source: usize) -> (HashMap<usize, f64>, HashMap<usize, usize>) {
        let mut dist = HashMap::new();
        let mut prev = HashMap::new();
        let mut heap = BinaryHeap::new();

        for &node in self.adjacency_list.keys() {
            dist.insert(node, f64::INFINITY);
        }
        
        dist.insert(source, 0.0);
        heap.push(State { cost: 0.0, position: source });

        while let Some(State { cost, position }) = heap.pop() {
            if cost > *dist.get(&position).unwrap_or(&f64::INFINITY) {
                continue;
            }

            if let Some(neighbors) = self.adjacency_list.get(&position) {
                for &(neighbor, weight) in neighbors {
                    let next_cost = cost + weight;
                    if next_cost < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        heap.push(State { cost: next_cost, position: neighbor });
                        dist.insert(neighbor, next_cost);
                        prev.insert(neighbor, position);
                    }
                }
            }
        }

        (dist, prev)
    }

    /// A* Search Algorithm. 
    /// Takes a heuristic closure that estimates the cost from a node to the target.
    /// Returns the shortest path from start to target.
    pub fn a_star<H>(&self, start: usize, target: usize, heuristic: H) -> Option<(f64, Vec<usize>)> 
    where
        H: Fn(usize) -> f64,
    {
        let mut dist = HashMap::new();
        let mut prev = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start, 0.0);
        heap.push(State { cost: heuristic(start), position: start });

        while let Some(State { cost: _f_score, position }) = heap.pop() {
            if position == target {
                let mut path = Vec::new();
                let mut current = target;
                while current != start {
                    path.push(current);
                    current = prev[&current];
                }
                path.push(start);
                path.reverse();
                return Some((dist[&target], path));
            }

            let g_score = *dist.get(&position).unwrap_or(&f64::INFINITY);

            if let Some(neighbors) = self.adjacency_list.get(&position) {
                for &(neighbor, weight) in neighbors {
                    let tentative_g = g_score + weight;
                    if tentative_g < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        prev.insert(neighbor, position);
                        dist.insert(neighbor, tentative_g);
                        let f_score = tentative_g + heuristic(neighbor);
                        heap.push(State { cost: f_score, position: neighbor });
                    }
                }
            }
        }
        None
    }

    /// Bellman-Ford Algorithm.
    /// Computes shortest paths from a source. 
    /// Returns an error if a negative weight cycle is detected.
    pub fn bellman_ford(&self, source: usize) -> SciResult<(HashMap<usize, f64>, HashMap<usize, usize>)> {
        let mut dist = HashMap::new();
        let mut prev = HashMap::new();
        let nodes = self.nodes();
        
        for &node in &nodes {
            dist.insert(node, f64::INFINITY);
        }
        dist.insert(source, 0.0);

        // Relax edges |V| - 1 times
        for _ in 0..(nodes.len().saturating_sub(1)) {
            let mut updated = false;
            for &u in &nodes {
                if let Some(neighbors) = self.adjacency_list.get(&u) {
                    let d_u = *dist.get(&u).unwrap();
                    if d_u != f64::INFINITY {
                        for &(v, weight) in neighbors {
                            if d_u + weight < *dist.get(&v).unwrap() {
                                dist.insert(v, d_u + weight);
                                prev.insert(v, u);
                                updated = true;
                            }
                        }
                    }
                }
            }
            if !updated { break; }
        }

        // Check for negative weight cycles
        for &u in &nodes {
            if let Some(neighbors) = self.adjacency_list.get(&u) {
                let d_u = *dist.get(&u).unwrap();
                if d_u != f64::INFINITY {
                    for &(v, weight) in neighbors {
                        if d_u + weight < *dist.get(&v).unwrap() - 1e-9 {
                            return Err(SciError::NonConvergent("Graph contains a negative weight cycle"));
                        }
                    }
                }
            }
        }

        Ok((dist, prev))
    }

    /// Prim's Algorithm for Minimum Spanning Tree.
    /// Returns the total weight and the edges of the MST.
    /// Assumes the graph is undirected.
    pub fn prim_mst(&self, start: usize) -> SciResult<(f64, Vec<(usize, usize, f64)>)> {
        if self.directed {
            return Err(SciError::InvalidParameter("Prim's algorithm requires an undirected graph"));
        }

        let mut mst_edges = Vec::new();
        let mut visited = HashSet::new();
        let mut total_weight = 0.0;

        visited.insert(start);
        
        let mut heap = BinaryHeap::new();
        if let Some(neighbors) = self.adjacency_list.get(&start) {
            for &(v, weight) in neighbors {
                heap.push(EdgeState { cost: weight, u: start, v });
            }
        }

        while let Some(EdgeState { cost, u, v }) = heap.pop() {
            if visited.contains(&v) {
                continue;
            }
            
            visited.insert(v);
            mst_edges.push((u, v, cost));
            total_weight += cost;

            if let Some(neighbors) = self.adjacency_list.get(&v) {
                for &(next_v, weight) in neighbors {
                    if !visited.contains(&next_v) {
                        heap.push(EdgeState { cost: weight, u: v, v: next_v });
                    }
                }
            }
        }

        Ok((total_weight, mst_edges))
    }

    /// PageRank Algorithm
    /// Calculates the centrality/importance of each node in the network.
    pub fn pagerank(&self, damping_factor: f64, tolerance: f64, max_iters: usize) -> HashMap<usize, f64> {
        let nodes = self.nodes();
        let n = nodes.len() as f64;
        let mut ranks = HashMap::new();
        
        if n == 0.0 {
            return ranks;
        }

        for &node in &nodes {
            ranks.insert(node, 1.0 / n);
        }

        let mut out_degree = HashMap::new();
        for &u in &nodes {
            let degree = self.adjacency_list.get(&u).map(|n| n.len()).unwrap_or(0);
            out_degree.insert(u, degree);
        }

        for _ in 0..max_iters {
            let mut new_ranks = HashMap::new();
            let mut dangling_sum = 0.0;

            for &u in &nodes {
                if out_degree[&u] == 0 {
                    dangling_sum += ranks[&u];
                }
            }

            let base_rank = (1.0 - damping_factor) / n + damping_factor * (dangling_sum / n);

            for &v in &nodes {
                new_ranks.insert(v, base_rank);
            }

            for &u in &nodes {
                if out_degree[&u] > 0 {
                    let rank_share = ranks[&u] / out_degree[&u] as f64;
                    if let Some(neighbors) = self.adjacency_list.get(&u) {
                        for &(v, _) in neighbors {
                            *new_ranks.get_mut(&v).unwrap() += damping_factor * rank_share;
                        }
                    }
                }
            }

            let mut diff = 0.0;
            for &u in &nodes {
                diff += (new_ranks[&u] - ranks[&u]).abs();
            }

            ranks = new_ranks;

            if diff < tolerance {
                break;
            }
        }

        ranks
    }
}
