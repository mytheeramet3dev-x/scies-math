# Graph Theory and Network Analysis (`graph`)

The `graph` module provides adjacency-list graph data structures, pathfinding algorithms, minimum spanning trees, and link-analysis centrality.

---

## 1. Supported Algorithms

| Algorithm | Method | Directness | Time Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `dijkstra` | Priority Queue (Min-Heap) | Directed / Undirected | $O((|V| + |E|) \log |V|)$ | Non-negative edge weights |
| `a_star` | Informed Heuristic Search | Directed / Undirected | $O(|E|)$ best case | Requires admissible heuristic $h(u) \le d(u, \text{goal})$ |
| `bellman_ford` | Edge Relaxation | Directed | $O(|V| \cdot |E|)$ | Detects negative weight cycles |
| `prim_mst` | Greedy Spanning Tree | Undirected | $O(|E| \log |V|)$ | Computes Minimum Spanning Tree |
| `pagerank` | Power Iteration | Directed | $O(k \cdot |E|)$ | Stationary Markov distribution |

---

## 2. PageRank Centrality (`pagerank`)

Computes the stationary probability distribution of a random surfer with damping factor $d \in (0, 1)$ (typically $0.85$):

$$\mathbf{PR}(u) = \frac{1 - d}{|V|} + d \sum_{v \in \mathcal{M}(u)} \frac{\mathbf{PR}(v)}{L(v)}$$

where $\mathcal{M}(u)$ is the set of pages linking to $u$, and $L(v)$ is the out-degree of page $v$.

---

## 3. Code Example

```rust
use scies_math_th::graph::Graph;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut g = Graph::new(false); // Undirected graph

    // Add edges: (u, v, weight)
    g.add_edge(0, 1, 4.0);
    g.add_edge(0, 2, 2.0);
    g.add_edge(2, 1, 1.0);
    g.add_edge(1, 3, 5.0);
    g.add_edge(2, 3, 8.0);

    // Dijkstra Shortest Path from 0 to 3
    let (path, cost) = g.dijkstra(0, 3)?;
    println!("Shortest path: {:?} with total cost: {:.1}", path, cost); // [0, 2, 1, 3], cost 8.0

    // Minimum Spanning Tree via Prim's algorithm
    let mst = g.prim_mst()?;
    let mst_weight: f64 = mst.iter().map(|(_, _, w)| w).sum();
    println!("MST edges: {:?}, Total Weight: {:.1}", mst, mst_weight);

    Ok(())
}
```
