# Mesh Decimation via Weighted Importance Mapping — libigl

**Undergraduate Thesis · Athens University of Economics and Business, Department of Informatics · 2025**

> **Authors:** C. Chrysafi · A. A. Manetas
>
> **Companion repository (NVIDIA OptiX — visibility analysis):** https://github.com/Heart-lex/MeshDecimation-OptiX

---

## Overview

This repository contains the **libigl-based mesh simplification** component of a two-part, visibility-driven mesh decimation pipeline. The pipeline reduces the polygon count of 3D scenes while selectively preserving perceptually important geometry — surfaces that are frequently seen by a virtual agent or camera.

Traditional mesh decimation treats every triangle as equally important. This produces efficient meshes but can destroy visually critical detail. Our approach addresses this by using **ray-traced visibility data** (computed in the companion OptiX repository) to assign per-face importance weights, which are then embedded into a modified least-cost edge-collapse algorithm. Rarely-seen geometry is decimated aggressively; frequently-viewed surfaces stay intact.

**Key result:** up to **85–90% polygon reduction** with near-imperceptible visual differences from the observer's point of view, verified with NVIDIA FLIP image comparison.

---

## Pipeline

```
                    glTF              weights.txt             OBJ
 ┌──────────────┐ ───────► ┌─────────────────┐ ──────────► ┌──────────────┐ ──────► ┌──────────────┐
 │ UE5 / Blender│          │  NVIDIA OptiX   │             │    libigl    │         │ UE5 / Blender│
 │ (scene setup)│          │ (ray-cast visi- │             │(weighted     │         │  (result)    │
 └──────────────┘          │  bility analysis│             │ simplification         └──────────────┘
                           └─────────────────┘             └──────────────┘
                              ↑ companion repo                  ↑ this repo
```

The pipeline is **offline** — it pre-processes exported scenes and produces optimized meshes that can be reimported into any compatible tool.

---

## How It Works

### Stage 1 — Visibility Analysis (OptiX, companion repo)

A virtual agent walks through the scene. At each step, rays are cast in all directions and every triangle intersection is recorded. Each hit accumulates a **perceptual weight** combining:

- **Distance weighting** — a logarithmic inverse-square falloff that emphasises nearby geometry:

$$w_d = \frac{1}{\log_{10}(1 + d^2)}$$

- **Foveated weighting** — models the drop in human visual acuity away from the fovea (centre of gaze):

| Region | Cone angle | Weight |
|--------|-----------|--------|
| Foveal | ~5° | 1.0 |
| Parafoveal | ~20° | 0.51 – 0.99 |
| Peripheral | ~45° | 0.1 – 0.5 |

The final per-hit importance is:

$$w_i = \log(1 + w_d \cdot w_f)$$

After all rays are traced, per-triangle weights are normalised to `[0, 1]` and exported as a plain-text file (one value per line, one line per triangle).

### Stage 2 — Weighted Simplification (this repo)

The libigl simplification extends the standard least-cost edge-collapse algorithm:

1. Compute all edges and adjacency (`igl::edge_flaps`).
2. Load the per-triangle weight table produced by the OptiX stage.
3. For each edge, compute a geometric cost and candidate collapse point (`shortest_edge_and_midpoint`).
4. **Multiply the cost by the local importance weight** — edges adjacent to high-importance faces become more expensive to collapse, preserving visible detail.
5. Push each edge into a min-heap priority queue ordered by cost.
6. Iteratively collapse the cheapest edge (`collapse_least_cost_edge`), respecting boundary, material, and structural constraints.

A custom `isStructurallyImportant()` guard additionally protects:
- Boundary and sharp edges
- Long structural edges
- Horizontal and vertical planes (floors and walls)

Material continuity is maintained after each collapse via an `update_materials` callback, and the final mesh is serialised with `writeOBJWithMaterialsAndUVs()` — preserving geometry, UV coordinates, normals, and MTL references.

---

## Results

Tests were run on a custom apartment scene containing **112,591 triangles**, built in Blender and Unreal Engine 5.

| Reduction | Triangles remaining | Visual difference (NVIDIA FLIP, agent POV) |
|-----------|--------------------|--------------------------------------------|
| 0% (original) | 112,591 | — reference |
| 50% | 56,261 | Negligible; only minor silhouette edges affected |
| 85% | 18,765 | Silhouette-only changes; occluded geometry heavily reduced |
| ~90% | ~11,000 | Geometry outside agent path removed; agent-visible surfaces intact |

NVIDIA FLIP image comparisons confirmed that differences, when viewed from positions along the agent's traversal path, are confined to mesh silhouettes and are imperceptible to the naked eye. Objects entirely occluded from the agent's perspective (e.g., the lower body of a camel behind a wall, toys in an unvisited room) were correctly identified and aggressively decimated.

---

## Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| CMake | ≥ 3.16 | |
| C++ compiler | C++17 | MSVC, GCC ≥ 9, or Clang ≥ 9 |
| libigl | 2.5.0 | Header-only; fetched automatically by CMake |
| Eigen3 | | Fetched automatically |
| OpenGL + GLFW + ImGui | | For the interactive viewer; fetched automatically |

> This repository runs **entirely on CPU**. No GPU is needed.
> The NVIDIA OptiX companion repository additionally requires a **CUDA-capable RTX GPU** and the OptiX 9.0 SDK.

---

## Building

```bash
git clone https://github.com/Heart-lex/MeshDecimation-libigl.git
cd MeshDecimation-libigl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

The compiled binary is placed in `build/bin/`.

---

## Usage

```
./MeshDecimation <input.obj> <weights.txt> [output.obj]
```

| Argument | Description |
|----------|-------------|
| `<input.obj>` | The mesh to simplify (must be manifold; exported from Blender or UE5 as OBJ) |
| `<weights.txt>` | Per-triangle importance map produced by the OptiX pipeline |
| `[output.obj]` | Optional output path (can also be saved interactively mid-run) |

The simplification runs iteratively without a fixed target. You can export the mesh at **any intermediate stage**, which makes it straightforward to generate a full LOD hierarchy from a single run.

---

## Input Requirements

For the pipeline to work correctly:

- **Manifold geometry only.** Each edge must be shared by exactly two faces. Non-manifold edges can be repaired in Blender using the *3D Printing Toolkit* add-on (Mesh → Clean Up → Make Manifold).
- **Single joined mesh.** All objects must be joined into one unified object in Blender before export (`Ctrl+J`). Separate objects can produce inconsistent importance distributions.
- **OBJ format.** Other formats (FBX, STL, PLY, OFF) were found to alter face indexing, breaking the per-triangle correspondence between the weight file and the mesh.

---

## File Formats at a Glance

| Stage | Format | Why |
|-------|--------|-----|
| Visibility analysis | `.gltf` | Encodes scene hierarchy, transforms, and materials in a GPU-friendly structure; native to both OptiX and UE5 |
| Simplification | `.obj` | Simple, explicit polygonal representation; direct per-vertex / per-face access; most flexible format supported by libigl |

---

## Limitations

| Limitation | Detail |
|-----------|--------|
| **Offline only** | A ~120k-triangle scene takes ~80 min to reach 90% reduction. The edge-collapse loop is serial; parallelisation is a natural future improvement. |
| **Static scenes** | Geometry and lighting must remain unchanged during visibility analysis. Moving objects, deformable meshes, and time-varying lighting are not supported. |
| **Visibility-only importance** | Weights are derived solely from ray hit frequency. Lighting contrast, texture saliency, and motion relevance are not yet incorporated. |
| **Single-layer occlusion** | Only the first ray intersection is recorded. Translucency and secondary reflections are not modelled. |
| **GPU requirement (OptiX stage)** | An NVIDIA RTX-capable GPU with CUDA support is required for the companion visibility analysis step. |

---

## Related Work

| Paper | Relevance |
|-------|-----------|
| **Xanthopoulos (2021)** — *Evaluation of Illumination-Based Mesh Simplification* | Direct predecessor. Lighting-driven per-primitive importance via multi-pass OptiX ray tracing fed into libigl. |
| **Liu et al. (2017)** — *Seam erasure and seam-aware decoupling of shape from mesh resolution* | Extended QSlim with texture-seam-aware decimation; informs how UV continuity is maintained. |
| **Lindstrom & Turk (2000)** — *Image-driven simplification* | Multi-view RMS image metric as edge collapse criterion; early work on removing invisible geometry. |
| **Garland & Heckbert (1997)** — *Surface Simplification Using Quadric Error Metrics* | Foundational edge-collapse algorithm underlying most modern decimation work, including libigl's. |

---

## Citation

If you use or build on this work, please cite:

```bibtex
@thesis{chrysafi_manetas_2025,
  author  = {Chrysafi, C. and Manetas, A. A.},
  title   = {Mesh Decimation / Simplification via Weighted Importance Mapping},
  school  = {Department of Informatics, Athens University of Economics and Business},
  year    = {2025},
  address = {Athens, Greece}
}
```

---

## License

This repository is a fork of [libigl](https://libigl.github.io/) v2.5.0.  
The libigl core is available under the [MPL2 license](LICENSE.MPL2).  
Copyleft modules are available under the [GPL license](LICENSE.GPL).  
Thesis-specific modifications are subject to the same MPL2 terms.
