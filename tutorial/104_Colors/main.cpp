#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/read_triangle_mesh.h>

#define TUTORIAL_SHARED_PATH "C:/Users/Christy-PC/source/repos/MeshDecimation-libigl/build/_deps/libigl_tutorial_data-src"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;
Eigen::VectorXd face_weights; // Per-face weights

// Function to convert per-face weights to per-vertex colors
Eigen::MatrixXd face_weights_to_vertex_colors(
    const Eigen::VectorXd& W,
    const Eigen::MatrixXi& F,
    int num_vertices)
{
    // Initialize vertex colors matrix (num_vertices x 3 for RGB)
    Eigen::MatrixXd vertex_colors = Eigen::MatrixXd::Zero(num_vertices, 3);
    Eigen::VectorXd vertex_weight_sum = Eigen::VectorXd::Zero(num_vertices);
    Eigen::VectorXi vertex_count = Eigen::VectorXi::Zero(num_vertices);

    // Accumulate weights for each vertex from adjacent faces
    for (int f = 0; f < F.rows(); f++) {
        for (int v = 0; v < 3; v++) { // Each face has 3 vertices
            int vertex_idx = F(f, v);
            vertex_weight_sum(vertex_idx) += W(f);
            vertex_count(vertex_idx)++;
        }
    }

    // Average the weights for each vertex
    Eigen::VectorXd vertex_weights(num_vertices);
    for (int v = 0; v < num_vertices; v++) {
        if (vertex_count(v) > 0) {
            vertex_weights(v) = vertex_weight_sum(v) / vertex_count(v);
        }
        else {
            vertex_weights(v) = 0.0;
        }
    }

    // Normalize weights to [0, 1] range
    double min_weight = vertex_weights.minCoeff();
    double max_weight = vertex_weights.maxCoeff();
    double weight_range = max_weight - min_weight;

    if (weight_range > 1e-10) { // Avoid division by zero
        vertex_weights = (vertex_weights.array() - min_weight) / weight_range;
    }

    // Convert normalized weights to colors using a heatmap (blue to red)
    for (int v = 0; v < num_vertices; v++) {
        double w = vertex_weights(v);

        //// Simple blue-to-red heatmap
        //vertex_colors(v, 0) = w;           // Red component
        //vertex_colors(v, 1) = 0.0;         // Green component
        //vertex_colors(v, 2) = 1.0 - w;     // Blue component

        // Alternative: more sophisticated heatmap
        
        if (w < 0.5) {
            vertex_colors(v, 0) = 0.0;                    // Red
            vertex_colors(v, 1) = 2.0 * w;                // Green
            vertex_colors(v, 2) = 1.0;                    // Blue
        } else {
            vertex_colors(v, 0) = 2.0 * (w - 0.5);        // Red
            vertex_colors(v, 1) = 1.0;                    // Green
            vertex_colors(v, 2) = 2.0 * (1.0 - w);        // Blue
        }
        
    }

    return vertex_colors;
}


int main(int argc, char *argv[])
{
  // Load a mesh in OBJ format
  std::string filename(TUTORIAL_SHARED_PATH "/models/san_miguel/san_miguel.obj");

  igl::read_triangle_mesh(filename, V, F);

  // Read weights from file
  std::ifstream fin(std::string(std::getenv("USERPROFILE")) + "\\Desktop\\san_miguel_weights.txt");
  if (!fin) {
      std::cerr << "Error: could not open weights file.\n";
      return -1; // exit if file can't be opened
  }

  Eigen::VectorXd W(F.rows());

  for (int i = 0; i < F.rows(); ++i)
  {
      fin >> W(i);
  }
  fin.close();

  // Set face weights for decimation
  igl::set_weights(W);

  //face_weights = Eigen::VectorXd::Random(F.rows()).cwiseAbs(); // Random weights [0,1]

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);

  // Convert per-face weights to per-vertex colors
  C = face_weights_to_vertex_colors(W, F, V.rows());

  //// Use the (normalized) vertex positions as colors
  //C =
  //  (V.rowwise()            - V.colwise().minCoeff()).array().rowwise()/
  //  (V.colwise().maxCoeff() - V.colwise().minCoeff()).array();

  viewer.data().set_colors(C); // Use per-vertex colors

  // Add per-vertex colors
  viewer.data().set_colors(C);

  // Launch the viewer
  viewer.launch();
}
