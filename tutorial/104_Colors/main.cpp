#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writePLY.h>
#include <fstream>
#include <Eigen/Core>

#define TUTORIAL_SHARED_PATH "C:/Users/Christy-PC/source/repos/MeshDecimation-libigl/build/_deps/libigl_tutorial_data-src"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;
Eigen::VectorXd face_weights; // Per-face weights


void writePLY_with_vertex_colors(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& C)
{
    std::ofstream ply(filename);
    if (!ply.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }

    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << V.rows() << "\n";
    ply << "property float x\nproperty float y\nproperty float z\n";
    ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ply << "element face " << F.rows() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "end_header\n";

    // vertices + colors
    for (int i = 0; i < V.rows(); ++i) {
        int r = static_cast<int>(std::round(std::clamp(C(i, 0), 0.0, 1.0) * 255.0));
        int g = static_cast<int>(std::round(std::clamp(C(i, 1), 0.0, 1.0) * 255.0));
        int b = static_cast<int>(std::round(std::clamp(C(i, 2), 0.0, 1.0) * 255.0));
        ply << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " "
            << r << " " << g << " " << b << "\n";
    }

    // faces
    for (int i = 0; i < F.rows(); ++i) {
        ply << F.cols();
        for (int j = 0; j < F.cols(); ++j)
            ply << " " << F(i, j);
        ply << "\n";
    }

    ply.close();
    std::cout << "Wrote " << filename << " with vertex colors." << std::endl;
}


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

    if (weight_range > 1e-10)
    { // Avoid division by zero
        vertex_weights = (vertex_weights.array() - min_weight) / weight_range;
    }

    // Convert normalized weights to colors using a heatmap (blue to red)
    for (int v = 0; v < num_vertices; v++)
    {
        double w = vertex_weights(v);

        // Heatmap (Blue = no visibility, Yellow = Max visibility)
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
    std::string model_name = "apartment_floor_simplified";
  // Load a mesh in OBJ format
  std::string filename(TUTORIAL_SHARED_PATH "/models/apartment_floor_decimated/" + model_name + ".obj");

  igl::read_triangle_mesh(filename, V, F);

  // Read weights from file
  std::ifstream fin(std::string(std::getenv("USERPROFILE")) + "\\Desktop\\apartment_floor_weights.txt");
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

  const auto& key_down =
      [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)->bool
      {
          switch (key)
          {
          case 'V':
          case'v':// Press 'V' or 'v' to save the mesh
          {
              std::string filename(TUTORIAL_SHARED_PATH "/models/apartment_floor/obj/" + model_name + ".ply");
              writePLY_with_vertex_colors(filename, V, F, C);
              break;
          }
          default:
              return false; 
          }
          return true;
      };

  // Add per-vertex colors
  viewer.data().set_colors(C);

  viewer.callback_key_down = key_down;

  // Launch the viewer
  viewer.launch();
}
