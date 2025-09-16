#include <igl/circulation.h>
#include <igl/collapse_least_cost_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/decimate_trivial_callbacks.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <igl/remove_unreferenced.h>
#include <igl/writeOBJ.h>
#include <filesystem>
#include <string>
#include <igl/remove_duplicate_vertices.h>
#include <igl/unique_rows.h>
//#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
#include <igl/split_nonmanifold.h>
#include <igl/resolve_duplicated_faces.h>

bool is_vertex_manifold(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    // Map from vertex to incident faces
    std::vector<std::vector<int>> vertex_faces(V.rows());
    for (int f = 0; f < F.rows(); ++f)
    {
        for (int j = 0; j < 3; ++j)
        {
            vertex_faces[F(f, j)].push_back(f);
        }
    }

    // Check each vertex
    for (int v = 0; v < V.rows(); ++v)
    {
        const auto& faces = vertex_faces[v];
        if (faces.empty()) continue; // isolated vertex, ignore

        // Build adjacency graph of faces around vertex v
        std::map<int, std::set<int>> face_adjacency;
        for (int i = 0; i < faces.size(); ++i)
        {
            int f1 = faces[i];
            for (int j = i + 1; j < faces.size(); ++j)
            {
                int f2 = faces[j];
                // Count shared edge
                int shared = 0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        if (F(f1, a) == F(f2, b)) shared++;
                if (shared == 2) // they share an edge
                {
                    face_adjacency[f1].insert(f2);
                    face_adjacency[f2].insert(f1);
                }
            }
        }

        // BFS/DFS to check connectivity
        std::set<int> visited;
        std::queue<int> q;
        q.push(faces[0]);
        visited.insert(faces[0]);
        while (!q.empty())
        {
            int cf = q.front(); q.pop();
            for (int nf : face_adjacency[cf])
            {
                if (visited.count(nf) == 0)
                {
                    visited.insert(nf);
                    q.push(nf);
                }
            }
        }

        if (visited.size() != faces.size())
        {
            // Faces around this vertex are disconnected -> non-manifold
            return false;
        }
    }

    return true;
}


void saveSimplifiedMesh(const std::string& original_path,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    // Desktop path
    std::string desktop_path = std::string(std::getenv("USERPROFILE")) + "\\Desktop\\";

    // Extract filename without path
    size_t last_slash = original_path.find_last_of("\\/");
    std::string filename_with_ext = (last_slash == std::string::npos)
        ? original_path
        : original_path.substr(last_slash + 1);

    // Find extension
    size_t last_dot = filename_with_ext.find_last_of(".");
    std::string filename = (last_dot == std::string::npos)
        ? filename_with_ext
        : filename_with_ext.substr(0, last_dot);
    std::string extension = (last_dot == std::string::npos)
        ? ".obj"
        : filename_with_ext.substr(last_dot);

    // Build new filename
    std::string out_filename = desktop_path + filename + "_simplified" + extension;

    // Save mesh
    if (!igl::writeOBJ(out_filename, V, F))
        std::cerr << "Failed to save simplified mesh." << std::endl;
    else
        std::cout << "Simplified mesh saved as " << out_filename << std::endl;
}


int main(int argc, char* argv[])
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;
  cout<<"Usage: ./703_Decimation_bin [filename.(off|obj|ply)]"<<endl;
  cout<<"  [space]  toggle animation."<<endl;
  cout<<"  'r'  reset."<<endl;
  // Load a closed manifold mesh
  string filename(TUTORIAL_SHARED_PATH "/fertility.off");
  if(argc>=2)
  {
    filename = argv[1];
  }
  MatrixXd V,OV;
  MatrixXi F,OF;
  read_triangle_mesh(filename,OV,OF);

  igl::opengl::glfw::Viewer viewer;

  // Prepare array-based edge data structures and priority queue
  VectorXi EMAP;
  MatrixXi E,EF,EI;
  igl::min_heap< std::tuple<double,int,int> > Q;
  Eigen::VectorXi EQ;
  // If an edge were collapsed, we'd collapse it to these points:
  MatrixXd C;
  int num_collapsed;

  // Function to reset original mesh and data structures
  const auto & reset = [&]()
  {
    F = OF;
    V = OV;
    edge_flaps(F,E,EMAP,EF,EI);
    C.resize(E.rows(),V.cols());
    VectorXd costs(E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    Q = {};
    EQ = Eigen::VectorXi::Zero(E.rows());
    {
      Eigen::VectorXd costs(E.rows());
      igl::parallel_for(E.rows(),[&](const int e)
      {
        double cost = e;
        RowVectorXd p(1,3);
        shortest_edge_and_midpoint(e,V,F,E,EMAP,EF,EI,cost,p);
        C.row(e) = p;
        costs(e) = cost;
      },10000);
      for(int e = 0;e<E.rows();e++)
      {
        Q.emplace(costs(e),e,0);
      }
    }

    num_collapsed = 0;
    viewer.data().clear();
    viewer.data().set_mesh(V,F);
    viewer.data().set_face_based(true);
  };

  const auto &pre_draw = [&](igl::opengl::glfw::Viewer & viewer)->bool
  {
    // If animating then collapse 10% of edges
    if(viewer.core().is_animating && !Q.empty())
    {
      bool something_collapsed = false;
      // collapse edge
      const int max_iter = std::ceil(0.01*Q.size());
      for(int j = 0;j<max_iter;j++)
      {
        igl::decimate_pre_collapse_callback always_try;
        igl::decimate_post_collapse_callback never_care;
        igl::decimate_trivial_callbacks(always_try,never_care);
        // Explicit template instanciations expect std::function not raw pointer
        // Only relevant if IGL_STATIC_LIBRARY is defined
        const std::function<void (int, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, double&, Eigen::Matrix<double, 1, -1, 1, 1, -1>&)> cp = shortest_edge_and_midpoint;
        int e,e1,e2,f1,f2;
        if(!collapse_least_cost_edge(
              cp,always_try,never_care,
              V,F,E,
              EMAP,EF,EI,
              Q,EQ,C,
              e,e1,e2,f1,f2))
        {
          break;
        }
        something_collapsed = true;
        num_collapsed++;
      }

      if(something_collapsed)
      {
        viewer.data().clear();
        viewer.data().set_mesh(V,F);
        viewer.data().set_face_based(true);
      }
    }
    return false;
  };

  const auto &key_down =
    [&](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
  {
    switch(key)
    {
      case ' ':
        viewer.core().is_animating ^= 1;
        break;
            case 'S':
            case's':// Press 'S' to save
            {
                saveSimplifiedMesh(filename, V, F);
                break;
            }
            break;
      case 'R':
      case 'r':
        reset();
        break;
      default:
        return false;
    }
    return true;
  };

  reset();
  viewer.core().background_color.setConstant(1);
  viewer.core().is_animating = true;
  viewer.callback_key_down = key_down;
  viewer.callback_pre_draw = pre_draw;
  return viewer.launch();
}
