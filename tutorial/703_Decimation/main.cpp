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
#include <igl/is_edge_manifold.h>
#include <igl/split_nonmanifold.h>

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
    const Eigen::MatrixXi& F, bool removeDuplicates)
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

    std::string out_filename;

    // Build new filename
    if (removeDuplicates)
    {
        out_filename = desktop_path + filename + "_clean" + extension;
    }
    else
    {
        out_filename = desktop_path + filename + "_simplified" + extension;
    }

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

    // Print usage instructions
    cout << "Usage: ./703_Decimation_bin [filename.(off|obj|ply)]" << endl;
    cout << "  [space]  toggle animation." << endl;
    cout << "  'r'  reset." << endl;

    // Default mesh filename if none provided on command line
    string filename(TUTORIAL_SHARED_PATH "/models/level/obj/level_enclosed.obj");
    string removeDuplicates = "true";
    if (argc >= 2)
    {
        // Override default filename with user input
        filename = argv[1];
        removeDuplicates = argv[2];

    }

    //// Matrices to hold vertex positions and face indices (original and working copies)
    MatrixXd V, OV; // V = current vertices, OV = original vertices
    MatrixXi F, OF; // F = current faces, OF = original faces

    // Intermediate matrices for cleaning
    MatrixXi F_clean; // Cleaned faces after preprocessing
    MatrixXd V_clean; // Cleaned vertices after preprocessing

    // Intermediate matrices for manifold processing
    MatrixXd V_manifold; // Manifold vertices after preprocessing
    MatrixXi F_manifold; // Manifold faces after preprocessing

    VectorXi I, J; // Index mappings
    VectorXi SVI; // mapping from original to split vertices

    //// Load mesh from file (triangle mesh)
    read_triangle_mesh(filename, OV, OF);

    if (removeDuplicates == "true")
    {
        igl::remove_duplicate_vertices(OV, OF, 1e-7, V_clean, I, J, F_clean);
        igl::remove_unreferenced(V_clean, F_clean, OV, OF, I);

        OV = V_clean;
        OF = F_clean;

        Eigen::MatrixXd V_manifold;
        Eigen::MatrixXi F_manifold;
        Eigen::VectorXi SVI; // mapping from original to split vertices

        igl::split_nonmanifold(OV, OF, V_manifold, F_manifold, SVI);

        bool isEdgeManifold = igl::is_edge_manifold(OF);
        bool isVertexManifold = is_vertex_manifold(OV, OF);

        printf("Is the mesh edge-manifold? %s\n", isEdgeManifold ? "Yes" : "No");
        printf("Is the mesh vertex-manifold? %s\n", isVertexManifold ? "Yes" : "No");

        //saveSimplifiedMesh(filename, OV, OF, true);
    }


    //////////////// DEBUG ////////////////

    //// Print original vertices and faces
    //std::cout << "Original vertices (OV):\n" << OV << std::endl;
    //std::cout << "Original faces (OF):\n" << OF << std::endl;

    cout << "\nLoaded mesh " << filename << " with " << OV.rows() << " vertices and " << OF.rows() << " faces\n";

    //////////////// DEBUG ////////////////

    // Allocate a vector with as many entries as there are faces
    Eigen::VectorXd W(OF.rows());

    // Read weights from file
    std::ifstream fin("C:\\Users\\vortimex\\Desktop\\level_enclosed_weights.txt");
    if (!fin) {
        std::cerr << "Error: could not open weights file.\n";
        return -1; // exit if file can't be opened
    }

    for (int i = 0; i < OF.rows(); ++i)
    {
        fin >> W(i);
    }
    fin.close();

    cout << W.rows() << endl;
    cout << OF.rows() << endl;

    // Create an OpenGL viewer window
    igl::opengl::glfw::Viewer viewer;

    // Data structures for edges and mesh connectivity
    VectorXi EMAP; // Map edges to edge indices
    MatrixXi E, EF, EI; // Edges, edge-face adjacency, edge indices per face
    igl::min_heap< std::tuple<double, int, int> > Q; // Priority queue for edges (cost, edge id, unused)
    Eigen::VectorXi EQ; // Edge queue index helper

    MatrixXd C; // Candidate collapse points per edge (usually midpoint)
    int num_collapsed; // Counter of how many edges have been collapsed

    // Lambda function to reset mesh and initialize all data structures
    const auto& reset = [&]()
        {
            // Reset faces and vertices to original mesh
            F = OF;
            V = OV;

            // Compute edge data structures for the mesh
            edge_flaps(F, E, EMAP, EF, EI);

            // Resize collapse points array to match number of edges
            C.resize(E.rows(), V.cols());
            VectorXd costs(E.rows());
            // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
            // Q.clear();
            // Clear priority queue by reinitializing empty queue
            Q = {};

            // Initialize EQ helper vector (one entry per edge) to zeros
            EQ = Eigen::VectorXi::Zero(E.rows());
            {
                // Temporary vector to hold edge costs
                Eigen::VectorXd costs(E.rows());

                // Set face weights for decimation
                igl::set_weights(W);

                // Compute cost and collapse points for each edge in parallel
                igl::parallel_for(E.rows(), [&](const int e)
                    {
                        double cost = e; // Initialize cost (this will be overwritten)
                        RowVectorXd p(1, 3); // To store midpoint position

                        // Compute shortest edge length and midpoint, update cost and collapse point
                        shortest_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);

                        C.row(e) = p; // Store candidate collapse point for edge e
                        costs(e) = cost; // Store computed cost
                    }, 10000); // Parallelize over edges, with 10,000 threads per batch

                // Push all edges into the priority queue with their cost
                for (int e = 0; e < E.rows(); e++)
                {
                    Q.emplace(costs(e), e, 0);
                }
            }

            // Reset the collapsed edges count
            num_collapsed = 0;

            // Clear any previous mesh data from viewer and set new mesh
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.data().set_face_based(true); // Use face-based rendering
        };

    // Lambda function called before every draw frame in the viewer
    const auto& pre_draw = [&](igl::opengl::glfw::Viewer& viewer)->bool
        {
            // Only run if animation is toggled on and there are edges to collapse
            if (viewer.core().is_animating && !Q.empty())
            {
                // Toggle animation flag off after one iteration (to step collapse one batch)
                viewer.core().is_animating ^= 1;

                bool something_collapsed = false;

                // Decide how many edges to collapse in this frame: 60% of remaining edges
                const int max_iter = std::ceil(0.6 * Q.size());
                for (int j = 0; j < max_iter; j++)
                {
                    // Callbacks controlling whether collapse is allowed / post-processing
                    igl::decimate_pre_collapse_callback always_try; // Always try to collapse edges
                    igl::decimate_post_collapse_callback never_care;  // No special post-collapse handling
                    igl::decimate_trivial_callbacks(always_try, never_care);

                    // Explicit template instanciations expect std::function not raw pointer
                    // Only relevant if IGL_STATIC_LIBRARY is defined
                    // Collapse function callback that computes cost and midpoint for collaps
                    const std::function<void(int, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, double&, Eigen::Matrix<double, 1, -1, 1, 1, -1>&)> cp = shortest_edge_and_midpoint;

                    // Variables to receive collapse results (edge and faces involved)
                    int e, e1, e2, f1, f2;

                    // Try collapsing the edge with lowest cost from priority queue
                    if (!collapse_least_cost_edge(
                        cp, always_try, never_care,
                        V, F, E,
                        EMAP, EF, EI,
                        Q, EQ, C,
                        e, e1, e2, f1, f2))
                    {
                        // If collapse failed, exit loop early
                        break;
                    }

                    // Record that a collapse happened
                    something_collapsed = true;
                    num_collapsed++;
                }

                // If any edge was collapsed, update the viewer mesh
                if (something_collapsed)
                {
                    viewer.data().clear();
                    viewer.data().set_mesh(V, F);
                    viewer.data().set_face_based(true);
                }
            }

            // Returning false indicates no further viewer updates needed now
            return false;
        };

    // Keyboard input callback handler
    const auto& key_down =
        [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)->bool
        {
            switch (key)
            {
            case ' ':
                // Toggle animation on/off when space bar pressed
                viewer.core().is_animating ^= 1;
                break;
            case 'V':
            case'v':// Press 'V' or 'v' to save the mesh
            {
                saveSimplifiedMesh(filename, V, F, false);
                break;
            }
            break;
            case 'R':
            case 'r':
                // Reset the mesh and data structures when 'r' pressed
                reset();
                break;
            default:
                return false; // Other keys not handled here
            }
            return true;
        };

    // Initialize everything before launching viewer
    reset();

    // Set viewer background color to white
    viewer.core().background_color.setConstant(1);

    // Start animation by default
    viewer.core().is_animating = false;

    // Assign callbacks for key presses and pre-draw frame updates
    viewer.callback_key_down = key_down;
    viewer.callback_pre_draw = pre_draw;

    // Launch the interactive OpenGL viewer
    return viewer.launch();
}