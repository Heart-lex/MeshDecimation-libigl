#include <igl/circulation.h>
#include <igl/collapse_least_cost_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/decimate_trivial_callbacks.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <igl/unique_rows.h>

// ============================================================================
// MATERIAL HANDLING STRUCTURES AND FUNCTIONS
// ============================================================================

// Structure to hold material groups from OBJ
struct MaterialGroup {
    std::string materialName;
    int startFace;
    int endFace;
};

// Parse material assignments from original OBJ
std::vector<MaterialGroup> parseMaterialGroups(const std::string& filename) {
    std::vector<MaterialGroup> groups;
    std::ifstream file(filename);
    std::string line;
    std::string currentMaterial;
    int faceCount = 0;
    int materialStartFace = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "usemtl") {
            // Save previous material group
            if (!currentMaterial.empty()) {
                groups.push_back({ currentMaterial, materialStartFace, faceCount - 1 });
            }
            // Start new material group
            std::string matName;
            std::getline(iss, matName);
            // Trim leading whitespace
            size_t start = matName.find_first_not_of(" \t");
            currentMaterial = (start == std::string::npos) ? matName : matName.substr(start);
            materialStartFace = faceCount;
        }
        else if (prefix == "f") {
            faceCount++;
        }
    }

    // Add last material group
    if (!currentMaterial.empty()) {
        groups.push_back({ currentMaterial, materialStartFace, faceCount - 1 });
    }

    return groups;
}

// Extract MTL filename from OBJ
std::string getMTLFilename(const std::string& objFilename) {
    std::ifstream file(objFilename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "mtllib") {
            std::string mtlFile;
            std::getline(iss, mtlFile);
            // Trim leading whitespace
            size_t start = mtlFile.find_first_not_of(" \t");
            return (start == std::string::npos) ? mtlFile : mtlFile.substr(start);
        }
    }

    return "";
}

// Build per-face material mapping
std::vector<std::string> buildFaceMaterialMap(
    const std::vector<MaterialGroup>& groups,
    int totalFaces) {

    std::vector<std::string> faceMaterials(totalFaces, "default");

    for (const auto& group : groups) {
        for (int i = group.startFace; i <= group.endFace && i < totalFaces; i++) {
            faceMaterials[i] = group.materialName;
        }
    }

    return faceMaterials;
}

// Write OBJ with materials and texture coordinates
void writeOBJWithMaterialsAndUVs(
    const std::string& filename,
    const std::string& mtlFilename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& TC,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& CN,
    const Eigen::MatrixXi& FN,
    const std::vector<std::string>& faceMaterials) {

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write MTL reference
    if (!mtlFilename.empty()) {
        file << "mtllib " << mtlFilename << "\n\n";
    }

    // Write vertices
    for (int i = 0; i < V.rows(); i++) {
        file << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    file << "\n";

    // Write texture coordinates
    if (TC.rows() > 0) {
        for (int i = 0; i < TC.rows(); i++) {
            file << "vt " << TC(i, 0) << " " << TC(i, 1);
            if (TC.cols() > 2) {
                file << " " << TC(i, 2);
            }
            file << "\n";
        }
        file << "\n";
    }

    // Write normals
    if (CN.rows() > 0) {
        for (int i = 0; i < CN.rows(); i++) {
            file << "vn " << CN(i, 0) << " " << CN(i, 1) << " " << CN(i, 2) << "\n";
        }
        file << "\n";
    }

    // Write faces grouped by material
    std::string currentMaterial;
    bool hasTexCoords = (FT.rows() == F.rows() && TC.rows() > 0);
    bool hasNormals = (FN.rows() == F.rows() && CN.rows() > 0);

    for (int i = 0; i < F.rows(); i++) {
        // Change material if needed
        if (i < faceMaterials.size() && faceMaterials[i] != currentMaterial) {
            currentMaterial = faceMaterials[i];
            file << "usemtl " << currentMaterial << "\n";
        }

        // Write face with texture coordinates and normals
        file << "f";
        for (int j = 0; j < F.cols(); j++) {
            file << " " << (F(i, j) + 1);

            if (hasTexCoords || hasNormals) {
                file << "/";
                if (hasTexCoords) {
                    file << (FT(i, j) + 1);
                }
                if (hasNormals) {
                    file << "/" << (FN(i, j) + 1);
                }
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Successfully wrote OBJ with materials to: " << filename << std::endl;
}

// ============================================================================
// ORIGINAL FUNCTIONS (unchanged)
// ============================================================================

// Compute face normal
Eigen::Vector3d faceNormal(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int f)
{
    if (f < 0 || f >= F.rows()) return Eigen::Vector3d(0, 0, 1);

    int v0 = F(f, 0), v1 = F(f, 1), v2 = F(f, 2);
    if (v0 >= V.rows() || v1 >= V.rows() || v2 >= V.rows())
        return Eigen::Vector3d(0, 0, 1);

    Eigen::Vector3d p0 = V.row(v0).transpose();
    Eigen::Vector3d p1 = V.row(v1).transpose();
    Eigen::Vector3d p2 = V.row(v2).transpose();

    Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0);
    double norm = normal.norm();
    return (norm > 1e-10) ? normal / norm : Eigen::Vector3d(0, 0, 1);
}

// Check if faces are likely floors (horizontal) or walls (vertical)
bool isFloorOrWall(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int face_id)
{
    if (face_id < 0 || face_id >= F.rows()) return false;

    Eigen::Vector3d normal = faceNormal(V, F, face_id);

    // Floor: normal points mostly up (Y-axis)
    if (std::abs(normal.y()) > 0.8) return true;

    // Wall: normal is mostly horizontal (low Y component)
    if (std::abs(normal.y()) < 0.3) return true;

    return false;
}

// Check if edge is structurally important (boundary, sharp, or connects important faces)
bool isStructurallyImportant(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& E, const Eigen::MatrixXi& EF,
    const Eigen::VectorXd& W, int edge_id)
{
    if (edge_id < 0 || edge_id >= EF.rows()) return true;

    int f1 = EF(edge_id, 0);
    int f2 = EF(edge_id, 1);

    // Boundary edges (only one adjacent face) 
    if (f1 < 0 || f2 < 0) return true;

    // Only protect EXTREMELY low weight faces (near zero)
    if (f1 >= 0 && f1 < W.rows() && W(f1) < 0.01)
        if (isFloorOrWall(V, F, f1))
            return true;
    if (f2 >= 0 && f2 < W.rows() && W(f2) < 0.01)
        if (isFloorOrWall(V, F, f2))
            return true;

    // Sharp edges (very sharp angles only)
    if (f1 >= 0 && f1 < F.rows() && f2 >= 0 && f2 < F.rows()) {
        Eigen::Vector3d n1 = faceNormal(V, F, f1);
        Eigen::Vector3d n2 = faceNormal(V, F, f2);
        double dot = n1.dot(n2);

        // Only protect very sharp edges (> 60 degrees, dot < 0.5)
        if (dot < 0.5) return true;
    }

    // Very long edges
    if (edge_id < E.rows()) {
        int v1 = E(edge_id, 0);
        int v2 = E(edge_id, 1);
        if (v1 >= 0 && v1 < V.rows() && v2 >= 0 && v2 < V.rows()) {
            double edge_length = (V.row(v1) - V.row(v2)).norm();
            if (edge_length > 5.0) return true;
        }
    }

    return false;
}

// MODIFIED: Save function now uses material-aware writer
void saveSimplifiedMesh(const std::string& original_path,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& CN,
    const Eigen::MatrixXi& FN,
    const Eigen::MatrixXd& TC,
    const Eigen::MatrixXi& FT,
    const std::vector<std::string>& faceMaterials,
    const std::string& mtlFilename)
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

    std::string out_filename = desktop_path + filename + "_simplified" + extension;

    // Copy MTL file if it exists
    if (!mtlFilename.empty()) {
        std::filesystem::path inputPath(original_path);
        std::filesystem::path mtlInputPath = inputPath.parent_path() / mtlFilename;
        std::filesystem::path mtlOutputPath = desktop_path + mtlFilename;

        try {
            if (std::filesystem::exists(mtlInputPath)) {
                std::filesystem::copy_file(mtlInputPath, mtlOutputPath,
                    std::filesystem::copy_options::overwrite_existing);
                std::cout << "Copied MTL file: " << mtlFilename << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Could not copy MTL file: " << e.what() << std::endl;
        }
    }

    // Save mesh with materials
    writeOBJWithMaterialsAndUVs(out_filename, mtlFilename, V, F, TC, FT, CN, FN, faceMaterials);

    std::cout << "Simplified mesh saved as " << out_filename << std::endl;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[])
{
    using namespace std;
    using namespace Eigen;
    using namespace igl;

    // Print usage instructions
    cout << "Usage: ./703_Decimation_bin [filename.obj] [decimation factor (integer, >0)]" << endl;
    cout << "  [space]  toggle animation." << endl;
    cout << "  'r'  reset." << endl;

    // Default mesh filename if none provided on command line
    string filename(TUTORIAL_SHARED_PATH "/models/apartment_floor/obj/apartment_floor.obj");
    unsigned int decimation_factor = 5; // Default: 20% of starting face count
    if (argc >= 2)
    {
        // Override default filename with user input
        filename = argv[1];
        decimation_factor = atoi(argv[2]);
    }

    if (decimation_factor < 2)
    {
        std::cerr << "Decimation factor has to be greater than 1!" << std::endl;
        return -1;
    }

    //// Matrices to hold vertex positions and face indices (original and working copies)
    MatrixXd V, OV, TC, CN; // V = current vertices, OV = original vertices, TC = texture coords, CN = vertex normals
    MatrixXi F, OF, FT, FN; // F = current faces, OF = original faces, FT = face texture indices, FN = face normal indices

    //// Load mesh from file (triangle mesh)
    if (!igl::readOBJ(filename, OV, TC, CN, OF, FT, FN)) {
        std::cerr << "ERROR: Could not read OBJ: " << filename << std::endl;
    }

    cout << "\nLoaded mesh " << filename << " with " << OV.rows() << " vertices and " << OF.rows() << " faces\n";

    // ========================================================================
    // MATERIAL PARSING - NEW CODE
    // ========================================================================

    // Parse material information from the original OBJ file
    std::vector<MaterialGroup> materialGroups = parseMaterialGroups(filename);
    std::vector<std::string> originalFaceMaterials = buildFaceMaterialMap(materialGroups, OF.rows());
    std::string mtlFilename = getMTLFilename(filename);

    std::cout << "Found " << materialGroups.size() << " material groups" << std::endl;
    if (!mtlFilename.empty()) {
        std::cout << "MTL file: " << mtlFilename << std::endl;
    }

    // Track current face materials (will be updated during decimation)
    std::vector<std::string> currentFaceMaterials = originalFaceMaterials;

    // ========================================================================

    // Allocate a vector with as many entries as there are faces
    Eigen::VectorXd W(OF.rows());

    // Read weights from file
    std::ifstream fin(std::string(std::getenv("USERPROFILE")) + "\\Desktop\\apartment_floor_weights.txt");
    if (!fin) {
        std::cerr << "Error: could not open weights file.\n";
        return -1; // exit if file can't be opened
    }

    for (int i = 0; i < OF.rows(); ++i)
    {
        fin >> W(i);
    }
    fin.close();

    std::cout << "Successfully read weights file." << std::endl;

    // Create an OpenGL viewer window
    igl::opengl::glfw::Viewer viewer;

    // Data structures for edges and mesh connectivity
    VectorXi EMAP; // Map edges to edge indices
    MatrixXi E, EF, EI; // Edges, edge-face adjacency, edge indices per face
    igl::min_heap< std::tuple<double, int, int> > Q; // Priority queue for edges (cost, edge id, unused)
    Eigen::VectorXi EQ; // Edge queue index helper

    MatrixXd C; // Candidate collapse points per edge (usually midpoint)
    int num_collapsed; // Counter of how many edges have been collapsed

    unsigned int face_target = 0; // Decimation stop condition
    unsigned int face_counter = -1; // Decimation face counter

    // Lambda function to reset mesh and initialize all data structures
    const auto& reset = [&]()
        {
            // Reset faces and vertices to original mesh
            F = OF;
            V = OV;

            // Reset material tracking
            currentFaceMaterials = originalFaceMaterials;

            // Compute edge data structures for the mesh
            edge_flaps(F, E, EMAP, EF, EI);

            // Resize collapse points array to match number of edges
            C.resize(E.rows(), V.cols());
            VectorXd costs(E.rows());
            // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
            // Q.clear();
            // Clear priority queue by reinitializing empty queue
            Q = {};

            unsigned int important_edges = 0;

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

                        if (isStructurallyImportant(V, F, E, EF, W, e))
                        {
                            // Set very high cost to prevent collapse
                            costs(e) = std::numeric_limits<double>::max();

                            // Set safe collapse point (midpoint)
                            if (e < E.rows() && E(e, 0) < V.rows() && E(e, 1) < V.rows()) {
                                C.row(e) = (V.row(E(e, 0)) + V.row(E(e, 1))) * 0.5;
                            }
                            else {
                                C.row(e) = Eigen::RowVectorXd::Zero(3);
                            }

                            ++important_edges;

                            return;
                        }

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

            face_counter = F.rows();
            face_target = F.rows() / decimation_factor; // Simplify to decimation_factor% of the initial faces (default: 20%)

            std::cout << "Number of important edges: " << important_edges << std::endl;

            std::cout << "Face target: " << face_target << std::endl;

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
            // Only run if animation is toggled on, there are edges to collapse
            // and we have not reached the face threshold yet
            if (viewer.core().is_animating && !Q.empty() && face_counter > face_target)
            {
                // Toggle animation flag off after one iteration (to step collapse one batch)
                //viewer.core().is_animating ^= 1;

                bool something_collapsed = false;

                // Decide how many edges to collapse in this frame: 100% of remaining edges
                const int max_iter = std::ceil(1.0 * Q.size());
                for (int j = 0; j < max_iter; j++)
                {
                    int e, e1, e2, f1, f2;

                    // Callbacks controlling whether collapse is allowed / post-processing
                    igl::decimate_pre_collapse_callback always_try; // Always try to collapse edges

                    // ================================================================
                    // MATERIAL TRACKING - NEW CODE
                    // Define a post-collapse callback to update material tracking
                    // ================================================================
                    igl::decimate_post_collapse_callback update_materials =
                        [&](const Eigen::MatrixXd& V,
                            const Eigen::MatrixXi& F,
                            const Eigen::MatrixXi& E,
                            const Eigen::VectorXi& EMAP,
                            const Eigen::MatrixXi& EF,
                            const Eigen::MatrixXi& EI,
                            const igl::min_heap<std::tuple<double, int, int>>& Q,
                            const Eigen::VectorXi& EQ,
                            const Eigen::MatrixXd& C,
                            const int e,
                            const int e1,
                            const int e2,
                            const int f1,
                            const int f2,
                            const bool collapsed) -> void
                        {
                            if (!collapsed) return;

                            // When faces f1 and f2 are removed, we need to update our material tracking
                            // The faces after f1 and f2 shift down by 2
                            if (f1 >= 0 && f2 >= 0 && f1 < currentFaceMaterials.size() && f2 < currentFaceMaterials.size()) {
                                int minF = std::min(f1, f2);
                                int maxF = std::max(f1, f2);

                                // Remove the two collapsed faces
                                currentFaceMaterials.erase(currentFaceMaterials.begin() + maxF);
                                currentFaceMaterials.erase(currentFaceMaterials.begin() + minF);
                            }
                        };

                    // Assign both callbacks
                    igl::decimate_trivial_callbacks(always_try, update_materials);

                    const std::function<void(int, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, double&, Eigen::Matrix<double, 1, -1, 1, 1, -1>&)> cp = shortest_edge_and_midpoint;

                    if (!collapse_least_cost_edge(
                        cp, always_try, update_materials,
                        V, F, E, EMAP, EF, EI, Q, EQ, C,
                        e, e1, e2, f1, f2))
                    {
                        break;
                    }

                    something_collapsed = true;
                    num_collapsed++;
                    face_counter -= 2;

                    // DEBUG
                    std::cout << "Edges: " << E.rows() << std::endl;
                    std::cout << "Collapsed Edges: " << num_collapsed << " | Q: " << Q.size() << std::endl;
                    std::cout << "Current Faces: " << face_counter << " | Face Decimation Target: " << face_target << std::endl;
                }

                if (something_collapsed)
                {
                    viewer.data().clear();
                    viewer.data().set_mesh(V, F);
                    viewer.data().set_face_based(true);
                }
            }
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
            case 'v':// Press 'V' or 'v' to save the mesh
            {
                // MODIFIED: Pass material information to save function
                saveSimplifiedMesh(filename, V, F, CN, FN, TC, FT, currentFaceMaterials, mtlFilename);
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
