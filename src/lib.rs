//! # Mesh Cartography Library Interface
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-11
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

// Import necessary modules and types
use wasm_bindgen::prelude::*;
use std::env;
use std::path::PathBuf;
use tri_mesh::{Mesh, VertexID, Vector3};
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use nalgebra::Vector2;
use web_sys::console;

pub mod mesh_definition;
use crate::mesh_definition::TexCoord;

pub mod io;
mod monotile_border;

pub mod geodesic_distance {
    pub mod cached_geodesic_distance_helper;
    pub mod dijkstra_distance_helper;
    pub mod gaussian_curvature;
    pub mod gaussian_cut_line_helper;
}

mod mesh_cutting;

mod mesh_metric {
    pub mod angle_distortion_helper;
    pub mod face_distortion_helper;
    pub mod length_distortion_helper;
}

mod surface_parameterization {
    pub mod boundary_matrix;
    pub mod laplacian_matrix;
    pub mod harmonic_parameterization_helper;
    pub mod tessellation_helper;
}

#[derive(Debug, Clone, PartialEq)]
struct VertexPosition(f64, f64, f64);

impl Eq for VertexPosition {}

impl Hash for VertexPosition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = (self.0 * 1e6 as f64) as i64;
        let y = (self.1 * 1e6 as f64) as i64;
        let z = (self.2 * 1e6 as f64) as i64;
        x.hash(state);
        y.hash(state);
        z.hash(state);
    }
}

fn get_mesh_cartography_lib_dir() -> PathBuf {
    // TODO: this is only a temporary solution
    #[cfg(target_arch = "wasm32")]
    {
        // Return a default path when running in WebAssembly
        PathBuf::from("/Users/jan-piotraschke/git_repos/MeshCartographyLib-rs")
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Try to get the environment variable in a non-WebAssembly environment
        PathBuf::from(env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set"))
    }
}

fn create_mesh(vertices_id: Vec<Vector3<f64>>, faces_id: Vec<u32>) -> Mesh {
    Mesh::new(&three_d_asset::TriMesh {
        positions: three_d_asset::Positions::F64(vertices_id),
        indices: three_d_asset::Indices::U32(faces_id),
        ..Default::default()
    })
}

pub fn create_mesh_from_grouped_vertices(grouped_vertices: Vec<Vec<Vector3<f64>>>) -> Mesh {
    // Flatten the Vec<Vec<Vector3<f64>>> to Vec<Vector3<f64>>
    let combined_vertices: Vec<Vector3<f64>> = grouped_vertices.into_iter().flatten().collect();

    // Remove duplicate vertices
    let mut unique_vertices = Vec::new();
    let mut index_map = HashMap::new();
    for (original_index, v) in combined_vertices.iter().enumerate() {
        let new_index = unique_vertices.iter().position(|&x| x == *v);
        match new_index {
            Some(index) => {
                // Vertex is a duplicate, map to existing index
                index_map.insert(original_index, index);
            },
            None => {
                // Vertex is unique, add to unique_vertices and map to its new index
                unique_vertices.push(*v);
                index_map.insert(original_index, unique_vertices.len() - 1);
            }
        }
    }

    // Create the face indices
    let mut face_indices = Vec::new();
    for (original_index, _) in combined_vertices.iter().enumerate().step_by(3) {
        if original_index + 2 < combined_vertices.len() {
            if let (Some(&a), Some(&b), Some(&c)) = (index_map.get(&original_index), index_map.get(&(original_index + 1)), index_map.get(&(original_index + 2))) {
                face_indices.extend_from_slice(&[a as u32, b as u32, c as u32]);
            }
        }
    }

    // Create the mesh
    create_mesh(unique_vertices, face_indices)
}

#[wasm_bindgen]
pub fn init_js(file_path_str: String) -> String {
    // Clone the file path string before passing it to the function
    let file_path_clone = file_path_str.clone();

    match io::load_mesh_from_js_obj(file_path_str) {
        Ok(mesh) => {
            console::log_1(&format!("Mesh loaded successfully").into());
            let mut processor = MeshProcessor::from_mesh(mesh);
            console::log_1(&format!("MeshProcessor initialized").into());

            // Use the cloned file path here
            let uv_mesh: Mesh = processor.create_uv_surface(&file_path_clone);
            console::log_1(&format!("UV surface created").into());

            let uv_mesh_js = io::convert_uv_mesh_to_string(&uv_mesh, &processor.mesh_tex_coords)
                .expect("Failed to save mesh to file");

            // // Get the uv mesh path
            // let uv_mesh_path = processor.mesh_uv_path.clone();
            // let uv_mesh_path = uv_mesh_path.into_os_string().into_string().unwrap();

            // // Open OBJ path and return file content as a string
            // let uv_mesh_data = fs::read_to_string(uv_mesh_path).expect("Unable to read file");

            // let tessellation_mesh = processor.create_tessellation_mesh(&mut uv_mesh);

            uv_mesh_js
        }
        Err(e) => {
            // Error loading the mesh
            console::log_1(&format!("Error loading mesh: {}", e).into());
            format!("Error: {}", e)
        }
    }
}

pub struct MeshProcessor {
    boundary_vertices: Vec<VertexID>,
    mesh_tex_coords: mesh_definition::MeshTexCoords,
    mesh_cartography_lib_dir: PathBuf,
    pub mesh_uv_path: PathBuf,
    surface_closed: Mesh,
    pub vertice_3_d: Vec<Vector3<f64>>,
    pub vertice_uv: Vec<Vector3<f64>>,
    pub border: HashMap<usize, Vec<TexCoord>>,
    pub twin_border_map: HashMap<usize, usize>,
}

impl MeshProcessor {
    pub fn from_mesh(surface_closed: Mesh)  -> Self {
        // Initialize the struct
        let processor = MeshProcessor {
            boundary_vertices: Vec::new(),
            mesh_tex_coords: mesh_definition::MeshTexCoords::new(&surface_closed),
            mesh_cartography_lib_dir: get_mesh_cartography_lib_dir(),
            mesh_uv_path: PathBuf::new(),
            surface_closed,
            vertice_3_d: Vec::new(),
            vertice_uv: Vec::new(),
            border: HashMap::new(),
            twin_border_map: HashMap::new(),
        };
        processor
    }

    // Function to create UV surface
    pub fn create_uv_surface(&mut self, mesh_path: &str) -> Mesh {
        let mesh_path = PathBuf::from(mesh_path);
        let mesh_file_name = mesh_path.file_stem().unwrap().to_str().unwrap().to_string();
        let save_path = self.mesh_cartography_lib_dir.join(format!("{}_open.obj", mesh_file_name));
        let save_path_uv = self.mesh_cartography_lib_dir.join(format!("{}_uv.obj", mesh_file_name));
        self.mesh_uv_path = save_path_uv.clone();

        let cutline_helper = crate::geodesic_distance::gaussian_cut_line_helper::MeshAnalysis::new(self.surface_closed.clone());
        let edge_path: Vec<tri_mesh::HalfEdgeID> = cutline_helper.get_gaussian_cutline();

        // Open up the mesh along the cutline
        let cut_mesh_helper = mesh_cutting::MeshCutting::new(self.surface_closed.clone());
        let surface_mesh = cut_mesh_helper.open_mesh_along_seam(edge_path);

        // Save the mesh
        #[cfg(not(target_arch = "wasm32"))] {
            io::save_mesh_as_obj(&surface_mesh, save_path).expect("Failed to save mesh to file");
        }

        let (boundary_vertices, mesh_tex_coords) = parameterize_mesh(&surface_mesh);

        // Set the fields of the struct
        self.boundary_vertices = boundary_vertices;
        self.mesh_tex_coords = mesh_tex_coords;

        for i in 0..surface_mesh.no_vertices() {
            let vertex_id = surface_mesh.vertex_iter().nth(i).unwrap();
            let position = surface_mesh.vertex_position(vertex_id);
            self.vertice_3_d.push(Vector3::new(position.x, position.y, position.z));

            let tex_coord = self.mesh_tex_coords.get_tex_coord(vertex_id).unwrap();
            self.vertice_uv.push(Vector3::new(tex_coord.0, tex_coord.1, 0.0));
        }

        #[cfg(not(target_arch = "wasm32"))] {
            io::save_uv_mesh_as_obj(&surface_mesh, &self.mesh_tex_coords, save_path_uv.clone())
                .expect("Failed to save mesh to file");

            // Load the mesh and the UV mesh
            let surface_mesh = io::load_mesh_from_obj(mesh_path.clone()).unwrap();
            let uv_mesh = io::load_mesh_from_obj(save_path_uv.clone()).unwrap();

            // Compute the angle distortion
            let angle_distortion_helper = mesh_metric::angle_distortion_helper::AngleDistortionHelper::new(&surface_mesh, &uv_mesh);
            let angle_distortion = angle_distortion_helper.compute_angle_distortion();
            log::info!("Angle distortion: {}", angle_distortion);

            // Compute the face distortion
            let face_distortion_helper = mesh_metric::face_distortion_helper::FaceDistortionHelper::new(&surface_mesh, &uv_mesh);
            let face_distortion = face_distortion_helper.compute_face_distortion();
            log::info!("Face distortion: {}", face_distortion);

            // Compute the length distortion
            let length_distortion_helper = mesh_metric::length_distortion_helper::LengthDistortionHelper::new(&surface_mesh, &uv_mesh);
            let length_distortion = length_distortion_helper.compute_length_distortion();
            log::info!("Length distortion: {}", length_distortion);
            uv_mesh
        }
        #[cfg(target_arch = "wasm32")]
        {
            surface_mesh
        }
    }

    // Create the Kachelmuster with Heesch numbers
    pub fn create_tessellation_mesh(&mut self, uv_mesh_centre: &mut Mesh) -> Mesh {
        // ! Temp: load the uv_mesh
        let (border_v_map, border_map) = monotile_border::get_sub_borders(&self.boundary_vertices, &self.mesh_tex_coords);
        let mesh_file_name = self.mesh_uv_path.file_stem().unwrap().to_str().unwrap().to_string();
        let save_path_uv2 = self.mesh_cartography_lib_dir.join(format!("{}_tessellation.obj", mesh_file_name));

        let mut grouped_face_vertices: Vec<Vec<Vector3<f64>>> = Vec::new();
        crate::surface_parameterization::tessellation_helper::collect_face_vertices(&uv_mesh_centre, &mut grouped_face_vertices);

        let mut tessellation = crate::surface_parameterization::tessellation_helper::Tessellation::new(border_v_map.clone(), border_map.clone());
        let size = border_map.len();
        for i in 0..size {
            let docking_side: usize = i;
            let next_index = (i + 1) % size;

            // Convert Vec<TexCoord> to Vec<Vector2<f64>> for border_map[&next_index]
            let border1: Vec<Vector2<f64>> = border_map[&next_index]
                .iter()
                .map(|coord| Vector2::new(coord.0, coord.1))
                .collect();

            // Convert Vec<TexCoord> to Vec<Vector2<f64>> for border_map[&i]
            let border2: Vec<Vector2<f64>> = border_map[&i]
                .iter()
                .map(|coord| Vector2::new(coord.0, coord.1))
                .collect();

            let rotation_angle = tessellation.calculate_angle(&border1, &border2);
            // ! Temp: load the uv_mesh
            let mut uv_mesh = io::load_mesh_from_obj(self.mesh_uv_path.clone()).unwrap();
            tessellation.rotate_and_shift_mesh(&mut uv_mesh, rotation_angle, docking_side);
            log::info!("docking_side: {}", docking_side);

            // Get the face vertices coordinates
            crate::surface_parameterization::tessellation_helper::collect_face_vertices(&uv_mesh, &mut grouped_face_vertices);
        }

        // Add the meshes together
        let tessellation_mesh = create_mesh_from_grouped_vertices(grouped_face_vertices);

        let (border, twin_border_map) = tessellation.get_border();
        self.border = border.clone();
        self.twin_border_map = twin_border_map.clone();

        // Save the mesh
        for vertex_id in tessellation_mesh.vertex_iter() {
            self.mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(tessellation_mesh.position(vertex_id).x, tessellation_mesh.position(vertex_id).y));
        }
        io::save_uv_mesh_as_obj(&tessellation_mesh, &self.mesh_tex_coords, save_path_uv2.clone())
            .expect("Failed to save mesh to file");

        tessellation_mesh
    }

    pub fn get_border(&self) -> (&HashMap<usize, Vec<TexCoord>>, &HashMap<usize, usize>) {
        (&self.border, &self.twin_border_map)
    }
}

fn parameterize_mesh(surface_mesh: &Mesh) -> (Vec<VertexID>, mesh_definition::MeshTexCoords) {
    let (boundary_edges, length) = get_boundary_edges(surface_mesh);

    let edge_list = boundary_edges.iter().cloned().collect::<Vec<_>>();  // Collect edges in a Vec to maintain order
    let (boundary_vertices, _) = get_boundary_vertices(&edge_list, &surface_mesh);

    let mut mesh_tex_coords = init_mesh_tex_coords(surface_mesh, &boundary_vertices, length);
    surface_parameterization::harmonic_parameterization_helper::harmonic_parameterization(surface_mesh, &mut mesh_tex_coords, true);  // Parameterize the mesh

    (boundary_vertices, mesh_tex_coords)
}

fn init_mesh_tex_coords(surface_mesh: &Mesh, boundary_vertices: &[VertexID], length: f64) -> mesh_definition::MeshTexCoords {
    let corner_count = 4;
    let side_length = length / corner_count as f64;
    let tolerance = 1e-4;
    let mut mesh_tex_coords = mesh_definition::MeshTexCoords::new(surface_mesh);

    for vertex_id in surface_mesh.vertex_iter() {
        mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(0.0, 0.0));  // Initialize to the origin
    }

    let tex_coords = monotile_border::distribute_vertices_around_square(boundary_vertices, side_length, tolerance, length);

    for (&vertex_id, tex_coord) in boundary_vertices.iter().zip(tex_coords.iter()) {
        mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(tex_coord.0, tex_coord.1));
    }

    mesh_tex_coords
}

fn get_boundary_edges(surface_mesh: &Mesh) -> (Vec<(VertexID, VertexID)>, f64) {
    let mut boundary_edges = Vec::new();
    let mut length = 0.0;

    for edge in surface_mesh.edge_iter() {
        let (v0, v1) = surface_mesh.edge_vertices(edge);
        if surface_mesh.is_vertex_on_boundary(v0) && surface_mesh.is_vertex_on_boundary(v1) {
            boundary_edges.push((v0, v1));
            length += surface_mesh.edge_length(edge);
        }
    }

    (boundary_edges, length)
}

fn get_boundary_vertices(edge_list: &[(VertexID, VertexID)], surface_mesh: &Mesh) -> (Vec<VertexID>, Vec<VertexID>) {
    if edge_list.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut boundary_vertices = Vec::new();
    let mut current_vertex = edge_list[0].0;
    boundary_vertices.push(current_vertex);

    while boundary_vertices.len() <= edge_list.len() {
        let mut found = false;
        for &(v0, v1) in edge_list {
            if (v0 == current_vertex && !boundary_vertices.contains(&v1)) ||
               (v1 == current_vertex && !boundary_vertices.contains(&v0)) {
                current_vertex = if v0 == current_vertex { v1 } else { v0 };
                boundary_vertices.push(current_vertex);
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
    }

    let unique_vertex_ids = sort_boundary_vertices(&mut boundary_vertices, &surface_mesh);

    (boundary_vertices, unique_vertex_ids)
}

fn sort_boundary_vertices(boundary_vertices: &mut Vec<VertexID>, surface_mesh: &Mesh) -> Vec<VertexID> {
    let mut position_map = HashMap::new();
    let mut unique_vertex_ids = Vec::new();

    for vertex_id in &*boundary_vertices {
        let position = surface_mesh.vertex_position(*vertex_id);
        let vertex_position = VertexPosition(position.x, position.y, position.z);

        match position_map.get(&vertex_position) {
            Some(&existing_vertex_id) if existing_vertex_id != *vertex_id => {
                unique_vertex_ids.retain(|&id| id != existing_vertex_id);
            },
            None => {
                position_map.insert(vertex_position, *vertex_id);
                unique_vertex_ids.push(*vertex_id);
            },
            _ => {}
        }
    }

    if let Some(first_unique_vertex_id) = unique_vertex_ids.first() {
        if let Some(index) = boundary_vertices.iter().position(|&id| id == *first_unique_vertex_id) {
            boundary_vertices.rotate_left(index);
        }
    }

    boundary_vertices.reverse();
    // rotate boundary_vertices by 1 to the right so that the first vertex id is unique_vertex_ids.first()
    boundary_vertices.rotate_right(1);

    unique_vertex_ids
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tri_mesh::vec3;

    fn count_mesh_degree(surface_mesh: &Mesh) -> HashMap<VertexID, usize> {
        // Iterate over the connected faces
        let connected_faces = Mesh::connected_components(&surface_mesh); // Vec<HashSet<FaceID>>
        let mut vertex_degree = HashMap::new();

        for face_id in connected_faces[0].iter() {
            let face = surface_mesh.face_vertices(*face_id);

            // Destructure the tuple and increment count for each VertexID
            let (v1, v2, v3) = face;
            *vertex_degree.entry(v1).or_insert(0) += 1;
            *vertex_degree.entry(v2).or_insert(0) += 1;
            *vertex_degree.entry(v3).or_insert(0) += 1;
        }

        vertex_degree
    }

    fn count_open_mesh_degree(surface_mesh: &Mesh, boundary_vertices: &Vec<VertexID>) -> HashMap<VertexID, usize> {
        let mut vertex_degree = count_mesh_degree(&surface_mesh);

        // Add +1 for each boundary vertex
        for vertex_id in boundary_vertices.iter() {
            *vertex_degree.entry(*vertex_id).or_insert(0) += 1;
        }

        vertex_degree
    }

    #[test]
    fn test_dep_mesh_library() {
        // Combine the two vectors into one
        let combined_vertices = vec![
            vec3(0.0, 2.0, 1.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        ];

        let mesh = Mesh::new(&three_d_asset::TriMesh {
            positions: three_d_asset::Positions::F64(combined_vertices),
            ..Default::default()
        });

        assert_eq!(mesh.no_vertices(), 3);
        assert_eq!(mesh.no_faces(), 1);

        // Save the mesh to file
        let mesh_cartography_lib_dir = get_mesh_cartography_lib_dir();
        let save_path = mesh_cartography_lib_dir.join("test_triangle.obj");
        io::save_mesh_as_obj(&mesh, save_path).expect("Failed to save mesh to file");
    }

    #[test]
    fn test_create_mesh() {
        let grouped_verts = vec![
            vec![vec3(0.0, 2.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)],
            vec![vec3(0.0, 1.0, 2.0), vec3(0.0, 2.0, 1.0), vec3(2.0, 0.0, 1.0)],
        ];

        let mesh = create_mesh_from_grouped_vertices(grouped_verts);
        assert_eq!(mesh.no_vertices(), 5);
        assert_eq!(mesh.no_faces(), 2);

        // Save the mesh to file
        let mesh_cartography_lib_dir = get_mesh_cartography_lib_dir();
        let save_path = mesh_cartography_lib_dir.join("test_mesh.obj");
        io::save_mesh_as_obj(&mesh, save_path).expect("Failed to save mesh");
    }

    #[test]
    fn test_mesh_with_non_overlapping_vertices() {
        let grouped_verts = vec![
            vec![vec3(1.0, 2.0, 3.0), vec3(3.0, 2.0, 1.0), vec3(2.0, 3.0, 1.0)],
            vec![vec3(4.0, 5.0, 6.0), vec3(6.0, 5.0, 4.0), vec3(5.0, 4.0, 6.0)],
        ];

        let mesh = create_mesh_from_grouped_vertices(grouped_verts);
        assert_eq!(mesh.no_vertices(), 6);
        assert_eq!(mesh.no_faces(), 2);
    }

    #[test]
    fn test_get_cutline_ends() {
        let surface_mesh = io::load_test_mesh();
        let (boundary_edges, _) = get_boundary_edges(&surface_mesh);
        let edge_list = boundary_edges.iter().cloned().collect::<Vec<_>>();
        let (_, unique_vertex_ids) = get_boundary_vertices(&edge_list, &surface_mesh);

        // Convert unique_vertex_ids to a Vec of integers
        let mut usize_values = Vec::new();

        for vertex_id in &unique_vertex_ids {
            let index_as_u32: u32 = **vertex_id;
            let index_as_usize: usize = index_as_u32 as usize;
            usize_values.push(index_as_usize);
        }

        assert_eq!(usize_values[0], 4466);
        assert_eq!(usize_values[1], 1897);
    }

    #[test]
    fn test_neighbors_based_on_L_matrix() {
        let surface_mesh = io::load_test_mesh();
        let (boundary_edges, _) = get_boundary_edges(&surface_mesh);
        let edge_list = boundary_edges.iter().cloned().collect::<Vec<_>>();
        let (boundary_vertices, _) = get_boundary_vertices(&edge_list, &surface_mesh);

        let vertex_degree = count_open_mesh_degree(&surface_mesh, &boundary_vertices);
        let l_mtx = surface_parameterization::laplacian_matrix::build_laplace_matrix(&surface_mesh, true);

        for vertex_id in surface_mesh.vertex_iter() {
            let index_as_u32: u32 = *vertex_id;
            let index_as_usize: usize = index_as_u32 as usize;
            let expected = vertex_degree.get(&vertex_id).unwrap();

            // -1 because the diagonal entry is not counted as it is the vertex itself
            assert_eq!(l_mtx.row(index_as_usize).values().len() - 1, *expected);
        }
    }

    #[test]
    fn test_find_boundary_edges() {
        let surface_mesh = io::load_test_mesh();
        let (boundary_edges, length) = get_boundary_edges(&surface_mesh);

        assert!((length - 42.3117).abs() <= 0.001);
        assert!(length > 0.0);
        assert_eq!(boundary_edges.len(), 112);

        for &(v0, v1) in &boundary_edges {
            // println!("v0: {:?}, v1: {:?}", v0, v1);
            assert!(surface_mesh.is_vertex_on_boundary(v0));
            assert!(surface_mesh.is_vertex_on_boundary(v1));
        }
    }

    #[test]
    fn test_parameterize_mesh() {
        let surface_mesh = io::load_test_mesh();
        let (boundary_edges, _) = get_boundary_edges(&surface_mesh);
        let edge_list = boundary_edges.iter().cloned().collect::<Vec<_>>();
        let (boundary_vertices, _) = get_boundary_vertices(&edge_list, &surface_mesh);

        assert_eq!(boundary_vertices.len(), 112);

        for vertex_id in boundary_vertices {
            // println!("vertex_id: {:?}", vertex_id);
            assert!(surface_mesh.is_vertex_on_boundary(vertex_id));
        }
    }

    #[test]
    fn test_assign_vertices_to_boundary() {
        let surface_mesh = io::load_test_mesh();
        let (boundary_edges, length) = get_boundary_edges(&surface_mesh);
        let edge_list = boundary_edges.iter().cloned().collect::<Vec<_>>();
        let (boundary_vertices, _) = get_boundary_vertices(&edge_list, &surface_mesh);

        let corner_count = 4;
        let side_length = length / corner_count as f64;
        let tolerance = 1e-4;

        let mut mesh_tex_coords = mesh_definition::MeshTexCoords::new(&surface_mesh);

        for vertex_id in surface_mesh.vertex_iter() {
            mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(0.0, 0.0)); // Initialize to the origin
        }

        let tex_coords = monotile_border::distribute_vertices_around_square(&boundary_vertices, side_length, tolerance, length);
        for (&vertex_id, tex_coord) in boundary_vertices.iter().zip(tex_coords.iter()) {
            mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(tex_coord.0, tex_coord.1));
        }

        let vertex_id = surface_mesh.vertex_iter().nth(4697).unwrap();
        let tex_coord = mesh_tex_coords.get_tex_coord(vertex_id).unwrap();
        assert_eq!(tex_coord.0, 1.0);
        assert_eq!(tex_coord.1, 0.0);

        let vertex_id = surface_mesh.vertex_iter().nth(4690).unwrap();
        let tex_coord = mesh_tex_coords.get_tex_coord(vertex_id).unwrap();
        assert_eq!(tex_coord.0, 0.75);
        assert_eq!(tex_coord.1, 0.0);

        let vertex_id = surface_mesh.vertex_iter().nth(3099).unwrap();
        let tex_coord = mesh_tex_coords.get_tex_coord(vertex_id).unwrap();
        assert_eq!(tex_coord.0, 0.0);
        assert_eq!(tex_coord.1, 1.0);
    }

    #[test]
    fn test_boundary_matrix_B_creation() {
        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = create_mocked_mesh_tex_coords();
        let b_mtx = surface_parameterization::boundary_matrix::set_boundary_constraints(&surface_mesh, &mut mesh_tex_coords);

        let mut num_boundary_vertices = 0;
        for i in 0..b_mtx.nrows() {
            let row_data: Vec<f64> = b_mtx.row(i).iter().cloned().collect();
            if surface_mesh.is_vertex_on_boundary(surface_mesh.vertex_iter().nth(i).unwrap()) {
                num_boundary_vertices += 1;
            } else {
                assert_eq!(row_data, vec![0.0, 0.0]);
            }
        }
        assert_eq!(num_boundary_vertices, 112);
    }

    #[test]
    fn test_harmonic_parameterization() {
        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = create_mocked_mesh_tex_coords();
        let _b_mtx = surface_parameterization::boundary_matrix::set_boundary_constraints(&surface_mesh, &mut mesh_tex_coords);
        let _l_mtx = surface_parameterization::laplacian_matrix::build_laplace_matrix(&surface_mesh, true);

        // println!("L: {:?}", L);

        // for vertex_id in surface_mesh.vertex_iter() {
        //     // convert vertex_id to usize
        //     let index_as_u32: u32 = *vertex_id;
        //     let index_as_usize: usize = index_as_u32 as usize;
        //     let row_data: Vec<f64> = B.row(index_as_usize).iter().cloned().collect();

        //     if surface_mesh.is_vertex_on_boundary(vertex_id) {
        //         println!("");
        //         println!("L.row({:?}): {:?}", vertex_id, L.row(index_as_usize));
        //         println!("L_sparse.row({:?}): {:?}", vertex_id, L_sparse.row(index_as_usize));
        //         // println!("B.row({:?}): {:?}", vertex_id, row_data);
        //         println!("");
        //     } else {
        //         // println!("L.row({:?}): {:?}", vertex_id, L.row(index_as_usize).values());
        //     }
        // }
    }

    #[test]
    #[allow(unused_mut)]
    #[allow(unused_variables)]
    fn test_using_mocked_data() {
        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = mesh_definition::MeshTexCoords::new(&surface_mesh);

        // Load B matrix
        let file_path = "test/data/B.csv";
        let b_dense_mtx = io::load_csv_to_dmatrix(file_path).expect("Failed to load matrix");

        // switch the first with the second column of B_dense
        let mut b_dense_mtx_switched = b_dense_mtx.clone();
        for i in 0..b_dense_mtx.nrows() {
            b_dense_mtx_switched[(i, 0)] = b_dense_mtx[(i, 1)];
            b_dense_mtx_switched[(i, 1)] = b_dense_mtx_switched[(i, 0)];
        }

        // Load L matrix
        let file_path = "test/data/L_sparse.csv";
        let l_sparse_mtx = io::load_sparse_csv_data_to_csr_matrix(file_path).expect("Failed to load matrix");

        // Load is_constrained vector
        let file_path = "test/data/is_constrained.csv";
        let is_constrained = io::load_csv_to_bool_vec(file_path).expect("Failed to load matrix");

        // // Solve the linear equation system
        // let result = surface_parameterization::harmonic_parameterization_helper::solve_using_qr_decomposition(&l_sparse_mtx, &b_dense_mtx_switched, is_constrained);

        // // Assign the result to the mesh
        // match result {
        //     Ok(X) => {
        //         for (vertex_id, row) in surface_mesh.vertex_iter().zip(X.row_iter()) {
        //             let tex_coord = TexCoord(row[0], row[1]);
        //             // println!("tex_coord: {:?} {:?}", row[0], row[1]);
        //             mesh_tex_coords.set_tex_coord(vertex_id, tex_coord);
        //         }
        //     }
        //     Err(e) => {
        //         println!("An error occurred: {}", e);
        //     }
        // }

        // let mesh_cartography_lib_dir = get_mesh_cartography_lib_dir();
        // let save_path_uv = mesh_cartography_lib_dir.join("ellipsoid_x4_uv.obj");
        // io::save_uv_mesh_as_obj(&surface_mesh, &mesh_tex_coords, save_path_uv)
        //     .expect("Failed to save mesh to file");
    }

    fn create_mocked_mesh_tex_coords() -> mesh_definition::MeshTexCoords {
        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = mesh_definition::MeshTexCoords::new(&surface_mesh);

        for vertex_id in surface_mesh.vertex_iter() {
            mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(0.0, 0.0)); // Initialize to the origin
        }

        // Insert mocked data
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4466).unwrap(), TexCoord(0.0, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4670).unwrap(), TexCoord(0.03571428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4671).unwrap(), TexCoord(0.07142857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4672).unwrap(), TexCoord(0.10714285714285715, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4673).unwrap(), TexCoord(0.14285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4674).unwrap(), TexCoord(0.17857142857142855, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4675).unwrap(), TexCoord(0.2142857142857143, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4676).unwrap(), TexCoord(0.25, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4677).unwrap(), TexCoord(0.2857142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4678).unwrap(), TexCoord(0.3214285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4679).unwrap(), TexCoord(0.3571428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4680).unwrap(), TexCoord(0.39285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4681).unwrap(), TexCoord(0.4285714285714286, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4682).unwrap(), TexCoord(0.46428571428571425, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4683).unwrap(), TexCoord(0.5, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4684).unwrap(), TexCoord(0.5357142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4685).unwrap(), TexCoord(0.5714285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4686).unwrap(), TexCoord(0.6071428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4687).unwrap(), TexCoord(0.6428571428571428, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4688).unwrap(), TexCoord(0.6785714285714286, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4689).unwrap(), TexCoord(0.7142857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4690).unwrap(), TexCoord(0.75, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4691).unwrap(), TexCoord(0.7857142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4692).unwrap(), TexCoord(0.8214285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4693).unwrap(), TexCoord(0.8571428571428572, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4694).unwrap(), TexCoord(0.8928571428571429, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4695).unwrap(), TexCoord(0.9285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4696).unwrap(), TexCoord(0.9642857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4697).unwrap(), TexCoord(1.0, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4698).unwrap(), TexCoord(1.0, 0.035714285714285664));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4699).unwrap(), TexCoord(1.0, 0.07142857142857133));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4700).unwrap(), TexCoord(1.0, 0.10714285714285715));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4701).unwrap(), TexCoord(1.0, 0.14285714285714282));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4702).unwrap(), TexCoord(1.0, 0.17857142857142846));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4703).unwrap(), TexCoord(1.0, 0.2142857142857143));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4704).unwrap(), TexCoord(1.0, 0.24999999999999994));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4705).unwrap(), TexCoord(1.0, 0.28571428571428564));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4706).unwrap(), TexCoord(1.0, 0.32142857142857145));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4707).unwrap(), TexCoord(1.0, 0.3571428571428571));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4708).unwrap(), TexCoord(1.0, 0.3928571428571428));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4709).unwrap(), TexCoord(1.0, 0.42857142857142844));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4710).unwrap(), TexCoord(1.0, 0.46428571428571425));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4711).unwrap(), TexCoord(1.0, 0.4999999999999999));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4712).unwrap(), TexCoord(1.0, 0.5357142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4713).unwrap(), TexCoord(1.0, 0.5714285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4714).unwrap(), TexCoord(1.0, 0.6071428571428571));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4715).unwrap(), TexCoord(1.0, 0.6428571428571427));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4716).unwrap(), TexCoord(1.0, 0.6785714285714284));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4717).unwrap(), TexCoord(1.0, 0.7142857142857144));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4718).unwrap(), TexCoord(1.0, 0.75));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4719).unwrap(), TexCoord(1.0, 0.7857142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4720).unwrap(), TexCoord(1.0, 0.8214285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4721).unwrap(), TexCoord(1.0, 0.857142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4722).unwrap(), TexCoord(1.0, 0.8928571428571427));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4723).unwrap(), TexCoord(1.0, 0.9285714285714284));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4724).unwrap(), TexCoord(1.0, 0.9642857142857143));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1897).unwrap(), TexCoord(1.0, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3800).unwrap(), TexCoord(0.9642857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1857).unwrap(), TexCoord(0.9285714285714287, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1858).unwrap(), TexCoord(0.892857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1859).unwrap(), TexCoord(0.8571428571428573, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1860).unwrap(), TexCoord(0.8214285714285714, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1861).unwrap(), TexCoord(0.7857142857142857, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1862).unwrap(), TexCoord(0.75, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1863).unwrap(), TexCoord(0.7142857142857144, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1908).unwrap(), TexCoord(0.6785714285714287, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1909).unwrap(), TexCoord(0.642857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1910).unwrap(), TexCoord(0.6071428571428574, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1867).unwrap(), TexCoord(0.5714285714285714, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1834).unwrap(), TexCoord(0.5357142857142857, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1801).unwrap(), TexCoord(0.5000000000000001, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1802).unwrap(), TexCoord(0.4642857142857144, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1803).unwrap(), TexCoord(0.42857142857142877, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1915).unwrap(), TexCoord(0.3928571428571431, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1917).unwrap(), TexCoord(0.3571428571428571, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1918).unwrap(), TexCoord(0.32142857142857145, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2069).unwrap(), TexCoord(0.2857142857142858, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2071).unwrap(), TexCoord(0.2500000000000001, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2178).unwrap(), TexCoord(0.21428571428571447, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2123).unwrap(), TexCoord(0.1785714285714288, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2239).unwrap(), TexCoord(0.14285714285714315, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2291).unwrap(), TexCoord(0.10714285714285715, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2540).unwrap(), TexCoord(0.0714285714285715, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2713).unwrap(), TexCoord(0.03571428571428583, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3099).unwrap(), TexCoord(0.0, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3189).unwrap(), TexCoord(0.0, 0.9642857142857146));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3245).unwrap(), TexCoord(0.0, 0.9285714285714287));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3458).unwrap(), TexCoord(0.0, 0.8928571428571433));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3524).unwrap(), TexCoord(0.0, 0.8571428571428573));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3522).unwrap(), TexCoord(0.0, 0.8214285714285721));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3521).unwrap(), TexCoord(0.0, 0.785714285714286));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3553).unwrap(), TexCoord(0.0, 0.75));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3585).unwrap(), TexCoord(0.0, 0.7142857142857147));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(61).unwrap(), TexCoord(0.0, 0.6785714285714287));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(59).unwrap(), TexCoord(0.0, 0.6428571428571433));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(56).unwrap(), TexCoord(0.0, 0.6071428571428574));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(57).unwrap(), TexCoord(0.0, 0.5714285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(52).unwrap(), TexCoord(0.0, 0.535714285714286));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(101).unwrap(), TexCoord(0.0, 0.5000000000000001));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(100).unwrap(), TexCoord(0.0, 0.46428571428571475));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(99).unwrap(), TexCoord(0.0, 0.42857142857142877));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(47).unwrap(), TexCoord(0.0, 0.39285714285714346));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(45).unwrap(), TexCoord(0.0, 0.35714285714285743));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(43).unwrap(), TexCoord(0.0, 0.32142857142857145));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(41).unwrap(), TexCoord(0.0, 0.28571428571428614));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(39).unwrap(), TexCoord(0.0, 0.2500000000000001));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(37).unwrap(), TexCoord(0.0, 0.2142857142857148));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(92).unwrap(), TexCoord(0.0, 0.1785714285714288));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(91).unwrap(), TexCoord(0.0, 0.1428571428571435));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(90).unwrap(), TexCoord(0.0, 0.10714285714285748));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4336).unwrap(), TexCoord(0.0, 0.0714285714285715));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4467).unwrap(), TexCoord(0.0, 0.03571428571428616));

        mesh_tex_coords
    }
}
