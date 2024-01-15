use tri_mesh::{Mesh, VertexID, Vector3};
use std::collections::HashSet;

pub struct MeshCutting {
    mesh: Mesh,
}

impl MeshCutting {
    pub fn new(mesh: Mesh) -> Self {
        MeshCutting { mesh }
    }

    pub fn open_mesh_along_seam(&self, edge_path: Vec<tri_mesh::HalfEdgeID>) -> Mesh {

        // 1. Find all halfedges that are inside the affected faces
        let zigzag_cutline = self.collect_zigzag_cutline(edge_path.clone());

        // ! TODO: refactor this as we only need this information and not that which is provided by zigzag_cutline
        // 1.1 collect the vertices of the zigzag cutline
        let mut zigzag_cutline_vertices = Vec::new();
        let mut unique_vertices = HashSet::new();

        for h in zigzag_cutline.iter() {
            let (v0, v1) = self.mesh.edge_vertices(*h);

            if unique_vertices.insert(v0) {
                zigzag_cutline_vertices.push(v0);
            }
            if unique_vertices.insert(v1) {
                zigzag_cutline_vertices.push(v1);
            }
        }

        // 3. Get all vertices of the open mesh
        let vertex_coord = self.collect_vertex_position(edge_path.clone());

        // 4. Assign new vertex indices to the affected faces
        let mut face_id = Vec::new();
        for f in self.mesh.face_iter() {
            // 3.1 Get the vertices of the face
            let (v0, v1, v2) = self.mesh.face_vertices(f);

            let mut v0_u32 = *v0;
            let mut v1_u32 = *v1;
            let mut v2_u32 = *v2;

            let v0_in_zigzag = zigzag_cutline_vertices.contains(&v0);
            let v1_in_zigzag = zigzag_cutline_vertices.contains(&v1);
            let v2_in_zigzag = zigzag_cutline_vertices.contains(&v2);

            if v0_in_zigzag && v1_in_zigzag && v2_in_zigzag {
                // Process each vertex separately by getting the second index of the vertex_coord which will be the newly added vertex
                if let Some((index, _)) = vertex_coord.iter().enumerate()
                    .filter(|&(_, v)| *v == self.mesh.position(v0))
                    .nth(1) {
                        v0_u32 = index as u32;
                }


                if let Some((index, _)) = vertex_coord.iter().enumerate()
                    .filter(|&(_, v)| *v == self.mesh.position(v1))
                    .nth(1) {
                        v1_u32 = index as u32;
                }

                if let Some((index, _)) = vertex_coord.iter().enumerate()
                    .filter(|&(_, v)| *v == self.mesh.position(v2))
                    .nth(1) {
                        v2_u32 = index as u32;
                }
            }

            face_id.push(v0_u32);
            face_id.push(v1_u32);
            face_id.push(v2_u32);
        }

        // Finally: Assemble the mesh
        self.create_mesh(vertex_coord, face_id)
    }

    fn collect_vertex_position(&self, edge_path: Vec<tri_mesh::HalfEdgeID>) -> Vec<Vector3<f64>> {
        // 0.1 Add all old vertices
        let mut vertex_coord = Vec::new();
        for vertex_id in self.mesh.vertex_iter() {
            let vertex_position = self.mesh.vertex_position(vertex_id);
            vertex_coord.push(vertex_position);
        }

        // 0.2 Add the vertices of the seam from every second entry of edge_path
        // This ensures that the first and last vertex of the seam don't get added
        for i in (1..edge_path.len()).step_by(1) {
            let edge = edge_path[i];
            let (v0, _) = self.mesh.edge_vertices(edge);
            let v0_position = self.mesh.vertex_position(v0);
            vertex_coord.push(v0_position);
        }
        vertex_coord
    }

    fn get_cutline(&self, edge_path: Vec<tri_mesh::HalfEdgeID>) -> Vec<VertexID> {
        let mut cutline_vertices = Vec::new();
        let first_edge = edge_path[0];
        let (v0, _) = self.mesh.edge_vertices(first_edge);
        cutline_vertices.push(v0);

        for halfedge_id in edge_path.iter() {
            let (v0, v1) = self.mesh.edge_vertices(*halfedge_id);
            if cutline_vertices.last() != Some(&v0) {
                panic!("Error in the sorted edge path of the cut line.");
            }
            cutline_vertices.push(v1);
        }
        cutline_vertices
    }

    fn collect_zigzag_cutline(&self, edge_path: Vec<tri_mesh::HalfEdgeID>) -> Vec<tri_mesh::HalfEdgeID> {
        // 0.1 Get the vertices of the cut line
        let cutline_vertices = self.get_cutline(edge_path.clone());

        let mut zigzag_cutline = edge_path.clone();
        for i in 0..(edge_path.len() - 1) {
            let h = edge_path[i];
            let vertice_after = cutline_vertices[i + 2];
            let mut walker: tri_mesh::Walker<'_> = self.mesh.walker_from_halfedge(h);

            loop {
                let h_center = walker.as_twin().as_next().halfedge_id().unwrap();
                let (v1, _) = self.mesh.edge_vertices(h_center);
                if v1 == vertice_after {
                    break;
                }
                zigzag_cutline.push(h_center);
            }
        }
        zigzag_cutline
    }

    fn create_mesh(&self, vertices_id: Vec<Vector3<f64>>, faces_id: Vec<u32>) -> Mesh {
        Mesh::new(&three_d_asset::TriMesh {
            positions: three_d_asset::Positions::F64(vertices_id),
            indices: three_d_asset::Indices::U32(faces_id),
            ..Default::default()
        })
    }
}