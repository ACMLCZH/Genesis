import numpy as np
import trimesh


def mesh_from_centerline(
    verts: np.ndarray,
    radii: np.ndarray,
    radial_segs=16,
    cap_segs=8,
    endcaps=True,
    is_loop=False,
    smooth_joints=False,
    joint_segs=4
) -> trimesh.Trimesh:
    """
    Build a tube mesh with rounded ends around a polyline (rod centerline).
    This implementation is inspired by robust methods used in tools like Polyscope,
    creating clean mitered or rounded joins without resampling the centerline.

    Parameters
    ----------
    verts : (N,3) ndarray
        Sequence of 3D points along the rod centerline.
    radii : (N,) ndarray
        Radii at each vertex.
    radial_segs : int
        Number of segments around the tube's circumference.
    cap_segs : int
        Number of segments for the hemispherical end caps (from base to pole).
    endcaps : bool
        If True, close the ends with hemispherical caps. Ignored if is_loop is True.
    is_loop : bool
        If True, connect the ends to form a closed loop (toroid-like).
    smooth_joints : bool
        If True, creates smooth, spherical joins at vertices instead of sharp mitered joins.
    joint_segs : int
        Number of segments to use for each smoothed joint sphere.

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    verts = np.asarray(verts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    N = len(verts)

    if N < 2: raise ValueError("Need at least 2 vertices for a rod")
    if is_loop and N < 3: raise ValueError("Need at least 3 vertices for a loop")
    if verts.shape[0] != radii.shape[0]: raise ValueError("verts and radii must have the same length")

    tangents = []
    if is_loop:
        for i in range(N):
            tangents.append(verts[(i + 1) % N] - verts[i])
    else:
        for i in range(N - 1):
            tangents.append(verts[i + 1] - verts[i])
    tangents = np.array([t / np.linalg.norm(t) if np.linalg.norm(t) > 1e-9 else t for t in tangents])

    # Use parallel transport to create a twist-minimizing frame
    basis_list = []
    prev_normal = None
    for i in range(len(tangents)):
        tangent_norm = tangents[i]
        if i == 0:
            helper = np.array([0, 0, 1]) if abs(tangent_norm[2]) < 0.9 else np.array([0, 1, 0])
            normal = np.cross(tangent_norm, helper)
            if np.linalg.norm(normal) > 1e-9: normal /= np.linalg.norm(normal)
        else:
            prev_tangent = tangents[i - 1]
            v = prev_tangent + tangent_norm
            v_dot_v = np.dot(v, v)
            if v_dot_v > 1e-8:
                reflection_vec = 2 * np.dot(prev_normal, v) / v_dot_v
                normal = prev_normal - v * reflection_vec
            else:
                axis = np.cross(prev_tangent, prev_normal)
                normal = np.cos(np.pi) * prev_normal + np.sin(np.pi) * np.cross(axis, prev_normal)
        
        binormal = np.cross(tangent_norm, normal)
        basis_list.append((normal, binormal))
        prev_normal = normal

    V_list, F_list = [], []
    UV_list = []
    rings_in = []
    rings_out = []

    for i in range(len(tangents)):
        p_start = verts[i]
        p_end = verts[(i + 1) % N]
        r_start, r_end = radii[i], radii[(i + 1) % N]
        normal, binormal = basis_list[i]

        ring_start_indices, ring_end_indices = [], []
        base_idx = len(V_list)
        for j in range(radial_segs):
            theta = 2 * np.pi * j / radial_segs
            offset = np.cos(theta) * normal + np.sin(theta) * binormal
            V_list.append(p_start + r_start * offset)
            V_list.append(p_end + r_end * offset)
            ring_start_indices.append(base_idx + 2*j)
            ring_end_indices.append(base_idx + 2*j + 1)
            
            u = j / (radial_segs // 2)
            u = 2 - u if u > 1 else u
            # v_start = i / max(1, len(tangents) - 1) if not is_loop else i / len(tangents)
            # v_end = (i + 1) / max(1, len(tangents) - 1) if not is_loop else (i + 1) / len(tangents)
            v_start = i
            v_end = i + 1
            UV_list.append([u, v_start])
            UV_list.append([u, v_end])

        rings_in.append(ring_start_indices)
        rings_out.append(ring_end_indices)

        for j in range(radial_segs):
            a = ring_start_indices[j]
            b = ring_start_indices[(j + 1) % radial_segs]
            c = ring_end_indices[j]
            d = ring_end_indices[(j + 1) % radial_segs]
            F_list.extend([[a, b, c], [d, c, b]])

    V_array = np.array(V_list)
    UV_array = np.array(UV_list)
    new_verts_for_joins = []
    new_faces_for_joins = []
    new_uvs_for_joins = []
    
    joint_indices = range(N) if is_loop else range(1, N - 1)
    for i in joint_indices:
        idx_in = (i - 1 + N) % N
        idx_out = i

        ring_v_indices_in = rings_out[idx_in]
        ring_v_indices_out = rings_in[idx_out]
        
        if smooth_joints and joint_segs > 0:
            center, radius = verts[i], radii[i]
            
            p_in_vectors = V_array[ring_v_indices_in] - center
            p_in_vectors /= np.linalg.norm(p_in_vectors, axis=1, keepdims=True)
            p_out_vectors = V_array[ring_v_indices_out] - center
            p_out_vectors /= np.linalg.norm(p_out_vectors, axis=1, keepdims=True)

            V_array[ring_v_indices_in] = center + radius * p_in_vectors
            V_array[ring_v_indices_out] = center + radius * p_out_vectors
            
            omega = np.arccos(np.clip(np.dot(p_in_vectors[0], p_out_vectors[0]), -1, 1))

            if omega < 1e-6 or np.isnan(omega) or abs(omega-np.pi) < 1e-6:
                for j in range(radial_segs): # Fallback: connect rings directly
                    a,b,c,d = ring_v_indices_in[j], ring_v_indices_in[(j+1)%radial_segs], ring_v_indices_out[j], ring_v_indices_out[(j+1)%radial_segs]
                    new_faces_for_joins.extend([[a, b, c], [d, c, b]])
                continue
            
            sin_omega = np.sin(omega)
            all_rings_indices = [ring_v_indices_in]
            base_v_idx = len(V_array) + len(new_verts_for_joins)

            for k in range(1, joint_segs):
                t = float(k) / joint_segs
                current_ring_indices = []
                for j in range(radial_segs):
                    slerp_vec = (np.sin((1-t)*omega)/sin_omega) * p_in_vectors[j] + (np.sin(t*omega)/sin_omega) * p_out_vectors[j]
                    new_verts_for_joins.append(center + radius * slerp_vec)
                    current_ring_indices.append(base_v_idx)
                    base_v_idx += 1
                    
                    u = j / (radial_segs // 2)
                    u = 2 - u if u > 1 else u
                    # v = i / max(1, len(tangents) - 1) if not is_loop else i / len(tangents)
                    v = i
                    new_uvs_for_joins.append([u, v])
                    
                all_rings_indices.append(current_ring_indices)
            all_rings_indices.append(ring_v_indices_out)

            for k in range(len(all_rings_indices) - 1):
                ring_A, ring_B = all_rings_indices[k], all_rings_indices[k+1]
                for j in range(radial_segs):
                    a,b,c,d = ring_A[j], ring_A[(j+1)%radial_segs], ring_B[j], ring_B[(j+1)%radial_segs]
                    new_faces_for_joins.extend([[a, b, c], [d, c, b]])
        else: # Miter join
            t_in, t_out = -tangents[idx_in], tangents[idx_out]
            miter_normal = t_in + t_out
            if np.linalg.norm(miter_normal) < 1e-8: continue
            miter_normal /= np.linalg.norm(miter_normal)
            for v_idx in ring_v_indices_in:
                p = V_array[v_idx]
                dist = np.dot(p - verts[i], miter_normal)
                V_array[v_idx] = p - dist * miter_normal
            for j in range(radial_segs): V_array[ring_v_indices_out[j]] = V_array[ring_v_indices_in[j]]
    
    if new_verts_for_joins:
        V_array = np.vstack([V_array, np.array(new_verts_for_joins)])
    V_list = V_array.tolist()
    F_list.extend(new_faces_for_joins)
    if new_uvs_for_joins:
        UV_array = np.vstack([UV_array, np.array(new_uvs_for_joins)])
    UV_list = UV_array.tolist()

    if not is_loop and endcaps and cap_segs > 0:
        for cap_type in ["start", "end"]:
            is_start = cap_type == "start"
            center, radius = (verts[0], radii[0]) if is_start else (verts[-1], radii[-1])
            tangent = -tangents[0] if is_start else tangents[-1]
            normal, binormal = basis_list[0] if is_start else basis_list[-1]
            prev_ring_indices = rings_in[0] if is_start else rings_out[-1]

            for k in range(1, cap_segs + 1):
                alpha, is_pole = k * (np.pi/2) / cap_segs, k == cap_segs
                ring_radius, displacement = radius * np.cos(alpha), radius * np.sin(alpha)
                ring_center = center + displacement * tangent
                current_ring_indices = []
                if not is_pole:
                    for j in range(radial_segs):
                        theta = 2*np.pi * j / radial_segs
                        offset = np.cos(theta)*normal + np.sin(theta)*binormal
                        V_list.append(ring_center + ring_radius*offset)
                        current_ring_indices.append(len(V_list)-1)
                        
                        u = j / radial_segs
                        v = ring_radius / radius
                        UV_list.append([u, v])
                else:
                    V_list.append(ring_center)
                    current_ring_indices = [len(V_list)-1] * radial_segs
                    UV_list.append([0.0, 0.0])  # Center of endcap
                
                for j in range(radial_segs):
                    a,b = prev_ring_indices[j], prev_ring_indices[(j+1)%radial_segs]
                    c,d = current_ring_indices[j], current_ring_indices[(j+1)%radial_segs]
                    if not is_pole:
                        faces = [[a,c,b], [d,b,c]] if is_start else [[a,b,c], [d,c,b]]
                        F_list.extend(faces)
                    else:
                        F_list.append([b,c,a] if is_start else [a,b,c])
                prev_ring_indices = current_ring_indices

    mesh = trimesh.Trimesh(
        vertices=np.array(V_list),
        faces=np.array(F_list, dtype=int),
        visual=trimesh.visual.TextureVisuals(uv=np.array(UV_list)),
        process=True,
    )
    return mesh
