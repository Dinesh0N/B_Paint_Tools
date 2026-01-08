import math
import array

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from . import utils
from . import ui


def get_text_content(props):
    """Get text content from either text property or text block.
    
    Returns the text block content if use_text_block is enabled and a text block is selected,
    otherwise returns the simple text property.
    """
    if props.use_text_block and props.text_block:
        return props.text_block.as_string()
    return props.text

# ----------------------------
# Operators (3D Texture Paint)
# ----------------------------
class TEXTURE_PAINT_OT_text_tool(Operator):
    bl_idname = "paint.text_tool_ttf"
    bl_label = "Text Tool (TTF/OTF)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    
    # Inline adjustment mode state
    _adjust_mode = None  # None, 'SIZE', or 'ROTATION'
    _adjust_start_value = 0
    _adjust_start_mouse_x = 0

    @classmethod
    def poll(cls, context):
        return (context.mode == 'PAINT_TEXTURE' and
                context.active_object and
                context.active_object.type == 'MESH')

    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        props = context.scene.text_tool_properties
        
        # Handle inline adjustment mode
        if self._adjust_mode:
            if event.type == 'MOUSEMOVE':
                delta = event.mouse_x - self._adjust_start_mouse_x
                if self._adjust_mode == 'SIZE':
                    new_size = int(self._adjust_start_value + delta * 0.5)
                    props.font_size = max(8, min(512, new_size))
                    context.area.header_text_set(f"Font Size: {props.font_size}  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                elif self._adjust_mode == 'ROTATION':
                    new_rotation = self._adjust_start_value + delta * 0.01
                    new_rotation = new_rotation % (2 * math.pi)
                    if new_rotation < 0:
                        new_rotation += 2 * math.pi
                    props.rotation = new_rotation
                    context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                return {'RUNNING_MODAL'}
            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                # Confirm adjustment
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment
                if self._adjust_mode == 'SIZE':
                    props.font_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'ROTATION':
                    props.rotation = self._adjust_start_value
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}
        
        # Check for adjustment shortcut keys
        if event.type == 'F' and event.value == 'PRESS':
            if event.ctrl:
                # Ctrl+F: Rotation adjustment
                self._adjust_mode = 'ROTATION'
                self._adjust_start_value = props.rotation
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            else:
                # F: Font size adjustment
                self._adjust_mode = 'SIZE'
                self._adjust_start_value = props.font_size
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE':
            utils.cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            utils.show_cursor = True
            
            # Calculate 3D preview scale (WYSIWYG)
            # We need to know how big 'N' pixels on the texture look on screen.
            obj = context.active_object
            if obj and obj.type == 'MESH':
                try:
                    from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d, location_3d_to_region_2d
                    from mathutils import Vector
                    
                    region = context.region
                    rv3d = context.region_data
                    coord = (event.mouse_region_x, event.mouse_region_y)
                    
                    view_vector = region_2d_to_vector_3d(region, rv3d, coord)
                    view_origin = region_2d_to_origin_3d(region, rv3d, coord)
                    
                    # Convert to local space
                    mat_inv = obj.matrix_world.inverted()
                    ray_origin = mat_inv @ view_origin
                    ray_target = mat_inv @ (view_origin + view_vector * 1000)
                    ray_dir = ray_target - ray_origin
                    
                    success, location, normal, face_index = obj.ray_cast(ray_origin, ray_dir)
                    
                    if success:
                        # Calculate Texel Density
                        mesh = obj.data
                        poly = mesh.polygons[face_index]
                        
                        # Get area of polygon in world space (approx/fast)
                        v0 = mesh.vertices[poly.vertices[0]].co
                        v1 = mesh.vertices[poly.vertices[1]].co
                        v2 = mesh.vertices[poly.vertices[2]].co
                        
                        # Apply world scale 
                        s = obj.scale
                        v0 = Vector((v0.x * s.x, v0.y * s.y, v0.z * s.z))
                        v1 = Vector((v1.x * s.x, v1.y * s.y, v1.z * s.z))
                        v2 = Vector((v2.x * s.x, v2.y * s.y, v2.z * s.z))
                        
                        area_world = 0.5 * ((v1 - v0).cross(v2 - v0)).length
                        
                        # Get UV area
                        uv_layer = mesh.uv_layers.active
                        if uv_layer:
                            loop0 = poly.loop_indices[0]
                            loop1 = poly.loop_indices[1]
                            loop2 = poly.loop_indices[2]
                            
                            u0 = uv_layer.data[loop0].uv
                            u1 = uv_layer.data[loop1].uv
                            u2 = uv_layer.data[loop2].uv
                            
                            # 2D cross product
                            area_uv = 0.5 * abs((u1.x - u0.x) * (u2.y - u0.y) - (u1.y - u0.y) * (u2.x - u0.x))
                            
                            if area_uv > 0.000001:
                                ratio = math.sqrt(area_world) / math.sqrt(area_uv) # Meters per 1.0 UV Unit
                                
                                # Get Texture Resolution
                                mat = obj.active_material
                                tex_res = 1024 # Default
                                if mat and mat.use_nodes:
                                    image_node = self.get_active_image_node(mat)
                                    if image_node and image_node.image:
                                        if image_node.image.size[0] > 0:
                                            tex_res = image_node.image.size[0]
                                
                                # Meters per Pixel
                                meters_per_pixel = ratio / tex_res
                                
                                # Font size is Texture Pixels
                                test_size_px = 100.0
                                world_size = test_size_px * meters_per_pixel
                                
                                # Project to screen to measure size
                                world_pos = obj.matrix_world @ location
                                
                                # Project Center
                                p2d_center = location_3d_to_region_2d(region, rv3d, world_pos)
                                
                                # Project Offset (perpendicular to view)
                                view_quat = rv3d.view_rotation
                                camera_right = view_quat @ Vector((1.0, 0.0, 0.0))
                                world_pos_offset = world_pos + camera_right * world_size
                                p2d_offset = location_3d_to_region_2d(region, rv3d, world_pos_offset)
                                
                                if p2d_center and p2d_offset:
                                    dist_screen = (p2d_offset - p2d_center).length
                                    utils.cursor_pixel_scale = dist_screen / test_size_px
                                else:
                                    utils.cursor_pixel_scale = 1.0
                            else:
                                utils.cursor_pixel_scale = 1.0
                        else:
                            utils.cursor_pixel_scale = 1.0
                    else:
                        pass # Keep previous scale or reset? Keeping helps smoothness
                except Exception as e:
                    # Keep silent to avoid spam
                    pass
            
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            success = self.place_text_at_surface(context, event)
            if success:
                self.report({'INFO'}, f"Text '{props.text}' placed.")
                self.remove_handler(context)
                return {'FINISHED'}
            else:
                self.remove_handler(context)
                return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            utils.show_cursor = False
            self.remove_handler(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            args = ()
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                ui.draw_cursor_callback_3d, args, 'WINDOW', 'POST_PIXEL')
            utils.show_cursor = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
            
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.show_cursor = False
        context.area.tag_redraw()

    def place_text_at_surface(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({'WARNING'}, "No active material with nodes found")
            return False
        image_node = self.get_active_image_node(mat)
        if not image_node or not image_node.image:
            self.report({'WARNING'}, "No active image texture found")
            return False

        props = context.scene.text_tool_properties
        
        hit_location, face_index, hit_uv, tangent_world, bitangent_world = self.view3d_raycast_uv(context, event, obj)
        if hit_location is None:
            self.report({'WARNING'}, "No surface intersection found")
            return False

        # Check projection mode
        if props.projection_mode == 'VIEW':
            # 3D Projected mode - project text from screen space onto texture
            self.render_text_projected_3d(context, event, obj, image_node.image, hit_location, face_index)
        else:
            # UV-based mode (original behavior)
            # Calculate view-aware rotation for the text in UV space
            uv_rotation = self.calculate_uv_rotation(context, event, tangent_world, bitangent_world)
            
            # Calculate screen-space size adjustment
            tex_font_size = props.font_size
            if utils.cursor_pixel_scale > 0.001:
                tex_font_size = int(props.font_size / utils.cursor_pixel_scale)
            
            self.render_text_to_image(context, image_node.image, hit_uv, uv_rotation, override_font_size=tex_font_size)
        
        # Force refresh to ensure 3D viewport updates immediately
        utils.force_texture_refresh(context, image_node.image)
        return True
        return True
    
    def calculate_uv_rotation(self, context, event, tangent_world, bitangent_world):
        """Calculate the rotation needed in UV space so text appears correctly oriented from the current view.
        
        The idea: Project the UV tangent (U direction) onto the screen plane.
        The angle of this projection tells us how text should be rotated in UV space
        to appear horizontal (or with user's desired rotation) on screen.
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        props = context.scene.text_tool_properties
        user_rotation = math.degrees(props.rotation)
        
        if tangent_world is None:
            # Fallback: just use user rotation
            return user_rotation
        
        region = context.region
        rv3d = context.region_data
        
        if not rv3d:
            return user_rotation
        
        # Get a reference point (use view origin projected towards scene)
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()
        
        # Create a point on the surface (approximate)
        ref_point = view_origin + view_dir * 5.0  # Arbitrary distance
        
        # Project tangent endpoint onto screen
        tangent_end = ref_point + tangent_world * 0.1
        
        ref_2d = location_3d_to_region_2d(region, rv3d, ref_point)
        tangent_2d = location_3d_to_region_2d(region, rv3d, tangent_end)
        
        if not props.align_to_view:
            # 3D Coordinate / Raw UV alignment
            # Blender internal images are bottom-up, UVs are bottom-up 0..1.
            # But visually, text might appear inverted depending on UV mapping.
            # Usually, standard UV mapping requires 180 flip or 0 depending on unwrapping convention.
            # Let's assume standard behavior: return user_rotation directly (maybe with coordinate flip).
            # Pillow renders text top-down. Blender UVs are bottom-up.
            # render_text_to_image implementation might already handle flipping or not.
            # Let's restart with user_rotation. 
            # Note: In view-aligned mode, we calculated compensation.
            # Here we just want strict UV mapping.
            # Ideally: 0 degrees = Text "Up" aligns with UV "V" direction.
            # Based on testing, 0 degrees might need offset.
            # Let's keep it simple: return user_rotation.
            return user_rotation

        if ref_2d is None or tangent_2d is None:
            return user_rotation
        
        # Calculate screen-space angle of the UV U-direction
        dx = tangent_2d.x - ref_2d.x
        dy = tangent_2d.y - ref_2d.y
        
        # Angle in degrees (atan2 gives angle from positive X axis)
        screen_angle = math.degrees(math.atan2(dy, dx))
        
        # The text in UV space should be rotated by the NEGATIVE of this angle
        # to appear horizontal on screen, plus any user-specified rotation
        uv_rotation = -screen_angle + user_rotation
        
        return uv_rotation

    def get_active_image_node(self, material):
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                return node
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                return node
        return None

    def build_bvh(self, obj):
        bm = bmesh.new()
        bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        return bvh

    def view3d_raycast_uv(self, context, event, obj):
        region = context.region
        rv3d = context.region_data
        if not rv3d:
            return None, None, None, None, None

        # Build ray from mouse in world space
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()

        # Define near/far along the ray
        near = view_origin + view_dir * 0.001
        far = view_origin + view_dir * 1e6

        # Transform to object local space
        inv = obj.matrix_world.inverted()
        ro_local = inv @ near
        rf_local = inv @ far
        rd_local = (rf_local - ro_local).normalized()

        # Create Evaluated BMesh
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Build BVH
        bvh = BVHTree.FromBMesh(bm)
        hit = bvh.ray_cast(ro_local, rd_local)
        
        if not hit or not hit[0]:
            bm.free()
            return None, None, None, None, None

        hit_loc_local, hit_normal_local, face_index, distance = hit
        
        # Get UV from BMesh
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return hit_loc_local, face_index, None, None, None

        face = bm.faces[face_index]
        p = hit_loc_local
        
        # Use Blender's built-in polygon interpolation for accurate UV calculation
        from mathutils.interpolate import poly_3d_calc
        
        # Get vertex coordinates and UV coordinates from the face
        vert_coords = [v.co for v in face.verts]
        loop_uvs = [loop[uv_layer].uv for loop in face.loops]
        
        # Calculate interpolation weights for point p on the polygon
        weights = poly_3d_calc(vert_coords, p)
        
        # Interpolate UV using the calculated weights
        u_interp = sum(w * uv.x for w, uv in zip(weights, loop_uvs))
        v_interp = sum(w * uv.y for w, uv in zip(weights, loop_uvs))
        best_uv = Vector((u_interp, v_interp))

        result_uv = (best_uv.x, best_uv.y)
        
        # Calculate UV tangent vectors (dP/dU and dP/dV) for rotation
        # We need at least 2 edges to compute tangent/bitangent
        tangent_local = None
        bitangent_local = None
        
        loops = list(face.loops)
        if len(loops) >= 3:
            # Use first triangle of the face for tangent calculation
            p0 = loops[0].vert.co
            p1 = loops[1].vert.co
            p2 = loops[2].vert.co
            
            uv0 = loops[0][uv_layer].uv
            uv1 = loops[1][uv_layer].uv
            uv2 = loops[2][uv_layer].uv
            
            # Edge vectors in 3D
            edge1 = p1 - p0
            edge2 = p2 - p0
            
            # Edge vectors in UV
            duv1 = uv1 - uv0
            duv2 = uv2 - uv0
            
            # Tangent and bitangent calculation
            denom = duv1.x * duv2.y - duv2.x * duv1.y
            if abs(denom) > 1e-8:
                r = 1.0 / denom
                # Tangent: direction of increasing U in 3D space
                tangent_local = (edge1 * duv2.y - edge2 * duv1.y) * r
                # Bitangent: direction of increasing V in 3D space  
                bitangent_local = (edge2 * duv1.x - edge1 * duv2.x) * r
                tangent_local.normalize()
                bitangent_local.normalize()
        
        bm.free()
        
        # Transform tangent/bitangent to world space
        tangent_world = None
        bitangent_world = None
        if tangent_local and bitangent_local:
            # Use the rotation part of the matrix (ignore translation/scale)
            mat_rot = obj.matrix_world.to_3x3().normalized()
            tangent_world = (mat_rot @ tangent_local).normalized()
            bitangent_world = (mat_rot @ bitangent_local).normalized()
        
        return hit_loc_local, face_index, result_uv, tangent_world, bitangent_world

    def render_text_to_image(self, context, image, uv_coord, view_rotation=None, override_font_size=None):
        props = context.scene.text_tool_properties
        width, height = image.size

        if uv_coord is None:
            self.report({'WARNING'}, "UV not found on that face")
            return

        u, v = uv_coord
        # Handle repeating textures (wrap UVs to 0-1 range)
        u = u % 1.0
        v = v % 1.0
        
        # Convert UV to pixel coordinates
        # UV (0,0) = bottom-left of image, UV (1,1) = top-right
        # Blender image pixels: row 0 = bottom, row (height-1) = top
        x = int(u * width)
        y = int(v * height)

        # Use view_rotation if provided (3D viewport), otherwise use user rotation (2D/fallback)
        if view_rotation is not None:
            rotation_degrees = view_rotation
        else:
            rotation_degrees = math.degrees(props.rotation)
            
        font_size = override_font_size if override_font_size is not None else props.font_size
        
        # Build gradient info if gradient is enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if outline is enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        # Get font path from the vector font object
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, size = utils.FontManager.create_text_image(text_content, font_path, font_size, props.color, rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info, alignment=props.text_alignment, line_spacing=props.line_spacing)
        if t_pixels is None or size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        tw, th = size

        # Apply horizontal anchor offset
        if props.anchor_horizontal == 'CENTER':
            x -= tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            x -= tw
        # LEFT: no offset
        
        # Apply vertical anchor offset
        if props.anchor_vertical == 'CENTER':
            y -= th // 2
        elif props.anchor_vertical == 'TOP':
            y -= th
        # BOTTOM: no offset

        # Save state for undo before modifying
        utils.ImageUndoStack.get().push_state(image)
        
        base = list(image.pixels)
        
        # Get blend mode from active brush
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend

        # Native rendering is bottom-up, matching Blender's image format
        for ty in range(th):
            by = y + ty
            if by < 0 or by >= height:
                continue
            for tx in range(tw):
                bx = x + tx
                if bx < 0 or bx >= width:
                    continue
                t_idx = (ty * tw + tx) * 4
                b_idx = (by * width + bx) * 4
                tr, tg, tb, ta = t_pixels[t_idx:t_idx + 4]
                if ta > 0:
                    utils.blend_pixel(base, b_idx, tr, tg, tb, ta, blend_mode)
        image.pixels = base

    def render_text_projected_3d(self, context, event, obj, image, hit_location_local, hit_face_index):
        """Render text projected from screen space onto the texture.
        
        Optimized version with:
        - Screen-space face bounding box culling
        - Bilinear texture sampling for quality
        - Pre-computed matrices
        - Reduced per-pixel overhead
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        props = context.scene.text_tool_properties
        region = context.region
        rv3d = context.region_data
        
        width, height = image.size
        if width == 0 or height == 0:
            return
        
        # 1. Get cursor position in screen space (center of text)
        cursor_x, cursor_y = event.mouse_region_x, event.mouse_region_y
        
        # 2. Render text to a buffer
        rotation_degrees = math.degrees(props.rotation)
        
        # Build gradient info if enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, t_size = utils.FontManager.create_text_image(
            text_content, font_path, props.font_size, props.color,
            rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info,
            alignment=props.text_alignment, line_spacing=props.line_spacing
        )
        if t_pixels is None or t_size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        
        tw, th = t_size
        
        # Text bounding box in screen space (using anchor properties)
        # Horizontal anchor
        if props.anchor_horizontal == 'CENTER':
            text_left = cursor_x - tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            text_left = cursor_x - tw
        else:  # LEFT
            text_left = cursor_x
        
        # Vertical anchor
        if props.anchor_vertical == 'CENTER':
            text_bottom = cursor_y - th // 2
        elif props.anchor_vertical == 'TOP':
            text_bottom = cursor_y - th
        else:  # BOTTOM
            text_bottom = cursor_y
        text_right = text_left + tw
        text_top = text_bottom + th
        
        # 3. Build BMesh from evaluated object 
        depsgraph = context.evaluated_depsgraph_get()
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            self.report({'WARNING'}, "No active UV layer found")
            return
        
        # 4. Pre-compute view info
        view_origin = region_2d_to_origin_3d(region, rv3d, (cursor_x, cursor_y))
        view_dir = region_2d_to_vector_3d(region, rv3d, (cursor_x, cursor_y)).normalized()
        mat_world = obj.matrix_world
        mat_world_3x3 = mat_world.to_3x3()
        
        # 5. Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        # Performance: Use array.array for pixel buffer
        # 'f' is for float (each pixel is 4 floats: R, G, B, A)
        num_pixels = width * height * 4
        base = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(base)
        
        # Get blend mode
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend
        
        # 6. Pre-filter faces by screen-space bounding box overlap & backface culling
        candidate_faces = []
        for face in bm.faces:
            # Better Backface Culling: Use face center to camera vector
            face_center_world = mat_world @ face.calc_center_median()
            face_to_camera = (view_origin - face_center_world).normalized()
            face_normal_world = (mat_world_3x3 @ face.normal).normalized()
            
            # Dot product for culling and grazing angle check
            dot = face_normal_world.dot(face_to_camera)
            
            # Reject if backfacing OR if at a very grazing angle (splatter prevention)
            if dot < 0.1:  # Threshold of ~84 degrees
                continue
            
            # Project all face vertices to screen and compute screen bounding box
            screen_coords = []
            valid_verts = 0
            for loop in face.loops:
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos:
                    screen_coords.append(screen_pos)
                    valid_verts += 1
            
            if valid_verts < 3:
                continue
            
            # Get screen bounding box
            face_screen_left = min(s.x for s in screen_coords)
            face_screen_right = max(s.x for s in screen_coords)
            face_screen_bottom = min(s.y for s in screen_coords)
            face_screen_top = max(s.y for s in screen_coords)
            
            # Check overlap with text bounds
            if (face_screen_right < text_left or face_screen_left > text_right or
                face_screen_top < text_bottom or face_screen_bottom > text_top):
                continue
            
            # This face overlaps - collect data
            face_uvs = [loop[uv_layer].uv.copy() for loop in face.loops]
            face_verts = [loop.vert.co.copy() for loop in face.loops]
            
            # Store distance for depth sorting
            dist = (face_center_world - view_origin).length
            
            candidate_faces.append({
                'uvs': face_uvs,
                'verts': face_verts,
                'screen': screen_coords,
                'dist': dist,
                'dot': dot
            })
            
        # Sort candidate faces front-to-back
        candidate_faces.sort(key=lambda f: f['dist'])
        
        # 7. Process candidate faces
        processed_pixels = set()
        
        # Determine depth threshold based on initial hit
        primary_depth = candidate_faces[0]['dist'] if candidate_faces else 0
        depth_threshold = 0.5  # Max allowed depth spread (in Blender units)
        
        for f_data in candidate_faces:
            face_uvs = f_data['uvs']
            face_verts = f_data['verts']
            n_verts = len(face_uvs)
            
            # Depth culling: Skip faces too far behind the primary hit
            if f_data['dist'] > primary_depth + depth_threshold:
                continue
            
            # Get UV bounding box
            uv_min_u = min(uv.x for uv in face_uvs)
            uv_max_u = max(uv.x for uv in face_uvs)
            uv_min_v = min(uv.y for uv in face_uvs)
            uv_max_v = max(uv.y for uv in face_uvs)
            
            px_min_x = max(0, int(uv_min_u * width))
            px_max_x = min(width, int(uv_max_u * width) + 1)
            px_min_y = max(0, int(uv_min_v * height))
            px_max_y = min(height, int(uv_max_v * height) + 1)
            
            uv_coords = [(uv.x, uv.y) for uv in face_uvs]
            vert_coords = [(v.x, v.y, v.z) for v in face_verts]
            
            for py in range(px_min_y, px_max_y):
                for px in range(px_min_x, px_max_x):
                    pixel_key = (px, py)
                    
                    if pixel_key in processed_pixels:
                        continue
                    
                    tex_u = (px + 0.5) / width
                    tex_v = (py + 0.5) / height
                    
                    inside = False
                    j = n_verts - 1
                    for i in range(n_verts):
                        if ((uv_coords[i][1] > tex_v) != (uv_coords[j][1] > tex_v)) and \
                           (tex_u < (uv_coords[j][0] - uv_coords[i][0]) * (tex_v - uv_coords[i][1]) / (uv_coords[j][1] - uv_coords[i][1]) + uv_coords[i][0]):
                            inside = not inside
                        j = i
                    
                    if not inside:
                        continue
                    
                    weights = self._uv_to_barycentric_fast(tex_u, tex_v, uv_coords, n_verts)
                    if weights is None:
                        continue
                    
                    lx = ly = lz = 0.0
                    for i, w in enumerate(weights):
                        lx += w * vert_coords[i][0]
                        ly += w * vert_coords[i][1]
                        lz += w * vert_coords[i][2]
                    
                    world_pos = mat_world @ Vector((lx, ly, lz))
                    screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                    if screen_pos is None:
                        continue
                    
                    sx, sy = screen_pos.x, screen_pos.y
                    if sx < text_left or sx >= text_right or sy < text_bottom or sy >= text_top:
                        continue
                    
                    # Mark pixel as processed as soon as we know it's under the text area
                    # Since we sort by depth, the first face to cover this UV pixel wins.
                    processed_pixels.add(pixel_key)
                    
                    tx_f = sx - text_left
                    ty_f = sy - text_bottom
                    
                    tx0 = int(tx_f)
                    ty0 = int(ty_f)
                    tx1 = min(tx0 + 1, tw - 1)
                    ty1 = min(ty0 + 1, th - 1)
                    fx, fy = tx_f - tx0, ty_f - ty0
                    
                    def sample(x, y):
                        if 0 <= x < tw and 0 <= y < th:
                            idx = (y * tw + x) * 4
                            return t_pixels[idx:idx+4]
                        return (0, 0, 0, 0)
                    
                    c00 = sample(tx0, ty0)
                    c10 = sample(tx1, ty0)
                    c01 = sample(tx0, ty1)
                    c11 = sample(tx1, ty1)
                    
                    tr = c00[0]*(1-fx)*(1-fy) + c10[0]*fx*(1-fy) + c01[0]*(1-fx)*fy + c11[0]*fx*fy
                    tg = c00[1]*(1-fx)*(1-fy) + c10[1]*fx*(1-fy) + c01[1]*(1-fx)*fy + c11[1]*fx*fy
                    tb = c00[2]*(1-fx)*(1-fy) + c10[2]*fx*(1-fy) + c01[2]*(1-fx)*fy + c11[2]*fx*fy
                    ta = c00[3]*(1-fx)*(1-fy) + c10[3]*fx*(1-fy) + c01[3]*(1-fx)*fy + c11[3]*fx*fy
                    
                    if ta > 0.001:
                        # INLINED BLENDING for performance
                        b_idx = (py * width + px) * 4
                        dr, dg, db, da = base[b_idx], base[b_idx+1], base[b_idx+2], base[b_idx+3]
                        
                        inv_ta = 1.0 - ta
                        
                        # Default MIX mode logic
                        if blend_mode == 'MIX':
                            base[b_idx]   = tr * ta + dr * inv_ta
                            base[b_idx+1] = tg * ta + dg * inv_ta
                            base[b_idx+2] = tb * ta + db * inv_ta
                            base[b_idx+3] = ta + da * inv_ta
                        else:
                            # Fallback to function for complex modes
                            utils.blend_pixel(base, b_idx, tr, tg, tb, ta, blend_mode)
        
        bm.free()
        image.pixels.foreach_set(base)
    
    def _uv_to_barycentric_fast(self, u, v, uv_coords, n):
        """Optimized barycentric calculation using tuples/inline logic."""
        if n == 3:
            u0, v0 = uv_coords[0]
            u1, v1 = uv_coords[1]
            u2, v2 = uv_coords[2]
            
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-10:
                return None
            
            w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
            w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
            w2 = 1.0 - w0 - w1
            
            return [w0, w1, w2]
        
        elif n == 4:
            # Fast quad split
            u0, v0 = uv_coords[0]
            u1, v1 = uv_coords[1]
            u2, v2 = uv_coords[2]
            u3, v3 = uv_coords[3]
            
            # Triangle 0-1-2
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) > 1e-10:
                w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
                w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                    return [w0, w1, w2, 0.0]
            
            # Triangle 0-2-3
            denom = (v2 - v3) * (u0 - u3) + (u3 - u2) * (v0 - v3)
            if abs(denom) > 1e-10:
                w0 = ((v2 - v3) * (u - u3) + (u3 - u2) * (v - v3)) / denom
                w2 = ((v3 - v0) * (u - u3) + (u0 - u3) * (v - v3)) / denom
                w3 = 1.0 - w0 - w2
                if w0 >= -0.01 and w2 >= -0.01 and w3 >= -0.01:
                    return [w0, 0.0, w2, w3]
            
        # Fallback to slower but robust method
        from mathutils import Vector
        from mathutils.interpolate import poly_3d_calc
        pts = [Vector((uv[0], uv[1], 0.0)) for uv in uv_coords]
        p = Vector((u, v, 0.0))
        try:
            return list(poly_3d_calc(pts, p))
        except:
            return [1.0 / n] * n
    
    def _point_in_polygon_2d(self, point, polygon):
        """Check if a 2D point is inside a 2D polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _uv_to_barycentric(self, u, v, face_uvs):
        """Convert UV coordinates to barycentric weights for the face.
        
        For triangles, use standard barycentric. For quads/ngons, use 
        generalized barycentric coordinates (mean value coordinates).
        """
        n = len(face_uvs)
        
        if n == 3:
            # Triangle: standard barycentric
            u0, v0 = face_uvs[0].x, face_uvs[0].y
            u1, v1 = face_uvs[1].x, face_uvs[1].y
            u2, v2 = face_uvs[2].x, face_uvs[2].y
            
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-10:
                return None
            
            w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
            w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
            w2 = 1.0 - w0 - w1
            
            return [w0, w1, w2]
        
        else:
            # Poly-based interpolation (more robust for Quads and N-gons)
            from mathutils import Vector
            from mathutils.interpolate import poly_3d_calc
            pts = [Vector((uv.x, uv.y, 0.0)) for uv in face_uvs]
            p = Vector((u, v, 0.0))
            try:
                weights = poly_3d_calc(pts, p)
                return list(weights)
            except:
                return [1.0 / n] * n

class IMAGE_PAINT_OT_text_tool(Operator):
    bl_idname = "image_paint.text_tool_ttf"
    bl_label = "Image Text Tool (TTF/OTF)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    
    # Inline adjustment mode state
    _adjust_mode = None  # None, 'SIZE', or 'ROTATION'
    _adjust_start_value = 0
    _adjust_start_mouse_x = 0

    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and sima.mode == 'PAINT' and sima.image is not None)

    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        props = context.scene.text_tool_properties
        
        # Handle inline adjustment mode
        if self._adjust_mode:
            if event.type == 'MOUSEMOVE':
                delta = event.mouse_x - self._adjust_start_mouse_x
                if self._adjust_mode == 'SIZE':
                    new_size = int(self._adjust_start_value + delta * 0.5)
                    props.font_size = max(8, min(512, new_size))
                    context.area.header_text_set(f"Font Size: {props.font_size}  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                elif self._adjust_mode == 'ROTATION':
                    new_rotation = self._adjust_start_value + delta * 0.01
                    new_rotation = new_rotation % (2 * math.pi)
                    if new_rotation < 0:
                        new_rotation += 2 * math.pi
                    props.rotation = new_rotation
                    context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                return {'RUNNING_MODAL'}
            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                # Confirm adjustment
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment
                if self._adjust_mode == 'SIZE':
                    props.font_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'ROTATION':
                    props.rotation = self._adjust_start_value
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}
        
        # Check for adjustment shortcut keys
        if event.type == 'F' and event.value == 'PRESS':
            if event.ctrl:
                # Ctrl+F: Rotation adjustment
                self._adjust_mode = 'ROTATION'
                self._adjust_start_value = props.rotation
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            else:
                # F: Font size adjustment
                self._adjust_mode = 'SIZE'
                self._adjust_start_value = props.font_size
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE':
            utils.cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            utils.show_cursor = True
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            ok = self.place_text_in_image(context, event)
            if ok:
                self.report({'INFO'}, f"Text '{props.text}' placed in image.")
                self.remove_handler(context)
                return {'FINISHED'}
            self.remove_handler(context)
            return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            utils.show_cursor = False
            self.remove_handler(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            args = ()
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_cursor_callback_image, args, 'WINDOW', 'POST_PIXEL')
            utils.show_cursor = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found, cannot run operator")
            return {'CANCELLED'}

    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.show_cursor = False
        context.area.tag_redraw()

    def place_text_in_image(self, context, event):
        sima = context.space_data
        if not sima.image:
            self.report({'WARNING'}, "No active image found")
            return False
        region = context.region
        coord = self.region_to_image_coord(sima, region, event.mouse_region_x, event.mouse_region_y)
        if coord is None:
            self.report({'WARNING'}, "Click outside image bounds")
            return False
        self.render_text_to_image_direct(context, sima.image, coord)
        
        # Force refresh to ensure 3D viewport updates immediately
        utils.force_texture_refresh(context, sima.image)
        
        context.area.tag_redraw()
        return True

    def region_to_image_coord(self, sima, region, mouse_x, mouse_y):
        """Convert region (screen) coordinates to image pixel coordinates.
        
        Properly handles pan and zoom by using the view2d transformation.
        """
        iw, ih = sima.image.size
        if iw == 0 or ih == 0:
            return None
        
        # Use view2d to convert screen coords to UV coords (0-1 range when image fills view)
        # view2d.region_to_view returns coordinates in "view" space
        # For Image Editor, this is in UV units where (0,0) is bottom-left of image
        view_x, view_y = region.view2d.region_to_view(mouse_x, mouse_y)
        
        # Convert UV to pixel coordinates
        # view coords are already in image-normalized space (0 to 1 for the image area)
        x = int(view_x * iw)
        y = int(view_y * ih)
        
        if 0 <= x < iw and 0 <= y < ih:
            return (x, y)
        return None

    def render_text_to_image_direct(self, context, image, coord):
        props = context.scene.text_tool_properties
        width, height = image.size
        x, y = coord
        rotation_degrees = math.degrees(props.rotation)
        
        # Build gradient info if gradient is enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if outline is enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        # Get font path from the vector font object
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, size = utils.FontManager.create_text_image(text_content, font_path, props.font_size, props.color, rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info, alignment=props.text_alignment, line_spacing=props.line_spacing)
        if t_pixels is None or size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        tw, th = size
        
        # Apply horizontal anchor offset
        if props.anchor_horizontal == 'CENTER':
            x -= tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            x -= tw
        # LEFT: no offset
        
        # Apply vertical anchor offset
        if props.anchor_vertical == 'CENTER':
            y -= th // 2
        elif props.anchor_vertical == 'TOP':
            y -= th
        # BOTTOM: no offset

        # Save state for undo before modifying
        utils.ImageUndoStack.get().push_state(image)
        
        base = list(image.pixels)
        
        # Get blend mode from active brush
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend

        # Native rendering is bottom-up, matching Blender's image format
        for ty in range(th):
            by = y + ty
            if by < 0 or by >= height:
                continue
            for tx in range(tw):
                bx = x + tx
                if bx < 0 or bx >= width:
                    continue
                t_idx = (ty * tw + tx) * 4
                b_idx = (by * width + bx) * 4
                tr, tg, tb, ta = t_pixels[t_idx:t_idx + 4]
                if ta > 0:
                    utils.blend_pixel(base, b_idx, tr, tg, tb, ta, blend_mode)
        image.pixels = base


# ----------------------------
# Gradient Tool Operators
# ----------------------------
class TEXTURE_PAINT_OT_gradient_tool(Operator):
    bl_idname = "paint.gradient_tool"
    bl_label = "Gradient Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None
    _end_pos = None
    _is_dragging = False
    
    # Realtime preview state
    _image = None
    _original_pixels = None  # Store original for restore
    _cached_face_data = None  # Pre-computed face UV/screen data
    _width = 0
    _height = 0
    
    # Throttling for performance
    _last_update_time = 0.0
    _last_end_pos = None
    _update_interval = 0.016  # ~60 FPS max
    _min_pos_change = 3  # Minimum pixel change to trigger update
    
    @classmethod
    def poll(cls, context):
        return (context.mode == 'PAINT_TEXTURE' and
                context.active_object and
                context.active_object.type == 'MESH')
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging and self._image:
                import time
                current_time = time.time()
                new_pos = (event.mouse_region_x, event.mouse_region_y)
                
                # Apply 45-degree snapping when Shift is held
                if event.shift and self._start_pos:
                    dx = new_pos[0] - self._start_pos[0]
                    dy = new_pos[1] - self._start_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance > 0:
                        angle = math.atan2(dy, dx)
                        # Snap to nearest 45 degrees (pi/4)
                        snap_angle = round(angle / (math.pi / 4)) * (math.pi / 4)
                        new_pos = (
                            self._start_pos[0] + int(distance * math.cos(snap_angle)),
                            self._start_pos[1] + int(distance * math.sin(snap_angle))
                        )
                
                # Throttle updates
                time_ok = (current_time - self._last_update_time) >= self._update_interval
                
                # Check if position changed enough
                pos_ok = True
                if self._last_end_pos:
                    dx = abs(new_pos[0] - self._last_end_pos[0])
                    dy = abs(new_pos[1] - self._last_end_pos[1])
                    pos_ok = (dx >= self._min_pos_change or dy >= self._min_pos_change)
                
                if time_ok and pos_ok:
                    self._end_pos = new_pos
                    self._last_end_pos = new_pos
                    self._last_update_time = current_time
                    utils.gradient_preview_start = self._start_pos
                    utils.gradient_preview_end = self._end_pos
                    self._apply_gradient_realtime(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # Start dragging - initialize state
                self._is_dragging = True
                self._start_pos = (event.mouse_region_x, event.mouse_region_y)
                self._end_pos = self._start_pos
                utils.gradient_preview_start = self._start_pos
                utils.gradient_preview_end = self._end_pos
                # Initialize realtime state
                if not self._init_realtime_state(context):
                    self._is_dragging = False
                    self.remove_handler(context)
                    return {'CANCELLED'}
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                # Final apply
                self._end_pos = (event.mouse_region_x, event.mouse_region_y)
                self._apply_gradient_realtime(context)
                # Save undo state
                if self._image and self._original_pixels:
                    utils.ImageUndoStack.get().push_state_from_array(self._image, self._original_pixels)
                self._cleanup_state()
                utils.gradient_preview_start = None
                utils.gradient_preview_end = None
                self.remove_handler(context)
                self.report({'INFO'}, "Gradient applied.")
                return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self._image and self._original_pixels:
                self._image.pixels.foreach_set(self._original_pixels)
                utils.force_texture_refresh(context, self._image)
            self._cleanup_state()
            utils.gradient_preview_start = None
            utils.gradient_preview_end = None
            self.remove_handler(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _cleanup_state(self):
        self._is_dragging = False
        self._image = None
        self._original_pixels = None
        self._cached_face_data = None
    
    def _init_realtime_state(self, context):
        """Initialize state for realtime preview: cache image, pixels, and face data."""
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            return False
        
        # Find active image node
        image_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                image_node = node
                break
        if not image_node:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    image_node = node
                    break
        
        if not image_node or not image_node.image:
            return False
        
        self._image = image_node.image
        self._width, self._height = self._image.size
        
        if self._width == 0 or self._height == 0:
            return False
        
        # Store original pixels
        num_pixels = self._width * self._height * 4
        self._original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(self._original_pixels)
        
        # Pre-compute face UV/screen data
        region = context.region
        rv3d = context.region_data
        mat_world = obj.matrix_world
        
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return False
        
        self._cached_face_data = []
        
        for face in bm.faces:
            loops = list(face.loops)
            n_verts = len(loops)
            if n_verts < 3:
                continue
            
            # Get UV and screen coords for each vertex
            face_data = []
            valid = True
            for loop in loops:
                uv = loop[uv_layer].uv
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos is None:
                    valid = False
                    break
                face_data.append((uv.x, uv.y, screen_pos.x, screen_pos.y))
            
            if valid:
                self._cached_face_data.append(face_data)
        
        bm.free()
        return True
    
    def _apply_gradient_realtime(self, context):
        """Apply gradient using cached data - OPTIMIZED with NumPy per-triangle."""
        if not self._image or not self._original_pixels or not self._cached_face_data:
            return
        
        props = context.scene.text_tool_properties
        grad_node = utils.get_gradient_node()
        if not grad_node:
            return
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return
        
        gradient_type = props.gradient_type
        is_linear = (gradient_type == 'LINEAR')
        width, height = self._width, self._height
        
        # Gradient parameters
        sx1, sy1 = self._start_pos
        sx2, sy2 = self._end_pos
        gdx = sx2 - sx1
        gdy = sy2 - sy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        # Try NumPy + multithreading for massive speedup
        try:
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            # Convert LUT to numpy array
            lut_arr = np.array([(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut], dtype=np.float32)
            
            # Get original as numpy
            result = np.array(self._original_pixels, dtype=np.float32).reshape(height, width, 4)
            
            # Flatten triangles for batch processing
            all_triangles = []
            for face_data in self._cached_face_data:
                n_verts = len(face_data)
                for tri_idx in range(n_verts - 2):
                    all_triangles.append((face_data[0], face_data[tri_idx + 1], face_data[tri_idx + 2]))
            
            if not all_triangles:
                self._image.pixels.foreach_set(result.flatten())
                utils.force_texture_refresh(context, self._image)
                return
            
            # Number of threads
            num_threads = min(os.cpu_count() or 4, 8)
            batch_size = max(1, len(all_triangles) // num_threads)
            
            # Process triangles in batch
            def process_triangle_batch(triangles, local_result):
                """Process a batch of triangles."""
                for tri in triangles:
                    v0, v1, v2 = tri
                    u0, v0_uv, sx0, sy0 = v0
                    u1, v1_uv, sx_1, sy_1 = v1
                    u2, v2_uv, sx_2, sy_2 = v2
                    
                    uv_min_u = min(u0, u1, u2)
                    uv_max_u = max(u0, u1, u2)
                    uv_min_v = min(v0_uv, v1_uv, v2_uv)
                    uv_max_v = max(v0_uv, v1_uv, v2_uv)
                    
                    px_min = max(0, int(uv_min_u * width))
                    px_max = min(width, int(uv_max_u * width) + 1)
                    py_min = max(0, int(uv_min_v * height))
                    py_max = min(height, int(uv_max_v * height) + 1)
                    
                    if px_max <= px_min or py_max <= py_min:
                        continue
                    
                    denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                    if abs(denom) < 0.0001:
                        continue
                    inv_denom = 1.0 / denom
                    
                    # Create coordinate grids
                    py_range = np.arange(py_min, py_max, dtype=np.float32)
                    px_range = np.arange(px_min, px_max, dtype=np.float32)
                    py_grid, px_grid = np.meshgrid(py_range, px_range, indexing='ij')
                    
                    tex_u = (px_grid + 0.5) / width
                    tex_v = (py_grid + 0.5) / height
                    
                    w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                    w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                    w2 = 1.0 - w0 - w1
                    
                    inside = (w0 >= -0.001) & (w1 >= -0.001) & (w2 >= -0.001)
                    if not np.any(inside):
                        continue
                    
                    sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                    sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                    
                    if is_linear:
                        t = ((sx - sx1) * gdx + (sy - sy1) * gdy) / grad_len_sq
                    else:
                        t = np.sqrt((sx - sx1)**2 + (sy - sy1)**2) / grad_len
                    
                    t = np.clip(t, 0.0, 1.0)
                    lut_indices = (t * (lut_len - 1)).astype(np.int32)
                    colors = lut_arr[lut_indices]
                    
                    alpha = colors[:, :, 3:4]
                    region_orig = local_result[py_min:py_max, px_min:px_max, :].copy()
                    
                    blended = np.zeros_like(region_orig)
                    blended[:, :, :3] = colors[:, :, :3] * alpha + region_orig[:, :, :3] * (1.0 - alpha)
                    blended[:, :, 3:4] = alpha + region_orig[:, :, 3:4] * (1.0 - alpha)
                    
                    inside_3d = inside[:, :, np.newaxis]
                    local_result[py_min:py_max, px_min:px_max, :] = np.where(inside_3d, blended, region_orig)
            
            # For thread safety, process all triangles serially but with NumPy vectorization
            # Threading for triangle batches can cause race conditions on overlapping regions
            # Instead, we'll use the optimized NumPy code which is already fast
            process_triangle_batch(all_triangles, result)
            
            # Set pixels
            self._image.pixels.foreach_set(result.flatten())
            utils.force_texture_refresh(context, self._image)
            return
            
        except ImportError:
            pass  # Fall back to Python
        
        # Fallback: Python loop (slower)
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        lut_max_idx = lut_len - 1
        base = array.array('f', self._original_pixels)
        
        for face_data in self._cached_face_data:
            n_verts = len(face_data)
            
            for tri_idx in range(n_verts - 2):
                v0 = face_data[0]
                v1 = face_data[tri_idx + 1]
                v2 = face_data[tri_idx + 2]
                
                u0, v0_uv, sx0, sy0 = v0
                u1, v1_uv, sx_1, sy_1 = v1
                u2, v2_uv, sx_2, sy_2 = v2
                
                uv_min_u = min(u0, u1, u2)
                uv_max_u = max(u0, u1, u2)
                uv_min_v = min(v0_uv, v1_uv, v2_uv)
                uv_max_v = max(v0_uv, v1_uv, v2_uv)
                
                px_min_x = max(0, int(uv_min_u * width))
                px_max_x = min(width, int(uv_max_u * width) + 1)
                px_min_y = max(0, int(uv_min_v * height))
                px_max_y = min(height, int(uv_max_v * height) + 1)
                
                denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                if abs(denom) < 0.0001:
                    continue
                inv_denom = 1.0 / denom
                
                for py in range(px_min_y, px_max_y):
                    row_offset = py * width * 4
                    for px in range(px_min_x, px_max_x):
                        tex_u = (px + 0.5) / width
                        tex_v = (py + 0.5) / height
                        
                        w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                        w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                        w2 = 1.0 - w0 - w1
                        
                        if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                            continue
                        
                        sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                        sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                        
                        if is_linear:
                            t = ((sx - sx1) * gdx + (sy - sy1) * gdy) / grad_len_sq
                        else:
                            t = math.sqrt((sx - sx1)**2 + (sy - sy1)**2) / grad_len
                        
                        if t < 0.0: t = 0.0
                        elif t > 1.0: t = 1.0
                        
                        color = lut_cache[int(t * lut_max_idx)]
                        
                        b_idx = row_offset + px * 4
                        ta = color[3]
                        inv_ta = 1.0 - ta
                        
                        base[b_idx]   = color[0] * ta + base[b_idx] * inv_ta
                        base[b_idx+1] = color[1] * ta + base[b_idx+1] * inv_ta
                        base[b_idx+2] = color[2] * ta + base[b_idx+2] * inv_ta
                        base[b_idx+3] = ta + base[b_idx+3] * inv_ta
        
        self._image.pixels.foreach_set(base)
        utils.force_texture_refresh(context, self._image)
    
    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            args = ()
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                ui.draw_gradient_preview_3d, args, 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            utils.gradient_preview_start = self._start_pos
            utils.gradient_preview_end = self._end_pos
            
            # Initialize realtime state
            if not self._init_realtime_state(context):
                self._is_dragging = False
                self.remove_handler(context)
                return {'CANCELLED'}
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
    
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.tag_redraw()
    
    def apply_gradient(self, context, event):
        """Apply gradient to the texture based on start/end positions.
        
        OPTIMIZED: Pre-compute screen positions for face corners and interpolate.
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({'WARNING'}, "No active material with nodes found")
            return False
        
        # Find active image node
        image_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                image_node = node
                break
        if not image_node:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    image_node = node
                    break
        
        if not image_node or not image_node.image:
            self.report({'WARNING'}, "No active image texture found")
            return False
        
        image = image_node.image
        props = context.scene.text_tool_properties
        width, height = image.size
        
        if width == 0 or height == 0:
            return False
        
        # Get gradient LUT
        grad_node = utils.get_gradient_node()
        if not grad_node:
            self.report({'WARNING'}, "No gradient color ramp found")
            return False
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return False
        
        gradient_type = props.gradient_type
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        # Get pixel buffer
        num_pixels = width * height * 4
        base = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(base)
        
        # Get blend mode
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend
        
        # Screen-space gradient parameters
        sx1, sy1 = self._start_pos
        sx2, sy2 = self._end_pos
        
        # Gradient vector
        gdx = sx2 - sx1
        gdy = sy2 - sy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        region = context.region
        rv3d = context.region_data
        mat_world = obj.matrix_world
        
        # Build BMesh for UV lookup
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            self.report({'WARNING'}, "No active UV layer found")
            return False
        
        # Pre-cache LUT for faster access
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        
        # OPTIMIZED: Process triangulated faces with corner screen projection
        for face in bm.faces:
            loops = list(face.loops)
            n_verts = len(loops)
            
            # Get UV and screen coords for each vertex
            face_data = []
            valid = True
            for loop in loops:
                uv = loop[uv_layer].uv
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos is None:
                    valid = False
                    break
                face_data.append((uv.x, uv.y, screen_pos.x, screen_pos.y))
            
            if not valid or n_verts < 3:
                continue
            
            # Triangulate the face for processing
            for tri_idx in range(n_verts - 2):
                v0 = face_data[0]
                v1 = face_data[tri_idx + 1]
                v2 = face_data[tri_idx + 2]
                
                u0, v0_uv, sx0, sy0 = v0
                u1, v1_uv, sx_1, sy_1 = v1
                u2, v2_uv, sx_2, sy_2 = v2
                
                # UV bounding box
                uv_min_u = min(u0, u1, u2)
                uv_max_u = max(u0, u1, u2)
                uv_min_v = min(v0_uv, v1_uv, v2_uv)
                uv_max_v = max(v0_uv, v1_uv, v2_uv)
                
                px_min_x = max(0, int(uv_min_u * width))
                px_max_x = min(width, int(uv_max_u * width) + 1)
                px_min_y = max(0, int(uv_min_v * height))
                px_max_y = min(height, int(uv_max_v * height) + 1)
                
                # Barycentric edge vectors
                denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                if abs(denom) < 0.0001:
                    continue
                inv_denom = 1.0 / denom
                
                for py in range(px_min_y, px_max_y):
                    for px in range(px_min_x, px_max_x):
                        tex_u = (px + 0.5) / width
                        tex_v = (py + 0.5) / height
                        
                        # Barycentric coordinates
                        w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                        w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                        w2 = 1.0 - w0 - w1
                        
                        # Outside triangle
                        if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                            continue
                        
                        # Interpolate screen position (MUCH faster than projection)
                        sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                        sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                        
                        # Calculate gradient factor
                        if gradient_type == 'LINEAR':
                            dx = sx - sx1
                            dy = sy - sy1
                            t = (dx * gdx + dy * gdy) / grad_len_sq
                        else:  # RADIAL
                            dx = sx - sx1
                            dy = sy - sy1
                            t = math.sqrt(dx * dx + dy * dy) / grad_len
                        
                        t = max(0.0, min(1.0, t))
                        
                        # Sample LUT
                        color = lut_cache[int(t * (lut_len - 1))]
                        
                        # Apply gradient
                        b_idx = (py * width + px) * 4
                        dr, dg, db, da = base[b_idx], base[b_idx+1], base[b_idx+2], base[b_idx+3]
                        
                        ta = color[3]
                        inv_ta = 1.0 - ta
                        
                        if blend_mode == 'MIX':
                            base[b_idx]   = color[0] * ta + dr * inv_ta
                            base[b_idx+1] = color[1] * ta + dg * inv_ta
                            base[b_idx+2] = color[2] * ta + db * inv_ta
                            base[b_idx+3] = ta + da * inv_ta
                        else:
                            utils.blend_pixel(base, b_idx, color[0], color[1], color[2], ta, blend_mode)
        
        bm.free()
        image.pixels.foreach_set(base)
        utils.force_texture_refresh(context, image)
        return True


class IMAGE_PAINT_OT_gradient_tool(Operator):
    bl_idname = "image_paint.gradient_tool"
    bl_label = "Image Gradient Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None
    _end_pos = None
    _is_dragging = False
    
    # Realtime preview state
    _image = None
    _original_pixels = None
    _width = 0
    _height = 0
    
    # Throttling for performance
    _last_update_time = 0.0
    _last_end_pos = None
    _update_interval = 0.016  # ~60 FPS max
    _min_pos_change = 3  # Minimum pixel change to trigger update
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and sima.mode == 'PAINT' and sima.image is not None)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging and self._image:
                import time
                current_time = time.time()
                new_pos = (event.mouse_region_x, event.mouse_region_y)
                
                # Apply 45-degree snapping when Shift is held
                if event.shift and self._start_pos:
                    dx = new_pos[0] - self._start_pos[0]
                    dy = new_pos[1] - self._start_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance > 0:
                        angle = math.atan2(dy, dx)
                        # Snap to nearest 45 degrees (pi/4)
                        snap_angle = round(angle / (math.pi / 4)) * (math.pi / 4)
                        new_pos = (
                            self._start_pos[0] + int(distance * math.cos(snap_angle)),
                            self._start_pos[1] + int(distance * math.sin(snap_angle))
                        )
                
                # Throttle updates
                time_ok = (current_time - self._last_update_time) >= self._update_interval
                
                # Check if position changed enough
                pos_ok = True
                if self._last_end_pos:
                    dx = abs(new_pos[0] - self._last_end_pos[0])
                    dy = abs(new_pos[1] - self._last_end_pos[1])
                    pos_ok = (dx >= self._min_pos_change or dy >= self._min_pos_change)
                
                if time_ok and pos_ok:
                    self._end_pos = new_pos
                    self._last_end_pos = new_pos
                    self._last_update_time = current_time
                    utils.gradient_preview_start = self._start_pos
                    utils.gradient_preview_end = self._end_pos
                    self._apply_gradient_realtime(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_dragging = True
                self._start_pos = (event.mouse_region_x, event.mouse_region_y)
                self._end_pos = self._start_pos
                utils.gradient_preview_start = self._start_pos
                utils.gradient_preview_end = self._end_pos
                # Initialize realtime state
                if not self._init_realtime_state(context):
                    self._is_dragging = False
                    self.remove_handler(context)
                    return {'CANCELLED'}
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                self._end_pos = (event.mouse_region_x, event.mouse_region_y)
                self._apply_gradient_realtime(context)
                # Save undo state
                if self._image and self._original_pixels:
                    utils.ImageUndoStack.get().push_state_from_array(self._image, self._original_pixels)
                self._cleanup_state()
                utils.gradient_preview_start = None
                utils.gradient_preview_end = None
                self.remove_handler(context)
                self.report({'INFO'}, "Gradient applied.")
                return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self._image and self._original_pixels:
                self._image.pixels.foreach_set(self._original_pixels)
                self._image.update()
            self._cleanup_state()
            utils.gradient_preview_start = None
            utils.gradient_preview_end = None
            self.remove_handler(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _cleanup_state(self):
        self._is_dragging = False
        self._image = None
        self._original_pixels = None
    
    def _init_realtime_state(self, context):
        """Initialize state for realtime preview."""
        sima = context.space_data
        self._image = sima.image
        if not self._image:
            return False
        
        self._width, self._height = self._image.size
        if self._width == 0 or self._height == 0:
            return False
        
        # Store original pixels
        num_pixels = self._width * self._height * 4
        self._original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(self._original_pixels)
        return True
    
    def _apply_gradient_realtime(self, context):
        """Apply gradient - OPTIMIZED with NumPy vectorization."""
        if not self._image or not self._original_pixels:
            return
        
        props = context.scene.text_tool_properties
        grad_node = utils.get_gradient_node()
        if not grad_node:
            return
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return
        
        gradient_type = props.gradient_type
        is_linear = (gradient_type == 'LINEAR')
        width, height = self._width, self._height
        
        # Convert screen positions to image coordinates
        region = context.region
        v1 = region.view2d.region_to_view(*self._start_pos)
        v2 = region.view2d.region_to_view(*self._end_pos)
        ix1, iy1 = v1[0] * width, v1[1] * height
        ix2, iy2 = v2[0] * width, v2[1] * height
        
        gdx, gdy = ix2 - ix1, iy2 - iy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        # Try NumPy + multithreading for massive speedup
        try:
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            # Convert LUT to numpy array
            lut_arr = np.array([(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut], dtype=np.float32)
            
            # Get original as numpy
            orig = np.array(self._original_pixels, dtype=np.float32).reshape(height, width, 4)
            
            # Number of threads (use available CPUs)
            num_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 threads
            strip_height = max(1, height // num_threads)
            
            # Result array
            final = np.zeros((height, width, 4), dtype=np.float32)
            
            def process_strip(start_y, end_y):
                """Process a horizontal strip of the image."""
                h = end_y - start_y
                w = width
                
                # Create coordinate grids for this strip
                py, px = np.mgrid[start_y:end_y, 0:w]
                dx = px.astype(np.float32) - ix1
                dy = py.astype(np.float32) - iy1
                
                # Calculate gradient factor
                if is_linear:
                    t = (dx * gdx + dy * gdy) / grad_len_sq
                else:
                    t = np.sqrt(dx * dx + dy * dy) / grad_len
                
                # Clamp
                t = np.clip(t, 0.0, 1.0)
                
                # Sample LUT
                lut_indices = (t * (lut_len - 1)).astype(np.int32)
                colors = lut_arr[lut_indices]
                
                # Get original strip
                orig_strip = orig[start_y:end_y, :, :]
                
                # Blend
                alpha = colors[:, :, 3:4]
                result_rgb = colors[:, :, :3] * alpha + orig_strip[:, :, :3] * (1.0 - alpha)
                result_alpha = alpha[:, :, 0] + orig_strip[:, :, 3] * (1.0 - alpha[:, :, 0])
                
                # Write to final
                final[start_y:end_y, :, :3] = result_rgb
                final[start_y:end_y, :, 3] = result_alpha
            
            # Process strips in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    start_y = i * strip_height
                    end_y = min((i + 1) * strip_height, height)
                    if i == num_threads - 1:
                        end_y = height  # Last strip gets remainder
                    if start_y < end_y:
                        futures.append(executor.submit(process_strip, start_y, end_y))
                
                # Wait for all to complete
                for f in futures:
                    f.result()
            
            # Set pixels
            self._image.pixels.foreach_set(final.flatten())
            self._image.update()
            return
            
        except ImportError:
            pass  # Fall back to Python
        
        # Fallback: Python loop (slower)
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        lut_max_idx = lut_len - 1
        base = array.array('f', self._original_pixels)
        
        for py in range(height):
            row_offset = py * width * 4
            dy = py - iy1
            
            for px in range(width):
                dx = px - ix1
                
                if is_linear:
                    t = (dx * gdx + dy * gdy) / grad_len_sq
                else:
                    t = math.sqrt(dx * dx + dy * dy) / grad_len
                
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                
                color = lut_cache[int(t * lut_max_idx)]
                
                b_idx = row_offset + px * 4
                ta = color[3]
                inv_ta = 1.0 - ta
                
                base[b_idx]   = color[0] * ta + base[b_idx] * inv_ta
                base[b_idx+1] = color[1] * ta + base[b_idx+1] * inv_ta
                base[b_idx+2] = color[2] * ta + base[b_idx+2] * inv_ta
                base[b_idx+3] = ta + base[b_idx+3] * inv_ta
        
        self._image.pixels.foreach_set(base)
        self._image.update()
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            args = ()
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_gradient_preview_image, args, 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            utils.gradient_preview_start = self._start_pos
            utils.gradient_preview_end = self._end_pos
            
            # Initialize realtime state
            if not self._init_realtime_state(context):
                self._is_dragging = False
                self.remove_handler(context)
                return {'CANCELLED'}
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.tag_redraw()


# ----------------------------
# Pen Tool (Image Editor)
# ----------------------------

# Global pen tool state for drawing
pen_points = []  # List of (anchor_x, anchor_y, handle_in_x, handle_in_y, handle_out_x, handle_out_y)
pen_preview_pos = None
pen_is_dragging = False
pen_drag_handle = None  # 'IN' or 'OUT'
pen_edit_point_idx = None  # Index of point being edited (for re-edit mode)
pen_edit_element = None  # 'ANCHOR', 'HANDLE_IN', 'HANDLE_OUT'


class IMAGE_PAINT_OT_pen_tool(Operator):
    bl_idname = "image_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths with stroke and fill"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _width = 0
    _height = 0
    _is_closed = False
    _hit_radius = 12  # Pixel radius for hit detection
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _hit_test(self, img_x, img_y):
        """Test if click is near any existing point or handle. Returns (index, element_type) or (None, None)."""
        global pen_points
        
        for i, pt in enumerate(pen_points):
            ax, ay = pt[0], pt[1]
            hi_x, hi_y = pt[2], pt[3]
            ho_x, ho_y = pt[4], pt[5]
            
            # Check anchor point
            dist_anchor = ((img_x - ax)**2 + (img_y - ay)**2)**0.5
            if dist_anchor < self._hit_radius:
                return (i, 'ANCHOR')
            
            # Check handle in (if different from anchor)
            if hi_x != ax or hi_y != ay:
                dist_hi = ((img_x - hi_x)**2 + (img_y - hi_y)**2)**0.5
                if dist_hi < self._hit_radius:
                    return (i, 'HANDLE_IN')
            
            # Check handle out (if different from anchor)
            if ho_x != ax or ho_y != ay:
                dist_ho = ((img_x - ho_x)**2 + (img_y - ho_y)**2)**0.5
                if dist_ho < self._hit_radius:
                    return (i, 'HANDLE_OUT')
        
        return (None, None)
    
    def modal(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_drag_handle
        global pen_edit_point_idx, pen_edit_element
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        # Convert to image coordinates
        region = context.region
        view2d = region.view2d
        uv = view2d.region_to_view(mx, my)
        img_x = uv[0] * self._width
        img_y = uv[1] * self._height
        
        pen_preview_pos = (img_x, img_y)
        
        if event.type == 'MOUSEMOVE':
            if pen_is_dragging:
                if pen_edit_point_idx is not None and pen_edit_element is not None:
                    # Re-editing existing point/handle
                    pt = pen_points[pen_edit_point_idx]
                    
                    if pen_edit_element == 'ANCHOR':
                        # Move anchor and both handles together
                        dx = img_x - pt[0]
                        dy = img_y - pt[1]
                        pen_points[pen_edit_point_idx] = (
                            img_x, img_y,
                            pt[2] + dx, pt[3] + dy,
                            pt[4] + dx, pt[5] + dy
                        )
                    elif pen_edit_element == 'HANDLE_IN':
                        # Move handle in, optionally mirror handle out
                        if event.alt:
                            # Alt held: move only this handle
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], img_x, img_y, pt[4], pt[5])
                        else:
                            # Mirror the handle out
                            dx = img_x - pt[0]
                            dy = img_y - pt[1]
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], img_x, img_y, pt[0] - dx, pt[1] - dy)
                    elif pen_edit_element == 'HANDLE_OUT':
                        # Move handle out, optionally mirror handle in
                        if event.alt:
                            # Alt held: move only this handle
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], pt[2], pt[3], img_x, img_y)
                        else:
                            # Mirror the handle in
                            dx = img_x - pt[0]
                            dy = img_y - pt[1]
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], pt[0] - dx, pt[1] - dy, img_x, img_y)
                
                elif pen_drag_handle == 'OUT' and len(pen_points) > 0:
                    # Creating new point - adjust handle
                    last_pt = pen_points[-1]
                    dx = img_x - last_pt[0]
                    dy = img_y - last_pt[1]
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[0] - dx, last_pt[1] - dy, img_x, img_y)
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # First check if clicking near first point to close path
                if len(pen_points) >= 3:
                    first_pt = pen_points[0]
                    dist = ((img_x - first_pt[0])**2 + (img_y - first_pt[1])**2)**0.5
                    if dist < self._hit_radius:
                        # Close the path
                        self._is_closed = True
                        self._apply_path(context)
                        self._cleanup(context)
                        return {'FINISHED'}
                
                # Hit test for re-editing existing points/handles
                hit_idx, hit_elem = self._hit_test(img_x, img_y)
                
                if hit_idx is not None:
                    # Start editing existing point/handle
                    pen_edit_point_idx = hit_idx
                    pen_edit_element = hit_elem
                    pen_is_dragging = True
                    context.area.header_text_set(f"Editing {hit_elem.lower()} of point {hit_idx + 1} | Alt to break handles | Release to confirm")
                    return {'RUNNING_MODAL'}
                
                # Add new point
                pen_points.append((img_x, img_y, img_x, img_y, img_x, img_y))
                pen_is_dragging = True
                pen_drag_handle = 'OUT'
                pen_edit_point_idx = None
                pen_edit_element = None
                context.area.header_text_set(f"Point {len(pen_points)} | Drag to adjust curve | Enter to apply | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE':
                pen_is_dragging = False
                pen_drag_handle = None
                pen_edit_point_idx = None
                pen_edit_element = None
                return {'RUNNING_MODAL'}
        
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            # Delete last point
            if len(pen_points) > 0:
                pen_points.pop()
                context.area.header_text_set(f"Point deleted | {len(pen_points)} points remaining")
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(pen_points) >= 2:
                self._apply_path(context)
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging
        
        if context.area.type == 'IMAGE_EDITOR':
            pen_points = []
            pen_preview_pos = None
            pen_is_dragging = False
            
            self._image = context.space_data.image
            self._width, self._height = self._image.size
            self._is_closed = False
            
            from . import ui
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_pen_preview, (context,), 'WINDOW', 'POST_PIXEL')
            
            context.window_manager.modal_handler_add(self)
            context.area.header_text_set("Click to add points | Drag to adjust curves | Enter/Space to apply | ESC to cancel")
            return {'RUNNING_MODAL'}
        
        return {'CANCELLED'}
    
    def _apply_path(self, context):
        """Render the path to the image using numpy (no PIL required)."""
        global pen_points
        
        if len(pen_points) < 2:
            return
        
        props = context.scene.text_tool_properties
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(self._image)
        
        import numpy as np
        
        # Get current pixels
        num_pixels = self._width * self._height * 4
        pixels = np.zeros(num_pixels, dtype=np.float32)
        self._image.pixels.foreach_get(pixels)
        pixels = pixels.reshape((self._height, self._width, 4))
        
        # Generate bezier curve points (in image coordinates)
        path_points = self._generate_bezier_points()
        
        if len(path_points) < 2:
            return
        
        # Get blend mode and anti-aliasing setting
        brush = context.tool_settings.image_paint.brush
        blend_mode = brush.blend if brush else 'MIX'
        use_aa = props.use_antialiasing
        
        # Draw fill using scanline algorithm
        if props.pen_use_fill and len(path_points) >= 3:
            fill_color = np.array(props.pen_fill_color, dtype=np.float32)
            self._fill_polygon(pixels, path_points, fill_color, blend_mode)
        
        # Draw stroke
        if props.pen_use_stroke:
            stroke_color = np.array(props.pen_stroke_color, dtype=np.float32)
            stroke_width = props.pen_stroke_width
            self._draw_polyline(pixels, path_points, stroke_color, stroke_width, blend_mode, use_aa)
            
            # Close path if needed
            if self._is_closed and len(path_points) >= 2:
                self._draw_line(pixels, path_points[-1], path_points[0], stroke_color, stroke_width, blend_mode, use_aa)
        
        # Set pixels back
        self._image.pixels.foreach_set(pixels.flatten())
        self._image.update()
    
    def _generate_bezier_points(self, segments_per_curve=20):
        """Generate points along the bezier path in image coordinates."""
        global pen_points
        
        points = []
        for i in range(len(pen_points) - 1):
            p0 = pen_points[i]
            p1 = pen_points[i + 1]
            
            # Bezier control points
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]  # handle_out of p0
            x2, y2 = p1[2], p1[3]  # handle_in of p1
            x3, y3 = p1[0], p1[1]
            
            for t in range(segments_per_curve + 1):
                t_val = t / segments_per_curve
                # Cubic bezier formula
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                points.append((int(x), int(y)))
        
        return points
    
    def _fill_polygon(self, pixels, points, color, blend_mode='MIX'):
        """Fill a polygon using vectorized scanline algorithm with blend mode support."""
        import numpy as np
        
        if len(points) < 3:
            return
        
        # Convert to numpy arrays
        pts = np.array(points)
        min_y = max(0, int(pts[:, 1].min()))
        max_y = min(self._height - 1, int(pts[:, 1].max()))
        min_x = max(0, int(pts[:, 0].min()))
        max_x = min(self._width - 1, int(pts[:, 0].max()))
        
        if max_y <= min_y or max_x <= min_x:
            return
        
        # Create coordinate grid for the bounding box
        yy, xx = np.mgrid[min_y:max_y+1, min_x:max_x+1]
        
        # Point-in-polygon using ray casting (vectorized)
        n = len(points)
        inside = np.zeros(yy.shape, dtype=bool)
        
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            
            # Check if edge crosses the horizontal ray
            cond1 = ((y1 <= yy) & (yy < y2)) | ((y2 <= yy) & (yy < y1))
            
            if y1 != y2:
                x_intersect = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
                crosses = cond1 & (xx < x_intersect)
                inside ^= crosses
        
        # Apply fill with blend mode
        alpha = color[3]
        mask_3d = inside[:, :, np.newaxis]
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        
        # Calculate blended color based on blend mode
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        # Apply with alpha where mask is true
        pixels[min_y:max_y+1, min_x:max_x+1] = np.where(
            mask_3d,
            dst * (1 - alpha) + blended * alpha,
            dst
        )
    
    def _draw_polyline(self, pixels, points, color, width, blend_mode='MIX', use_aa=True):
        """Draw a polyline with given width, blend mode, and anti-aliasing support."""
        import numpy as np
        
        if len(points) < 2:
            return
        
        # Get bounding box of all points + stroke width padding
        pts = np.array(points)
        pad = width + 2
        min_x = max(0, int(pts[:, 0].min()) - pad)
        max_x = min(self._width - 1, int(pts[:, 0].max()) + pad)
        min_y = max(0, int(pts[:, 1].min()) - pad)
        max_y = min(self._height - 1, int(pts[:, 1].max()) + pad)
        
        if max_x <= min_x or max_y <= min_y:
            return
        
        # Create stroke accumulation mask for the bounding region
        stroke_height = max_y - min_y + 1
        stroke_width_px = max_x - min_x + 1
        stroke_mask = np.zeros((stroke_height, stroke_width_px), dtype=np.float32)
        
        # Pre-compute brush kernel
        half_w = width // 2 + 1
        by, bx = np.ogrid[-half_w:half_w+1, -half_w:half_w+1]
        dist = np.sqrt(bx**2 + by**2).astype(np.float32)
        
        # Create brush with optional anti-aliasing
        if use_aa:
            # Anti-aliased soft brush with smooth falloff
            brush = np.clip(1.0 - (dist - width/2.0 + 0.5) / 1.5, 0.0, 1.0)
        else:
            # Hard edge brush (no anti-aliasing)
            brush = (dist <= width/2.0).astype(np.float32)
        brush_h, brush_w = brush.shape
        
        # Collect all stroke centerline points using Bresenham
        for i in range(len(points) - 1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
            
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
            err = dx - dy
            
            while True:
                # Stamp brush onto stroke_mask (local coords)
                lx, ly = x1 - min_x, y1 - min_y
                
                # Calculate brush bounds clipped to mask
                msy = max(0, ly - half_w)
                mey = min(stroke_height, ly + half_w + 1)
                msx = max(0, lx - half_w)
                mex = min(stroke_width_px, lx + half_w + 1)
                
                # Corresponding brush region
                bsy = msy - (ly - half_w)
                bey = bsy + (mey - msy)
                bsx = msx - (lx - half_w)
                bex = bsx + (mex - msx)
                
                if mey > msy and mex > msx:
                    # Max blend (accumulate)
                    stroke_mask[msy:mey, msx:mex] = np.maximum(
                        stroke_mask[msy:mey, msx:mex],
                        brush[bsy:bey, bsx:bex]
                    )
                
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        
        # Apply accumulated stroke to pixels with blend mode
        alpha = color[3]
        final_alpha = stroke_mask * alpha
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        
        # Calculate blended color based on blend mode
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        # Apply with accumulated alpha
        for c in range(4):
            pixels[min_y:max_y+1, min_x:max_x+1, c] = (
                dst[:, :, c] * (1 - final_alpha) +
                blended[:, :, c] * final_alpha
            )
    
    def _draw_line(self, pixels, p1, p2, color, width, blend_mode='MIX', use_aa=True):
        """Draw a single line segment."""
        self._draw_polyline(pixels, [p1, p2], color, width, blend_mode, use_aa)
    
    def _apply_blend_mode(self, dst, src_color, blend_mode):
        """Apply blend mode to destination RGB with source RGB. Alpha is kept separate."""
        import numpy as np
        
        # Split RGB and calculate
        d_rgb = dst[..., :3]
        s_rgb = np.ones_like(d_rgb) * src_color[:3]
        
        out_rgb = s_rgb.copy()
        
        if blend_mode == 'MIX':
            out_rgb = s_rgb
        elif blend_mode == 'DARKEN':
            out_rgb = np.minimum(d_rgb, s_rgb)
        elif blend_mode == 'MUL':
            out_rgb = d_rgb * s_rgb
        elif blend_mode == 'LIGHTEN':
            out_rgb = np.maximum(d_rgb, s_rgb)
        elif blend_mode == 'SCREEN':
            out_rgb = 1.0 - (1.0 - d_rgb) * (1.0 - s_rgb)
        elif blend_mode == 'ADD':
            out_rgb = np.clip(d_rgb + s_rgb, 0.0, 1.0)
        elif blend_mode == 'SUB':
            out_rgb = np.clip(d_rgb - s_rgb, 0.0, 1.0)
        elif blend_mode == 'OVERLAY':
            # Hard light if src < 0.5? No, Overlay is:
            # if dst < 0.5: 2 * dst * src
            # else: 1 - 2 * (1 - dst) * (1 - src)
            mask = d_rgb < 0.5
            out_rgb = np.where(mask, 
                               2.0 * d_rgb * s_rgb, 
                               1.0 - 2.0 * (1.0 - d_rgb) * (1.0 - s_rgb))
        elif blend_mode == 'DIFFERENCE':
            out_rgb = np.abs(d_rgb - s_rgb)
        elif blend_mode == 'EXCLUSION':
            out_rgb = d_rgb + s_rgb - 2.0 * d_rgb * s_rgb
        elif blend_mode == 'SOFT_LIGHT':
            # Pegtop formula
            out_rgb = (1.0 - 2.0 * s_rgb) * (d_rgb ** 2) + 2.0 * s_rgb * d_rgb
        elif blend_mode == 'HARD_LIGHT':
            # Overlay with swapped inputs
            mask = s_rgb < 0.5
            out_rgb = np.where(mask,
                               2.0 * s_rgb * d_rgb,
                               1.0 - 2.0 * (1.0 - s_rgb) * (1.0 - d_rgb))
        elif blend_mode == 'LINEAR_LIGHT':
            out_rgb = np.clip(d_rgb + 2.0 * s_rgb - 1.0, 0.0, 1.0)
        elif blend_mode == 'VIVID_LIGHT':
            # Color Burn / Color Dodge split
            out_rgb = np.where(s_rgb < 0.5,
                               1.0 - np.clip((1.0 - d_rgb) / (2.0 * s_rgb + 0.001), 0.0, 1.0),
                               np.clip(d_rgb / (2.0 * (1.0 - s_rgb) + 0.001), 0.0, 1.0))
        elif blend_mode == 'PIN_LIGHT':
            # Lighten / Darken split
            out_rgb = np.where(s_rgb < 0.5,
                               np.minimum(d_rgb, 2.0 * s_rgb),
                               np.maximum(d_rgb, 2.0 * s_rgb - 1.0))
        elif blend_mode == 'DIVIDE':
             out_rgb = np.clip(d_rgb / (s_rgb + 0.001), 0.0, 1.0)
             
        # Re-attach alpha channel (set to 1.0 so alpha compositing logic works correctly)
        # Using 1.0 ensures that 'src_alpha' controls the mix factor, not the color value itself
        alpha = np.ones(dst.shape[:2] + (1,), dtype=np.float32)
        
        return np.dstack((out_rgb, alpha))
    
    def _cleanup(self, context):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_edit_point_idx, pen_edit_element
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        pen_edit_point_idx = None
        pen_edit_element = None
        
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
        context.area.tag_redraw()


# ----------------------------
# Clone Tool Size/Strength Adjust Operators
# ----------------------------
class IMAGE_PAINT_OT_clone_adjust_size(Operator):
    bl_idname = "image_paint.clone_adjust_size"
    bl_label = "Adjust Clone Brush Size"
    bl_description = "Interactively adjust clone brush size"
    bl_options = {'REGISTER'}
    
    _start_pos = 0
    _start_value = 0
    
    @classmethod
    def poll(cls, context):
        return context.area.type == 'IMAGE_EDITOR'
    
    def modal(self, context, event):
        props = context.scene.text_tool_properties
        mx = event.mouse_region_x
        
        if event.type == 'MOUSEMOVE':
            delta = (mx - self._start_pos) * 0.5
            new_size = int(max(1, min(500, self._start_value + delta)))
            props.clone_brush_size = new_size
            context.area.header_text_set(f"Clone Brush Size: {new_size} | Click to confirm | ESC to cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.clone_brush_size = self._start_value
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._start_pos = event.mouse_region_x
        self._start_value = props.clone_brush_size
        context.area.header_text_set(f"Clone Brush Size: {self._start_value} | Move mouse to adjust | Click to confirm")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class IMAGE_PAINT_OT_clone_adjust_strength(Operator):
    bl_idname = "image_paint.clone_adjust_strength"
    bl_label = "Adjust Clone Brush Strength"
    bl_description = "Interactively adjust clone brush strength"
    bl_options = {'REGISTER'}
    
    _start_pos = 0
    _start_value = 0.0
    
    @classmethod
    def poll(cls, context):
        return context.area.type == 'IMAGE_EDITOR'
    
    def modal(self, context, event):
        props = context.scene.text_tool_properties
        mx = event.mouse_region_x
        
        if event.type == 'MOUSEMOVE':
            delta = (mx - self._start_pos) * 0.005
            new_strength = max(0.0, min(1.0, self._start_value + delta))
            props.clone_brush_strength = new_strength
            context.area.header_text_set(f"Clone Brush Strength: {new_strength:.2f} | Click to confirm | ESC to cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.clone_brush_strength = self._start_value
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._start_pos = event.mouse_region_x
        self._start_value = props.clone_brush_strength
        context.area.header_text_set(f"Clone Brush Strength: {self._start_value:.2f} | Move mouse to adjust | Click to confirm")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ----------------------------
# Clone Tool (Image Editor)
# ----------------------------
class IMAGE_PAINT_OT_clone_tool(Operator):
    bl_idname = "image_paint.clone_tool"
    bl_label = "Image Clone Tool"
    bl_description = "Clone pixels from source to destination (Ctrl+Click to set source)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _is_painting = False
    _source_offset = (0, 0)  # Offset from cursor to source in image pixels
    _source_uv = (0, 0)  # Source position in UV space
    _last_paint_pos = None
    _original_pixels = None
    _width = 0
    _height = 0
    
    # Interactive adjustment state
    _adjust_mode = None  # 'SIZE' or 'STRENGTH'
    _adjust_start_pos = None
    _adjust_start_value = 0.0
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        utils.clone_cursor_pos = (mx, my)
        
        # Handle adjustment mode FIRST (F/Shift+F size/strength adjustment)
        if self._adjust_mode:
            props = context.scene.text_tool_properties
            
            if event.type == 'MOUSEMOVE':
                delta = (mx - self._adjust_start_pos) * 0.5
                
                if self._adjust_mode == 'SIZE':
                    new_size = int(max(1, min(500, self._adjust_start_value + delta)))
                    props.clone_brush_size = new_size
                    context.area.header_text_set(f"Size: {new_size} | Click to confirm | ESC to cancel")
                elif self._adjust_mode == 'STRENGTH':
                    new_strength = max(0.0, min(1.0, self._adjust_start_value + delta * 0.01))
                    props.clone_brush_strength = new_strength
                    context.area.header_text_set(f"Strength: {new_strength:.2f} | Click to confirm | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                # Confirm adjustment
                self._adjust_mode = None
                self._adjust_start_pos = None
                if utils.clone_source_set:
                    context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength")
                else:
                    context.area.header_text_set("Ctrl+Click to set source | F for size | Shift+F for strength")
                return {'RUNNING_MODAL'}
            
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment - restore original value
                if self._adjust_mode == 'SIZE':
                    props.clone_brush_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'STRENGTH':
                    props.clone_brush_strength = self._adjust_start_value
                self._adjust_mode = None
                self._adjust_start_pos = None
                if utils.clone_source_set:
                    context.area.header_text_set("Source set - Click to paint")
                else:
                    context.area.header_text_set("Ctrl+Click to set source first")
                return {'RUNNING_MODAL'}
            
            return {'RUNNING_MODAL'}
        
        # Handle F key for size adjustment, Shift+F for strength
        if event.type == 'F' and event.value == 'PRESS':
            props = context.scene.text_tool_properties
            if event.shift:
                # Shift+F: Adjust strength
                self._adjust_mode = 'STRENGTH'
                self._adjust_start_pos = mx
                self._adjust_start_value = props.clone_brush_strength
                context.area.header_text_set(f"Strength: {props.clone_brush_strength:.2f} | Move mouse to adjust | Click to confirm")
            else:
                # F: Adjust size
                self._adjust_mode = 'SIZE'
                self._adjust_start_pos = mx
                self._adjust_start_value = props.clone_brush_size
                context.area.header_text_set(f"Size: {props.clone_brush_size} | Move mouse to adjust | Click to confirm")
            return {'RUNNING_MODAL'}
        
        # Update source crosshair position if painting
        if self._is_painting and utils.clone_source_set:
            region = context.region
            view2d = region.view2d
            uv_cursor = view2d.region_to_view(mx, my)
            source_uv_x = uv_cursor[0] + self._source_offset[0]
            source_uv_y = uv_cursor[1] + self._source_offset[1]
            source_screen = view2d.view_to_region(source_uv_x, source_uv_y, clip=False)
            utils.clone_source_pos = source_screen
        
        if event.type == 'MOUSEMOVE':
            if self._is_painting and utils.clone_source_set:
                self._paint_clone(context, mx, my)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if event.ctrl:
                    # Set source point
                    utils.clone_source_pos = (mx, my)
                    utils.clone_source_set = True
                    
                    # Store source position in UV space for offset calculation
                    region = context.region
                    view2d = region.view2d
                    self._source_uv = view2d.region_to_view(mx, my)
                    context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength")
                    return {'RUNNING_MODAL'}
                else:
                    if utils.clone_source_set:
                        # Start painting
                        self._is_painting = True
                        self._last_paint_pos = None
                        
                        # Calculate offset in UV space
                        region = context.region
                        view2d = region.view2d
                        cursor_uv = view2d.region_to_view(mx, my)
                        self._source_offset = (
                            self._source_uv[0] - cursor_uv[0],
                            self._source_uv[1] - cursor_uv[1]
                        )
                        
                        # Save undo state
                        utils.ImageUndoStack.get().push_state(self._image)
                        
                        # Paint first dab
                        self._paint_clone(context, mx, my)
                        context.area.header_text_set("Painting... Release to finish")
                    else:
                        context.area.header_text_set("Ctrl+Click to set source first | F for size | Shift+F for strength")
                    return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_painting:
                self._is_painting = False
                self._last_paint_pos = None
                context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength | ESC to exit")
                return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_clone_preview_image, (), 'WINDOW', 'POST_PIXEL')
            
            self._image = context.space_data.image
            self._width, self._height = self._image.size
            
            # Read pixels once
            num_pixels = self._width * self._height * 4
            self._original_pixels = array.array('f', [0.0] * num_pixels)
            self._image.pixels.foreach_get(self._original_pixels)
            
            mx, my = event.mouse_region_x, event.mouse_region_y
            utils.clone_cursor_pos = (mx, my)
            
            # If invoked with Ctrl held (from keymap), auto-set source
            if event.ctrl:
                utils.clone_source_pos = (mx, my)
                utils.clone_source_set = True
                region = context.region
                view2d = region.view2d
                self._source_uv = view2d.region_to_view(mx, my)
                context.area.header_text_set("Source set! Click to paint | Ctrl+Click for new source | ESC to exit")
            elif utils.clone_source_set:
                context.area.header_text_set("Click to paint | Ctrl+Click to set new source | ESC to exit")
            else:
                context.area.header_text_set("Ctrl+Click to set source")
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def _cleanup(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.clone_cursor_pos = None
        self._image = None
        self._original_pixels = None
        context.area.tag_redraw()
    
    def _paint_clone(self, context, mx, my):
        """Paint cloned pixels at the cursor position - OPTIMIZED with NumPy."""
        if not self._image or not utils.clone_source_set:
            return
        
        props = context.scene.text_tool_properties
        brush_size = props.clone_brush_size
        falloff_preset = props.clone_falloff_preset
        strength = props.clone_brush_strength
        
        region = context.region
        view2d = region.view2d
        
        # Convert cursor to image coordinates
        cursor_uv = view2d.region_to_view(mx, my)
        cursor_px_x = int(cursor_uv[0] * self._width)
        cursor_px_y = int(cursor_uv[1] * self._height)
        
        # Source position in image coordinates
        source_uv = (cursor_uv[0] + self._source_offset[0], cursor_uv[1] + self._source_offset[1])
        source_px_x = int(source_uv[0] * self._width)
        source_px_y = int(source_uv[1] * self._height)
        
        try:
            import numpy as np
            
            # Get pixels as numpy array
            num_pixels = self._width * self._height * 4
            pixels = np.zeros(num_pixels, dtype=np.float32)
            self._image.pixels.foreach_get(pixels)
            pixels = pixels.reshape((self._height, self._width, 4))
            
            # Calculate brush region bounds
            x_min = max(0, cursor_px_x - brush_size)
            x_max = min(self._width, cursor_px_x + brush_size + 1)
            y_min = max(0, cursor_px_y - brush_size)
            y_max = min(self._height, cursor_px_y + brush_size + 1)
            
            # Source region bounds
            src_x_min = source_px_x - cursor_px_x + x_min
            src_x_max = source_px_x - cursor_px_x + x_max
            src_y_min = source_px_y - cursor_px_y + y_min
            src_y_max = source_px_y - cursor_px_y + y_max
            
            # Clip source bounds
            if src_x_min < 0:
                x_min -= src_x_min
                src_x_min = 0
            if src_y_min < 0:
                y_min -= src_y_min
                src_y_min = 0
            if src_x_max > self._width:
                x_max -= (src_x_max - self._width)
                src_x_max = self._width
            if src_y_max > self._height:
                y_max -= (src_y_max - self._height)
                src_y_max = self._height
            
            if x_max <= x_min or y_max <= y_min:
                return
            
            # Create coordinate grids for the brush region
            y_coords = np.arange(y_min, y_max)
            x_coords = np.arange(x_min, x_max)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Calculate distance from cursor center (normalized 0-1)
            dist = np.sqrt((xx - cursor_px_x)**2 + (yy - cursor_px_y)**2)
            
            # Create brush mask
            mask = dist <= brush_size
            
            # Normalize distance to 0-1 range
            t = np.clip(dist / brush_size, 0.0, 1.0) if brush_size > 0 else np.zeros_like(dist)
            
            # Calculate falloff based on preset
            if falloff_preset == 'SMOOTH':
                # Smooth: 3t² - 2t³ (smoothstep)
                falloff = 1.0 - (3.0 * t**2 - 2.0 * t**3)
            elif falloff_preset == 'SMOOTHER':
                # Smoother: 6t⁵ - 15t⁴ + 10t³ (smootherstep)
                falloff = 1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)
            elif falloff_preset == 'SPHERE':
                # Sphere: sqrt(1 - t²)
                falloff = np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))
            elif falloff_preset == 'ROOT':
                # Root: 1 - sqrt(t)
                falloff = 1.0 - np.sqrt(t)
            elif falloff_preset == 'SHARP':
                # Sharp: (1 - t)²
                falloff = (1.0 - t)**2
            elif falloff_preset == 'LINEAR':
                # Linear: 1 - t
                falloff = 1.0 - t
            elif falloff_preset == 'CONSTANT':
                # Constant: no falloff
                falloff = np.ones_like(t)
            elif falloff_preset == 'CUSTOM':
                # Custom: sample from brush curve
                brush = context.tool_settings.image_paint.brush
                if brush and brush.curve_distance_falloff:
                    curve = brush.curve_distance_falloff
                    # Sample curve for each distance value (vectorized approach)
                    falloff = np.zeros_like(t)
                    for i in range(t.shape[0]):
                        for j in range(t.shape[1]):
                            falloff[i, j] = curve.evaluate(curve.curves[0], t[i, j])
                else:
                    # Fallback to smooth if no brush curve
                    falloff = 1.0 - (3.0 * t**2 - 2.0 * t**3)
            else:
                falloff = 1.0 - t  # Default to linear
            
            # Apply strength
            falloff = falloff * mask * strength
            falloff = falloff[:, :, np.newaxis]  # Expand for RGBA
            
            # Extract source and destination regions
            dst_region = pixels[y_min:y_max, x_min:x_max]
            src_region = pixels[src_y_min:src_y_max, src_x_min:src_x_max]
            
            # Ensure regions match
            min_h = min(dst_region.shape[0], src_region.shape[0], falloff.shape[0])
            min_w = min(dst_region.shape[1], src_region.shape[1], falloff.shape[1])
            
            if min_h > 0 and min_w > 0:
                # Get blend mode from brush
                brush = context.tool_settings.image_paint.brush
                blend_mode = brush.blend if brush else 'MIX'
                
                dst = dst_region[:min_h, :min_w]
                src = src_region[:min_h, :min_w]
                fall = falloff[:min_h, :min_w]
                
                # Apply blend mode
                if blend_mode == 'MIX':
                    blended = src
                elif blend_mode == 'DARKEN':
                    blended = np.minimum(dst, src)
                elif blend_mode == 'MUL':
                    blended = dst * src
                elif blend_mode == 'LIGHTEN':
                    blended = np.maximum(dst, src)
                elif blend_mode == 'SCREEN':
                    blended = 1.0 - (1.0 - dst) * (1.0 - src)
                elif blend_mode == 'ADD':
                    blended = np.clip(dst + src, 0.0, 1.0)
                elif blend_mode == 'SUB':
                    blended = np.clip(dst - src, 0.0, 1.0)
                elif blend_mode == 'OVERLAY':
                    # Overlay: combination of multiply and screen
                    blended = np.where(dst < 0.5,
                                       2.0 * dst * src,
                                       1.0 - 2.0 * (1.0 - dst) * (1.0 - src))
                elif blend_mode == 'DIFFERENCE':
                    blended = np.abs(dst - src)
                elif blend_mode == 'DIVIDE':
                    blended = np.clip(dst / (src + 0.001), 0.0, 1.0)
                elif blend_mode == 'ERASE_ALPHA':
                    blended = dst.copy()
                    blended[:, :, 3] = dst[:, :, 3] * (1.0 - src[:, :, 3])
                elif blend_mode == 'ADD_ALPHA':
                    blended = dst.copy()
                    blended[:, :, 3] = np.clip(dst[:, :, 3] + src[:, :, 3], 0.0, 1.0)
                else:
                    # Default to mix
                    blended = src
                
                # Apply with falloff
                dst_region[:min_h, :min_w] = dst + (blended - dst) * fall
            
            # Set pixels back
            self._image.pixels.foreach_set(pixels.flatten())
            self._image.update()
            
        except ImportError:
            # Fallback to simple Python (slower)
            num_pixels = self._width * self._height * 4
            pixels = array.array('f', [0.0] * num_pixels)
            self._image.pixels.foreach_get(pixels)
            
            for dy in range(-brush_size, brush_size + 1):
                for dx in range(-brush_size, brush_size + 1):
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > brush_size:
                        continue
                    
                    falloff = 1.0 - (dist / brush_size) if brush_size > 0 else 1.0
                    falloff = max(0.0, min(1.0, falloff / (1.0 - hardness + 0.001)))
                    
                    dst_x, dst_y = cursor_px_x + dx, cursor_px_y + dy
                    src_x, src_y = source_px_x + dx, source_px_y + dy
                    
                    if not (0 <= dst_x < self._width and 0 <= dst_y < self._height):
                        continue
                    if not (0 <= src_x < self._width and 0 <= src_y < self._height):
                        continue
                    
                    src_idx = (src_y * self._width + src_x) * 4
                    dst_idx = (dst_y * self._width + dst_x) * 4
                    
                    for c in range(4):
                        pixels[dst_idx + c] += (pixels[src_idx + c] - pixels[dst_idx + c]) * falloff
            
            self._image.pixels.foreach_set(pixels)
            self._image.update()





# ----------------------------
# Modal Adjust Operators (Font Size / Rotation)
# ----------------------------

def _draw_adjust_preview_3d():
    """Draw text preview during adjust operations in 3D view."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return
    
    props = context.scene.text_tool_properties
    if not props.text:
        return
    
    # Draw at center of region
    region = context.region
    x = region.width // 2
    y = region.height // 2
    
    # Draw using blf
    import blf
    font_path = props.font_file.filepath if props.font_file else None
    font_id = utils._get_blf_font_id(font_path)
    
    font_size = max(8, min(props.font_size, 500))
    blf.size(font_id, font_size)
    
    # Get text dimensions
    text_width, text_height = blf.dimensions(font_id, props.text)
    
    # Set color
    if props.use_gradient:
        r, g, b, a = 1.0, 1.0, 1.0, 0.8
    else:
        r, g, b = props.color[0], props.color[1], props.color[2]
        a = props.color[3] if len(props.color) > 3 else 0.8
    blf.color(font_id, r, g, b, a)
    
    # Apply rotation
    if props.rotation != 0.0:
        blf.enable(font_id, blf.ROTATION)
        blf.rotation(font_id, props.rotation)
        
        # Center the rotated text
        cos_r = math.cos(props.rotation)
        sin_r = math.sin(props.rotation)
        offset_x = -text_width / 2
        offset_y = -text_height / 2
        rotated_offset_x = offset_x * cos_r - offset_y * sin_r
        rotated_offset_y = offset_x * sin_r + offset_y * cos_r
        blf.position(font_id, x + rotated_offset_x, y + rotated_offset_y, 0)
    else:
        blf.position(font_id, x - text_width / 2, y - text_height / 2, 0)
    
    blf.draw(font_id, props.text)
    
    if props.rotation != 0.0:
        blf.disable(font_id, blf.ROTATION)


def _draw_adjust_preview_image():
    """Draw text preview during adjust operations in Image Editor."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return
    
    props = context.scene.text_tool_properties
    if not props.text:
        return
    
    # Draw at center of region
    region = context.region
    x = region.width // 2
    y = region.height // 2
    
    # Draw using blf
    import blf
    font_path = props.font_file.filepath if props.font_file else None
    font_id = utils._get_blf_font_id(font_path)
    
    # Scale font size based on zoom
    font_size = props.font_size
    try:
        sima = context.space_data
        if sima.type == 'IMAGE_EDITOR' and sima.image:
            i_width, i_height = sima.image.size
            if i_width > 0:
                cx = region.width / 2
                cy = region.height / 2
                v0 = region.view2d.region_to_view(cx, cy)
                v1 = region.view2d.region_to_view(cx + 100, cy)
                dist_view_x = v1[0] - v0[0]
                if abs(dist_view_x) > 0.000001:
                    img_subset_pixels = dist_view_x * i_width
                    scale = 100.0 / img_subset_pixels
                    font_size = int(props.font_size * scale)
    except Exception:
        pass
    
    font_size = max(8, min(font_size, 500))
    blf.size(font_id, font_size)
    
    # Get text dimensions
    text_width, text_height = blf.dimensions(font_id, props.text)
    
    # Set color
    if props.use_gradient:
        r, g, b, a = 1.0, 1.0, 1.0, 0.8
    else:
        r, g, b = props.color[0], props.color[1], props.color[2]
        a = props.color[3] if len(props.color) > 3 else 0.8
    blf.color(font_id, r, g, b, a)
    
    # Apply rotation
    if props.rotation != 0.0:
        blf.enable(font_id, blf.ROTATION)
        blf.rotation(font_id, props.rotation)
        
        # Center the rotated text
        cos_r = math.cos(props.rotation)
        sin_r = math.sin(props.rotation)
        offset_x = -text_width / 2
        offset_y = -text_height / 2
        rotated_offset_x = offset_x * cos_r - offset_y * sin_r
        rotated_offset_y = offset_x * sin_r + offset_y * cos_r
        blf.position(font_id, x + rotated_offset_x, y + rotated_offset_y, 0)
    else:
        blf.position(font_id, x - text_width / 2, y - text_height / 2, 0)
    
    blf.draw(font_id, props.text)
    
    if props.rotation != 0.0:
        blf.disable(font_id, blf.ROTATION)


class TEXTTOOL_OT_adjust_font_size(Operator):
    """Adjust font size by dragging left/right"""
    bl_idname = "texttool.adjust_font_size"
    bl_label = "Adjust Font Size"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}

    _initial_size: int = 64
    _initial_mouse_x: int = 0
    _sensitivity: float = 0.5  # pixels per mouse pixel
    _draw_handler = None
    _handler_space = None

    @classmethod
    def poll(cls, context):
        # Only active when our Text Tool is selected
        if not (context.mode in {'PAINT_TEXTURE', 'PAINT'} or 
                (context.area and context.area.type == 'IMAGE_EDITOR')):
            return False
        
        # Check if our tool is active
        ws = context.workspace
        if not ws:
            return False
        
        try:
            if context.area and context.area.type == 'VIEW_3D':
                tool = ws.tools.from_space_view3d_mode('PAINT_TEXTURE')
                return tool.idname == 'texture_paint.text_tool_ttf'
            elif context.area and context.area.type == 'IMAGE_EDITOR':
                tool = ws.tools.from_space_image_mode('PAINT')
                return tool.idname == 'image_paint.text_tool_ttf'
        except (AttributeError, KeyError):
            pass
        return False

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._initial_size = props.font_size
        self._initial_mouse_x = event.mouse_x
        
        # Add draw handler for preview
        if context.area.type == 'VIEW_3D':
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _draw_adjust_preview_3d, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'VIEW_3D'
        elif context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                _draw_adjust_preview_image, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'IMAGE_EDITOR'
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
        return {'RUNNING_MODAL'}

    def _remove_handler(self):
        if self._draw_handler:
            if self._handler_space == 'VIEW_3D':
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            elif self._handler_space == 'IMAGE_EDITOR':
                bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

    def modal(self, context, event):
        props = context.scene.text_tool_properties
        
        if event.type == 'MOUSEMOVE':
            delta = event.mouse_x - self._initial_mouse_x
            new_size = int(self._initial_size + delta * self._sensitivity)
            new_size = max(8, min(512, new_size))  # Clamp to valid range
            props.font_size = new_size
            context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, f"Font Size: {props.font_size}")
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.font_size = self._initial_size
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, "Font size change cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}


class TEXTTOOL_OT_adjust_rotation(Operator):
    """Adjust text rotation by dragging left/right"""
    bl_idname = "texttool.adjust_rotation"
    bl_label = "Adjust Rotation"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}

    _initial_rotation: float = 0.0
    _initial_mouse_x: int = 0
    _sensitivity: float = 0.01  # radians per mouse pixel
    _draw_handler = None
    _handler_space = None

    @classmethod
    def poll(cls, context):
        # Only active when our Text Tool is selected
        if not (context.mode in {'PAINT_TEXTURE', 'PAINT'} or 
                (context.area and context.area.type == 'IMAGE_EDITOR')):
            return False
        
        # Check if our tool is active
        ws = context.workspace
        if not ws:
            return False
        
        try:
            if context.area and context.area.type == 'VIEW_3D':
                tool = ws.tools.from_space_view3d_mode('PAINT_TEXTURE')
                return tool.idname == 'texture_paint.text_tool_ttf'
            elif context.area and context.area.type == 'IMAGE_EDITOR':
                tool = ws.tools.from_space_image_mode('PAINT')
                return tool.idname == 'image_paint.text_tool_ttf'
        except (AttributeError, KeyError):
            pass
        return False

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._initial_rotation = props.rotation
        self._initial_mouse_x = event.mouse_x
        
        # Add draw handler for preview
        if context.area.type == 'VIEW_3D':
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _draw_adjust_preview_3d, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'VIEW_3D'
        elif context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                _draw_adjust_preview_image, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'IMAGE_EDITOR'
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
        return {'RUNNING_MODAL'}

    def _remove_handler(self):
        if self._draw_handler:
            if self._handler_space == 'VIEW_3D':
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            elif self._handler_space == 'IMAGE_EDITOR':
                bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

    def modal(self, context, event):
        props = context.scene.text_tool_properties
        
        if event.type == 'MOUSEMOVE':
            delta = event.mouse_x - self._initial_mouse_x
            new_rotation = self._initial_rotation + delta * self._sensitivity
            # Wrap to 0-360 degrees (0 to 2*pi radians)
            new_rotation = new_rotation % (2 * math.pi)
            if new_rotation < 0:
                new_rotation += 2 * math.pi
            props.rotation = new_rotation
            context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, f"Rotation: {math.degrees(props.rotation):.1f}°")
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.rotation = self._initial_rotation
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, "Rotation change cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}


class TEXTURE_PAINT_OT_refresh_fonts(Operator):
    bl_idname = "paint.refresh_fonts_ttf"
    bl_label = "Refresh Fonts"
    bl_description = "Scan custom directories and load fonts into Blender"
    bl_options = {'REGISTER'}

    def execute(self, context):
        stats = utils.load_custom_fonts_to_blender()
        self.report({'INFO'}, f"Loaded {stats['loaded']} new fonts from custom paths")
        return {'FINISHED'}


# ----------------------------
# Crop Tool (Image Editor)
# ----------------------------
class IMAGE_PAINT_OT_crop_tool(Operator):
    bl_idname = "image_paint.crop_tool"
    bl_label = "Image Crop Tool"
    bl_description = "Crop the image by selecting a rectangular region (Enter/Space to confirm)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None  # Top-left corner (min x, min y in screen coords after normalization)
    _end_pos = None    # Bottom-right corner
    _is_dragging = False
    _selection_complete = False
    _image = None
    
    # Resize handle state
    _resize_mode = None  # None, 'TL', 'TR', 'BL', 'BR', 'T', 'B', 'L', 'R', 'MOVE'
    _drag_offset = (0, 0)  # For move mode
    HANDLE_SIZE = 12  # Pixels for handle hit detection
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _get_normalized_rect(self):
        """Return (x1, y1, x2, y2) with x1 < x2 and y1 < y2."""
        if not self._start_pos or not self._end_pos:
            return None
        x1, y1 = self._start_pos
        x2, y2 = self._end_pos
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    def _hit_test_handle(self, mx, my):
        """Check if mouse is over a resize handle. Returns handle name or None."""
        rect = self._get_normalized_rect()
        if not rect:
            return None
        
        x1, y1, x2, y2 = rect
        hs = self.HANDLE_SIZE
        
        # Corner handles (priority)
        if abs(mx - x1) < hs and abs(my - y1) < hs:
            return 'BL'  # Bottom-left
        if abs(mx - x2) < hs and abs(my - y1) < hs:
            return 'BR'  # Bottom-right
        if abs(mx - x1) < hs and abs(my - y2) < hs:
            return 'TL'  # Top-left
        if abs(mx - x2) < hs and abs(my - y2) < hs:
            return 'TR'  # Top-right
        
        # Edge handles
        if abs(mx - x1) < hs and y1 < my < y2:
            return 'L'  # Left edge
        if abs(mx - x2) < hs and y1 < my < y2:
            return 'R'  # Right edge
        if abs(my - y1) < hs and x1 < mx < x2:
            return 'B'  # Bottom edge
        if abs(my - y2) < hs and x1 < mx < x2:
            return 'T'  # Top edge
        
        # Inside rectangle = move
        if x1 < mx < x2 and y1 < my < y2:
            return 'MOVE'
        
        return None
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        # Confirm crop with Enter or Spacebar
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if self._selection_complete and self._start_pos and self._end_pos:
                success = self._apply_crop(context)
                self._cleanup(context)
                if success:
                    self.report({'INFO'}, "Image cropped successfully")
                    return {'FINISHED'}
                else:
                    self.report({'WARNING'}, "Crop cancelled - selection too small")
                    return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging:
                props = context.scene.text_tool_properties
                
                if self._resize_mode == 'MOVE':
                    # Move entire selection
                    dx = mx - self._drag_offset[0]
                    dy = my - self._drag_offset[1]
                    rect = self._get_normalized_rect()
                    if rect:
                        x1, y1, x2, y2 = rect
                        w, h = x2 - x1, y2 - y1
                        self._start_pos = (dx, dy)
                        self._end_pos = (dx + w, dy + h)
                elif self._resize_mode in ('TL', 'TR', 'BL', 'BR', 'T', 'B', 'L', 'R'):
                    # Resize from handle
                    rect = self._get_normalized_rect()
                    if rect:
                        x1, y1, x2, y2 = rect
                        
                        if 'L' in self._resize_mode:
                            x1 = mx
                        if 'R' in self._resize_mode:
                            x2 = mx
                        if 'T' in self._resize_mode:
                            y2 = my
                        if 'B' in self._resize_mode:
                            y1 = my
                        
                        # Apply aspect ratio constraint if locked
                        if props.crop_lock_aspect:
                            aspect = props.crop_aspect_width / props.crop_aspect_height
                            w, h = x2 - x1, y2 - y1
                            if abs(w) > 0 and abs(h) > 0:
                                current_aspect = abs(w) / abs(h)
                                if current_aspect > aspect:
                                    # Too wide, adjust width
                                    new_w = abs(h) * aspect
                                    if 'L' in self._resize_mode:
                                        x1 = x2 - new_w
                                    else:
                                        x2 = x1 + new_w
                                else:
                                    # Too tall, adjust height
                                    new_h = abs(w) / aspect
                                    if 'B' in self._resize_mode:
                                        y1 = y2 - new_h
                                    else:
                                        y2 = y1 + new_h
                        
                        self._start_pos = (x1, y1)
                        self._end_pos = (x2, y2)
                else:
                    # Initial drag - new selection
                    new_x, new_y = mx, my
                    
                    # Apply aspect ratio constraint if locked
                    if props.crop_lock_aspect:
                        aspect = props.crop_aspect_width / props.crop_aspect_height
                        sx, sy = self._start_pos
                        w = new_x - sx
                        h = new_y - sy
                        if abs(w) > 0 and abs(h) > 0:
                            current_aspect = abs(w) / abs(h)
                            if current_aspect > aspect:
                                # Too wide, adjust width
                                new_w = abs(h) * aspect * (1 if w > 0 else -1)
                                new_x = sx + new_w
                            else:
                                # Too tall, adjust height
                                new_h = abs(w) / aspect * (1 if h > 0 else -1)
                                new_y = sy + new_h
                    
                    self._end_pos = (new_x, new_y)
                
                utils.crop_preview_start = self._start_pos
                utils.crop_preview_end = self._end_pos
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if self._selection_complete:
                    # Check if clicking on a handle to resize
                    handle = self._hit_test_handle(mx, my)
                    if handle:
                        self._is_dragging = True
                        self._resize_mode = handle
                        if handle == 'MOVE':
                            rect = self._get_normalized_rect()
                            if rect:
                                self._drag_offset = (mx - rect[0], my - rect[1])
                        context.area.header_text_set(f"Resizing: {handle} | Release to confirm adjustment")
                        return {'RUNNING_MODAL'}
                
                # Start new selection
                self._is_dragging = True
                self._selection_complete = False
                self._resize_mode = None
                self._start_pos = (mx, my)
                self._end_pos = self._start_pos
                self._image = context.space_data.image
                utils.crop_preview_start = self._start_pos
                utils.crop_preview_end = self._end_pos
                context.area.header_text_set("Drag to select crop region")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                props = context.scene.text_tool_properties
                
                # Check for single-click (minimal movement)
                if self._start_pos:
                    dx = abs(mx - self._start_pos[0])
                    dy = abs(my - self._start_pos[1])
                    is_single_click = (dx < 5 and dy < 5)
                else:
                    is_single_click = False
                
                # If single-click and resolution mode, create fixed-size region
                if is_single_click and props.crop_use_resolution and self._image:
                    # Get image size and view2d for coordinate conversion
                    img_width, img_height = self._image.size
                    view2d = context.region.view2d
                    
                    # Calculate region size in screen space based on resolution
                    # Convert resolution pixels to screen pixels using view2d scale
                    uv_center = view2d.region_to_view(mx, my)
                    
                    # Calculate half-sizes in UV space
                    half_w_uv = (props.crop_resolution_x / img_width) / 2
                    half_h_uv = (props.crop_resolution_y / img_height) / 2
                    
                    # Convert back to screen coordinates
                    uv_x1, uv_y1 = uv_center[0] - half_w_uv, uv_center[1] - half_h_uv
                    uv_x2, uv_y2 = uv_center[0] + half_w_uv, uv_center[1] + half_h_uv
                    
                    screen_x1, screen_y1 = view2d.view_to_region(uv_x1, uv_y1)
                    screen_x2, screen_y2 = view2d.view_to_region(uv_x2, uv_y2)
                    
                    self._start_pos = (screen_x1, screen_y1)
                    self._end_pos = (screen_x2, screen_y2)
                    utils.crop_preview_start = self._start_pos
                    utils.crop_preview_end = self._end_pos
                else:
                    self._end_pos = (mx, my) if not self._resize_mode else self._end_pos
                    utils.crop_preview_end = self._end_pos
                
                self._is_dragging = False
                self._selection_complete = True
                self._resize_mode = None
                context.area.header_text_set("Drag handles to resize | Enter/Space to crop | ESC to cancel")
                return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_crop_preview_image, (), 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._selection_complete = False
            self._resize_mode = None
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            self._image = context.space_data.image
            utils.crop_preview_start = self._start_pos
            utils.crop_preview_end = self._end_pos
            context.area.header_text_set("Drag to select crop region")
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def _cleanup(self, context):
        """Clean up state and handlers."""
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        
        utils.crop_preview_start = None
        utils.crop_preview_end = None
        self._is_dragging = False
        self._selection_complete = False
        self._resize_mode = None
        self._image = None
        context.area.tag_redraw()
    
    def _apply_crop(self, context):
        """Apply the crop operation to the image."""
        if not self._image or not self._start_pos or not self._end_pos:
            return False
        
        props = context.scene.text_tool_properties
        
        # Convert screen coordinates to image coordinates
        region = context.region
        view2d = region.view2d
        
        # Get image dimensions
        img_width, img_height = self._image.size
        if img_width == 0 or img_height == 0:
            return False
        
        # Convert screen positions to UV coordinates (0-1 range)
        uv_start = view2d.region_to_view(*self._start_pos)
        uv_end = view2d.region_to_view(*self._end_pos)
        
        # Convert UV to pixel coordinates
        px_x1 = int(uv_start[0] * img_width)
        px_y1 = int(uv_start[1] * img_height)
        px_x2 = int(uv_end[0] * img_width)
        px_y2 = int(uv_end[1] * img_height)
        
        # Ensure proper ordering (min to max)
        crop_x1 = min(px_x1, px_x2)
        crop_y1 = min(px_y1, px_y2)
        crop_x2 = max(px_x1, px_x2)
        crop_y2 = max(px_y1, px_y2)
        
        # If expand canvas is disabled, clamp to image bounds
        if not props.crop_expand_canvas:
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(img_width, crop_x2)
            crop_y2 = min(img_height, crop_y2)
        
        # Calculate new dimensions
        new_width = crop_x2 - crop_x1
        new_height = crop_y2 - crop_y1
        
        # Minimum crop size check
        if new_width < 2 or new_height < 2:
            return False
        
        # Save undo state before modifying
        utils.ImageUndoStack.get().push_state(self._image)
        
        # Get original pixels
        num_pixels = img_width * img_height * 4
        original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(original_pixels)
        
        # Create new pixel array, filled with fill color if expanding
        new_num_pixels = new_width * new_height * 4
        fill_r, fill_g, fill_b, fill_a = props.crop_fill_color
        new_pixels = array.array('f', [fill_r, fill_g, fill_b, fill_a] * (new_width * new_height))
        
        # Copy pixels from source to destination
        for y in range(new_height):
            src_y = crop_y1 + y
            # Skip rows outside original image
            if src_y < 0 or src_y >= img_height:
                continue
            
            # Calculate source and destination X ranges
            src_x_start = max(0, crop_x1)
            src_x_end = min(img_width, crop_x2)
            
            # Calculate offset in destination
            dst_x_offset = src_x_start - crop_x1
            copy_width = src_x_end - src_x_start
            
            if copy_width <= 0:
                continue
            
            src_start = (src_y * img_width + src_x_start) * 4
            src_end = src_start + copy_width * 4
            dst_start = (y * new_width + dst_x_offset) * 4
            dst_end = dst_start + copy_width * 4
            new_pixels[dst_start:dst_end] = original_pixels[src_start:src_end]
        
        # Apply resolution scaling if enabled
        if props.crop_use_resolution:
            final_width = props.crop_resolution_x
            final_height = props.crop_resolution_y
        else:
            final_width = new_width
            final_height = new_height
        
        # Resize image and set new pixels
        self._image.scale(new_width, new_height)
        self._image.pixels.foreach_set(new_pixels)
        
        # If resolution is different, scale again
        if props.crop_use_resolution and (final_width != new_width or final_height != new_height):
            self._image.scale(final_width, final_height)
        
        self._image.update()
        
        return True


class TEXTURE_PAINT_OT_input_text(Operator):
    bl_idname = "paint.input_text_ttf"
    bl_label = "Input Text"
    bl_options = {'REGISTER', 'UNDO'}

    text_input: StringProperty(
        name="Enter Text",
        description="Text to be painted",
        default=""
    )

    def execute(self, context):
        props = context.scene.text_tool_properties
        props.text = self.text_input
        return {'FINISHED'}

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self.text_input = props.text
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "text_input")


# ----------------------------
# Undo/Redo Operators
# ----------------------------
def _get_active_image_for_undo(context):
    """Get the active image for undo/redo based on current context."""
    # Check if we're in image editor
    if context.area and context.area.type == 'IMAGE_EDITOR':
        if context.space_data and context.space_data.image:
            return context.space_data.image
    
    # Check if we're in 3D viewport texture paint
    if context.mode == 'PAINT_TEXTURE':
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.active_material:
            mat = obj.active_material
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.select:
                        return node.image
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        return node.image
    return None





class TEXTTOOL_OT_undo(Operator):
    bl_idname = "texttool.undo"
    bl_label = "Undo Text Paint"
    bl_description = "Undo the last text paint operation"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        image = _get_active_image_for_undo(context)
        return utils.ImageUndoStack.get().can_undo(image)

    def execute(self, context):
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().undo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Text paint undone")
            return {'FINISHED'}
        self.report({'WARNING'}, "Nothing to undo")
        return {'CANCELLED'}


class TEXTTOOL_OT_redo(Operator):
    bl_idname = "texttool.redo"
    bl_label = "Redo Text Paint"
    bl_description = "Redo the last undone text paint operation"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        image = _get_active_image_for_undo(context)
        return utils.ImageUndoStack.get().can_redo(image)

    def execute(self, context):
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().redo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Text paint redone")
            return {'FINISHED'}
        self.report({'WARNING'}, "Nothing to redo")
        return {'CANCELLED'}

# ----------------------------
# Texture Randomizer Operators
# ----------------------------

class TEXTTOOL_OT_add_texture(Operator):
    bl_idname = "texttool.add_texture"
    bl_label = "Add Texture"
    bl_description = "Add a texture to the randomization list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.text_tool_properties
        props.texture_list.add()
        props.texture_index = len(props.texture_list) - 1
        return {'FINISHED'}

class TEXTTOOL_OT_remove_texture(Operator):
    bl_idname = "texttool.remove_texture"
    bl_label = "Remove Texture"
    bl_description = "Remove the selected texture from the list"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.text_tool_properties
        return props.texture_list and len(props.texture_list) > 0

    def execute(self, context):
        props = context.scene.text_tool_properties
        index = props.texture_index
        props.texture_list.remove(index)
        props.texture_index = min(max(0, index - 1), len(props.texture_list) - 1)
        return {'FINISHED'}

class TEXTTOOL_OT_move_texture(Operator):
    bl_idname = "texttool.move_texture"
    bl_label = "Move Texture"
    bl_description = "Move the selected texture up or down in the list"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        props = context.scene.text_tool_properties
        return props.texture_list and len(props.texture_list) > 1

    def execute(self, context):
        props = context.scene.text_tool_properties
        index = props.texture_index
        new_index = index - 1 if self.direction == 'UP' else index + 1
        
        if 0 <= new_index < len(props.texture_list):
            props.texture_list.move(index, new_index)
            props.texture_index = new_index
            return {'FINISHED'}
        return {'CANCELLED'}

# ----------------------------
# Pen Tool (3D Viewport)
# ----------------------------

class TEXTURE_PAINT_OT_pen_tool(Operator):
    bl_idname = "texture_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths on 3D surface"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _is_closed = False
    
    @classmethod
    def poll(cls, context):
        return (context.area.type == 'VIEW_3D' and 
                context.mode == 'PAINT_TEXTURE')
    
    def modal(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_drag_handle
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        # Screen coordinates for 3D view are just (mx, my)
        # We store them directly
        
        pen_preview_pos = (mx, my)
        
        if event.type == 'MOUSEMOVE':
            if pen_is_dragging and len(pen_points) > 0:
                # Adjust handle of last point
                last_pt = pen_points[-1]
                if pen_drag_handle == 'OUT':
                    # Set handle out
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[2], last_pt[3], mx, my)
                    # Mirror handle in
                    dx = mx - last_pt[0]
                    dy = my - last_pt[1]
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[0] - dx, last_pt[1] - dy, mx, my)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # Check if clicking near first point to close path
                if len(pen_points) >= 3:
                    first_pt = pen_points[0]
                    dist = ((mx - first_pt[0])**2 + (my - first_pt[1])**2)**0.5
                    if dist < 10:
                        # Close the path
                        self._is_closed = True
                        self._apply_path(context)
                        self._cleanup(context)
                        return {'FINISHED'}
                
                # Add new point
                pen_points.append((mx, my, mx, my, mx, my))
                pen_is_dragging = True
                pen_drag_handle = 'OUT'
                context.area.header_text_set(f"Point {len(pen_points)} | Drag to adjust curve | Enter to apply | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE':
                pen_is_dragging = False
                pen_drag_handle = None
                return {'RUNNING_MODAL'}
        
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            # Delete last point
            if len(pen_points) > 0:
                pen_points.pop()
                context.area.header_text_set(f"Point deleted | {len(pen_points)} points remaining")
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(pen_points) >= 2:
                self._apply_path(context)
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging
        
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        self._is_closed = False
        
        # Add draw handler
        from . import ui
        # We need a 3D specific draw handler because draw_pen_preview assumes image coords and does conversion
        # Use draw_pen_preview but we need to trick it or modify it?
        # Creating a new draw handler is cleaner.
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            ui.draw_pen_preview_3d, (context,), 'WINDOW', 'POST_PIXEL')
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Click to add points | Drag to adjust curves | Enter/Space to apply | ESC to cancel")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        global pen_points, pen_preview_pos, pen_is_dragging
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def _apply_path(self, context):
        """Project the screen path to 3D surface and paint."""
        global pen_points
        if len(pen_points) < 2:
            return
        
        # Get active object and image
        obj = context.active_object
        if not obj:
            return
            
        # Try to find active image from material
        mat = obj.active_material
        if not mat:
            return
            
        nodes = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
        active_node = nodes[0] if nodes else None
        for n in nodes:
            if n.select:
                active_node = n
                break
        
        if not active_node or not active_node.image:
            self.report({'WARNING'}, "No active image found on material")
            return
            
        image = active_node.image
        width, height = image.size
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        import numpy as np
        
        # Get pixels
        num_pixels = width * height * 4
        pixels = np.zeros(num_pixels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape((height, width, 4))
        
        # Generate dense screen points
        screen_points = self._generate_bezier_points(segments_per_curve=40)
        
        # Project to UV space
        uv_paths = []
        current_path = []
        
        last_uv = None
        
        # Reuse view3d_raycast_uv from TEXTURE_PAINT_OT_text_tool?
        # We can reproduce the logic here for simplicity or call it if we had an instance (we don't)
        
        for pt in screen_points:
            # Create a mock event for raycasting
            class MockEvent:
                def __init__(self, x, y):
                    self.mouse_region_x = x
                    self.mouse_region_y = y
            
            mock_evt = MockEvent(pt[0], pt[1])
            
            # Perform raycast
            hit_loc, face_idx, uv, _, _ = self.view3d_raycast_uv(context, mock_evt, obj)
            
            if uv:
                uv_pixel = (int(uv[0] * width), int(uv[1] * height))
                
                if last_uv:
                    # Check for seam (distance threshold)
                    # In UV space (0-1), a jump of > 0.1 is likely a seam or gap
                    dist = ((uv[0] - last_uv[0])**2 + (uv[1] - last_uv[1])**2)**0.5
                    if dist > 0.1:
                        # Start new path
                        if current_path:
                            uv_paths.append(current_path)
                            current_path = []
                
                current_path.append(uv_pixel)
                last_uv = uv
            else:
                # Gap in projection (off mesh)
                if current_path:
                    uv_paths.append(current_path)
                    current_path = []
                last_uv = None
        
        if current_path:
            uv_paths.append(current_path)
            
        # Draw strokes on image
        props = context.scene.text_tool_properties
        brush = context.tool_settings.image_paint.brush
        blend_mode = brush.blend if brush else 'MIX'
        use_aa = props.use_antialiasing
        
        if props.pen_use_stroke:
            stroke_color = np.array(props.pen_stroke_color, dtype=np.float32)
            stroke_width = props.pen_stroke_width
            
            # Helper to draw polyline (copied from Image Pen Tool)
            # We can use the method from IMAGE_PAINT_OT_pen_tool if we make it static or mixin
            # For now, let's duplicate the _draw_polyline helper or just define it here
            
            for path in uv_paths:
                if len(path) >= 2:
                    self._draw_polyline(pixels, path, stroke_color, stroke_width, width, height, blend_mode, use_aa)
        
        # Apply pixels
        image.pixels.foreach_set(pixels.flatten())
        image.update()

    def _generate_bezier_points(self, segments_per_curve=20):
        global pen_points
        points = []
        for i in range(len(pen_points) - 1):
            p0 = pen_points[i]
            p1 = pen_points[i + 1]
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]
            x2, y2 = p1[2], p1[3]
            x3, y3 = p1[0], p1[1]
            for t in range(segments_per_curve + 1):
                t_val = t / segments_per_curve
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                points.append((int(x), int(y)))
        return points

    def view3d_raycast_uv(self, context, event, obj):
        region = context.region
        rv3d = context.region_data
        if not rv3d:
            return None, None, None, None, None

        # Build ray from mouse in world space
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()

        near = view_origin + view_dir * 0.001
        far = view_origin + view_dir * 1e6

        inv = obj.matrix_world.inverted()
        ro_local = inv @ near
        rf_local = inv @ far
        rd_local = (rf_local - ro_local).normalized()

        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        bvh = BVHTree.FromBMesh(bm)
        hit = bvh.ray_cast(ro_local, rd_local)
        
        if not hit or not hit[0]:
            bm.free()
            return None, None, None, None, None

        hit_loc_local, hit_normal_local, face_index, distance = hit
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return hit_loc_local, face_index, None, None, None

        face = bm.faces[face_index]
        p = hit_loc_local
        
        from mathutils.interpolate import poly_3d_calc
        
        vert_coords = [v.co for v in face.verts]
        loop_uvs = [loop[uv_layer].uv for loop in face.loops]
        
        weights = poly_3d_calc(vert_coords, p)
        
        u_interp = sum(w * uv.x for w, uv in zip(weights, loop_uvs))
        v_interp = sum(w * uv.y for w, uv in zip(weights, loop_uvs))
        best_uv = Vector((u_interp, v_interp))
        result_uv = (best_uv.x, best_uv.y)
        
        bm.free()
        return hit_loc_local, face_index, result_uv, None, None

    def _apply_blend_mode(self, dst, src_color, blend_mode):
        import numpy as np
        d_rgb = dst[..., :3]
        s_rgb = np.ones_like(d_rgb) * src_color[:3]
        out_rgb = s_rgb.copy()
        
        if blend_mode == 'MIX': out_rgb = s_rgb
        elif blend_mode == 'DARKEN': out_rgb = np.minimum(d_rgb, s_rgb)
        elif blend_mode == 'MUL': out_rgb = d_rgb * s_rgb
        elif blend_mode == 'LIGHTEN': out_rgb = np.maximum(d_rgb, s_rgb)
        elif blend_mode == 'SCREEN': out_rgb = 1.0 - (1.0 - d_rgb) * (1.0 - s_rgb)
        elif blend_mode == 'ADD': out_rgb = np.clip(d_rgb + s_rgb, 0.0, 1.0)
        elif blend_mode == 'SUB': out_rgb = np.clip(d_rgb - s_rgb, 0.0, 1.0)
        elif blend_mode == 'OVERLAY':
            mask = d_rgb < 0.5
            out_rgb = np.where(mask, 2.0 * d_rgb * s_rgb, 1.0 - 2.0 * (1.0 - d_rgb) * (1.0 - s_rgb))
        
        alpha = np.ones(dst.shape[:2] + (1,), dtype=np.float32)
        return np.dstack((out_rgb, alpha))

    def _draw_polyline(self, pixels, points, color, width, img_width, img_height, blend_mode='MIX', use_aa=True):
        import numpy as np
        if len(points) < 2: return
        
        pts = np.array(points)
        pad = width + 2
        min_x = max(0, int(pts[:, 0].min()) - pad)
        max_x = min(img_width - 1, int(pts[:, 0].max()) + pad)
        min_y = max(0, int(pts[:, 1].min()) - pad)
        max_y = min(img_height - 1, int(pts[:, 1].max()) + pad)
        
        if max_x <= min_x or max_y <= min_y: return
        
        stroke_height = max_y - min_y + 1
        stroke_width_px = max_x - min_x + 1
        stroke_mask = np.zeros((stroke_height, stroke_width_px), dtype=np.float32)
        
        half_w = width // 2 + 1
        by, bx = np.ogrid[-half_w:half_w+1, -half_w:half_w+1]
        dist = np.sqrt(bx**2 + by**2).astype(np.float32)
        
        if use_aa: brush = np.clip(1.0 - (dist - width/2.0 + 0.5) / 1.5, 0.0, 1.0)
        else: brush = (dist <= width/2.0).astype(np.float32)
        
        for i in range(len(points) - 1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
            err = dx - dy
            
            while True:
                lx, ly = x1 - min_x, y1 - min_y
                msy = max(0, ly - half_w)
                mey = min(stroke_height, ly + half_w + 1)
                msx = max(0, lx - half_w)
                mex = min(stroke_width_px, lx + half_w + 1)
                
                bsy = msy - (ly - half_w)
                bey = bsy + (mey - msy)
                bsx = msx - (lx - half_w)
                bex = bsx + (mex - msx)
                
                if mey > msy and mex > msx:
                    stroke_mask[msy:mey, msx:mex] = np.maximum(stroke_mask[msy:mey, msx:mex], brush[bsy:bey, bsx:bex])
                
                if x1 == x2 and y1 == y2: break
                e2 = 2 * err
                if e2 > -dy: err -= dy; x1 += sx
                if e2 < dx: err += dx; y1 += sy
        
        alpha = color[3]
        final_alpha = stroke_mask * alpha
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        for c in range(4):
            pixels[min_y:max_y+1, min_x:max_x+1, c] = dst[:, :, c] * (1 - final_alpha) + blended[:, :, c] * final_alpha


# ============================================================
# Layer Operatorsnow only this 
# ============================================================

import numpy as np

class IMAGE_EDIT_OT_make_selection(bpy.types.Operator):
    """Make a selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_editor_plus.make_selection"
    bl_label = "Make Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'  # 'SET', 'ADD', 'SUBTRACT'

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size[0], img.size[1]

        if event.type == 'MOUSEMOVE':
            if self.lmb:
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                if area_session.selection_region:
                    area_session.selection_region[1] = region_pos
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                area_session.selection_region = [region_pos, region_pos]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.selection_region:
                    area_session.selecting = False
                    # Pass mode to convert_selection
                    utils.layer_convert_selection(context, mode=self.mode)
                    # Clear the temporary region data after converting
                    area_session.selection_region = None
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.imageeditorplus_properties
                    img_props.selected_layer_index = -1
                    return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel only the current drag, not all selections
            area_session.selection_region = None
            area_session.selecting = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode: modifier keys override UI selection
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            # Use UI-selected mode when no modifier keys
            wm = context.window_manager
            if hasattr(wm, 'imageeditorplus_properties'):
                self.mode = wm.imageeditorplus_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.selection_region = None
        area_session.selecting = True
        self.lmb = True
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        area_session.selection_region = [region_pos, region_pos[:]]
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_cancel_selection(bpy.types.Operator):
    """Cancel the selection"""
    bl_idname = "image_editor_plus.cancel_selection"
    bl_label = "Cancel Selection"

    def execute(self, context):
        area_session = utils.layer_get_area_session(context)
        # Check if there's anything to cancel (any selection type or negation)
        has_selections = (area_session.selections or area_session.ellipses or 
                         area_session.lassos or area_session._neg_rects or
                         area_session._neg_ellipses or area_session._neg_lassos or
                         area_session.selection_region or area_session.ellipse_region or
                         area_session.lasso_points)
        if not has_selections:
            return {'CANCELLED'}
        utils.layer_cancel_selection(context)
        context.area.tag_redraw()
        return {'FINISHED'}

class IMAGE_EDIT_OT_undo_selection(bpy.types.Operator):
    """Undo the last selection change"""
    bl_idname = "image_editor_plus.undo_selection"
    bl_label = "Undo"

    def execute(self, context):
        # 1. Try Selection Undo
        area_session = utils.layer_get_area_session(context)
        if area_session.undo_selection():
            utils.layer_clear_paint_mask(context)
            # Re-apply paint mask if there are selections
            if area_session.selections or area_session.ellipses or area_session.lassos:
                utils.layer_apply_selection_as_paint_mask(context)
            context.area.tag_redraw()
            return {'FINISHED'}
            
        # 2. Try Image Pixel Undo (Custom Stack)
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().undo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Paint undone")
            return {'FINISHED'}
            
        # 3. Try Native Blender Undo
        try:
            bpy.ops.ed.undo()
            return {'FINISHED'}
        except Exception:
            return {'CANCELLED'}

class IMAGE_EDIT_OT_redo_selection(bpy.types.Operator):
    """Redo the last undone selection change"""
    bl_idname = "image_editor_plus.redo_selection"
    bl_label = "Redo"

    def execute(self, context):
        # 1. Try Selection Redo
        area_session = utils.layer_get_area_session(context)
        if area_session.redo_selection():
            utils.layer_clear_paint_mask(context)
            # Re-apply paint mask if there are selections
            if area_session.selections or area_session.ellipses or area_session.lassos:
                utils.layer_apply_selection_as_paint_mask(context)
            context.area.tag_redraw()
            return {'FINISHED'}
            
        # 2. Try Image Pixel Redo (Custom Stack)
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().redo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Paint redone")
            return {'FINISHED'}
            
        # 3. Try Native Blender Redo
        try:
            bpy.ops.ed.redo()
            return {'FINISHED'}
        except Exception:
            return {'CANCELLED'}


class IMAGE_EDIT_OT_make_ellipse_selection(bpy.types.Operator):
    """Make an ellipse selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_editor_plus.make_ellipse_selection"
    bl_label = "Make Ellipse Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self.lmb:
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                if area_session.ellipse_region:
                    area_session.ellipse_region[1] = region_pos
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                area_session.ellipse_region = [region_pos, region_pos[:]]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.ellipse_region:
                    area_session.selecting = False
                    utils.layer_convert_ellipse_selection(context, mode=self.mode)
                    # Clear the temporary region data after converting
                    area_session.ellipse_region = None
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.imageeditorplus_properties
                    img_props.selected_layer_index = -1
                    return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            area_session.ellipse_region = None
            area_session.selecting = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode: modifier keys override UI selection
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            wm = context.window_manager
            if hasattr(wm, 'imageeditorplus_properties'):
                self.mode = wm.imageeditorplus_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.ellipse_region = None
        area_session.selecting = True
        self.lmb = True
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        area_session.ellipse_region = [region_pos, region_pos[:]]
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_make_lasso_selection(bpy.types.Operator):
    """Make a lasso selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_editor_plus.make_lasso_selection"
    bl_label = "Make Lasso Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'
        self.min_dist = 3  # Minimum distance between points for performance

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self.lmb and area_session.lasso_points:
                x, y = event.mouse_region_x, event.mouse_region_y
                # Add point if far enough from last point (performance optimization)
                last = area_session.lasso_points[-1]
                dist = ((x - last[0])**2 + (y - last[1])**2)**0.5
                if dist >= self.min_dist:
                    area_session.lasso_points.append([x, y])
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                x, y = event.mouse_region_x, event.mouse_region_y
                area_session.lasso_points = [[x, y]]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.lasso_points and len(area_session.lasso_points) >= 3:
                    area_session.selecting = False
                    utils.layer_convert_lasso_selection(context, mode=self.mode)
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.imageeditorplus_properties
                    img_props.selected_layer_index = -1
                    area_session.lasso_points = None
                    return {'FINISHED'}
                else:
                    area_session.lasso_points = None
                    return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            area_session.lasso_points = None
            area_session.selecting = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            wm = context.window_manager
            if hasattr(wm, 'imageeditorplus_properties'):
                self.mode = wm.imageeditorplus_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.lasso_points = None
        area_session.selecting = True
        self.lmb = True
        x, y = event.mouse_region_x, event.mouse_region_y
        area_session.lasso_points = [[x, y]]
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_swap_colors(bpy.types.Operator):
    """Swap foreground and background color"""
    bl_idname = "image_editor_plus.swap_colors"
    bl_label = "Swap Colors"

    def execute(self, context):
        wm = context.window_manager
        props = wm.imageeditorplus_properties
        props.foreground_color, props.background_color = props.background_color[:], props.foreground_color[:]
        return {'FINISHED'}

class IMAGE_EDIT_OT_fill_with_fg_color(bpy.types.Operator):
    """Fill the image with foreground color"""
    bl_idname = "image_editor_plus.fill_with_fg_color"
    bl_label = "Fill with FG Color"

    def execute(self, context):
        wm = context.window_manager
        props = wm.imageeditorplus_properties
        color = props.foreground_color[:] + (1.0,)
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = color
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = color
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_fill_with_bg_color(bpy.types.Operator):
    """Fill the image with background color"""
    bl_idname = "image_editor_plus.fill_with_bg_color"
    bl_label = "Fill with BG Color"

    def execute(self, context):
        wm = context.window_manager
        props = wm.imageeditorplus_properties
        color = props.background_color[:] + (1.0,)
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = color
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = color
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_clear(bpy.types.Operator):
    """Clear the image"""
    bl_idname = "image_editor_plus.clear"
    bl_label = "Clear"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = (0, 0, 0, 0)
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = (0, 0, 0, 0)
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_cut(bpy.types.Operator):
    """Cut the image"""
    bl_idname = "image_editor_plus.cut"
    bl_label = "Cut"

    def execute(self, context):
        session = utils.layer_get_session()
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels
        session.copied_image_pixels = target_pixels.copy()
        session.copied_image_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            session.copied_layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            session.copied_layer_settings = None
        utils.ImageUndoStack.get().push_state(img)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = (0, 0, 0, 0)
        else:
            pixels[:] = (0, 0, 0, 0)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Cut selected image.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_copy(bpy.types.Operator):
    """Copy the image"""
    bl_idname = "image_editor_plus.copy"
    bl_label = "Copy"

    def execute(self, context):
        session = utils.layer_get_session()
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels
        session.copied_image_pixels = target_pixels
        session.copied_image_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            session.copied_layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            session.copied_layer_settings = None
        self.report({'INFO'}, 'Copied selected image.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_paste(bpy.types.Operator):
    """Paste the image"""
    bl_idname = "image_editor_plus.paste"
    bl_label = "Paste"

    def execute(self, context):
        session = utils.layer_get_session()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        target_pixels = session.copied_image_pixels
        if target_pixels is None:
            return {'CANCELLED'}
        utils.layer_create_layer(img, target_pixels, session.copied_image_settings, session.copied_layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_cut_to_layer(bpy.types.Operator):
    """Cut selection and paste as new layer"""
    bl_idname = "image_editor_plus.cut_to_layer"
    bl_label = "Cut Selection to New Layer"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        base_img = context.area.spaces.active.image
        if not img or not base_img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]].copy()
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels.copy()
        
        img_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            layer_settings = None
        
        utils.ImageUndoStack.get().push_state(img)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = (0, 0, 0, 0)
        else:
            pixels[:] = (0, 0, 0, 0)
        utils.layer_write_pixels_to_image(img, pixels)
        
        utils.layer_create_layer(base_img, target_pixels, img_settings, layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Cut selection to new layer.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_copy_to_layer(bpy.types.Operator):
    """Copy selection and paste as new layer"""
    bl_idname = "image_editor_plus.copy_to_layer"
    bl_label = "Copy Selection to New Layer"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        base_img = context.area.spaces.active.image
        if not img or not base_img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]].copy()
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels.copy()
        
        img_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            layer_settings = None
        
        utils.layer_create_layer(base_img, target_pixels, img_settings, layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Copied selection to new layer.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_add_image_layer(bpy.types.Operator):
    """Add image file(s) as new layer(s)"""
    bl_idname = "image_editor_plus.add_image_layer"
    bl_label = "Add Image as Layer"
    
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.tiff;*.exr;*.hdr", options={'HIDDEN'})

    def execute(self, context):
        import os
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.files:
            filepaths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        else:
            filepaths = [self.filepath]
        
        added_count = 0
        for filepath in filepaths:
            if not filepath or not os.path.isfile(filepath):
                continue
            
            try:
                layer_source = bpy.data.images.load(filepath)
            except:
                self.report({'WARNING'}, f"Could not load: {os.path.basename(filepath)}")
                continue
            
            original_filename = os.path.splitext(os.path.basename(filepath))[0]
            target_pixels = utils.layer_read_pixels_from_image(layer_source)
            
            img_settings = {
                'is_float': layer_source.is_float,
                'colorspace_name': layer_source.colorspace_settings.name
            }
            
            utils.layer_create_layer(img, target_pixels, img_settings, None, custom_label=original_filename)
            bpy.data.images.remove(layer_source)
            added_count += 1
        
        if added_count == 0:
            self.report({'ERROR'}, "No images were added")
            return {'CANCELLED'}
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        self.report({'INFO'}, f'{added_count} image(s) added as layers.')
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_new_image_layer(bpy.types.Operator):
    """Create a new blank image as a layer"""
    bl_idname = "image_editor_plus.new_image_layer"
    bl_label = "New Image Layer"
    
    layer_name: bpy.props.StringProperty(name='Name', default='New Layer')
    width: bpy.props.IntProperty(name='Width', default=512, min=1, max=16384)
    height: bpy.props.IntProperty(name='Height', default=512, min=1, max=16384)
    color: bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', size=4, min=0, max=1, default=(1.0, 1.0, 1.0, 0.0))
    use_base_size: bpy.props.BoolProperty(name='Use Base Image Size', default=True)

    def invoke(self, context, event):
        img = context.area.spaces.active.image
        if img:
            self.width = img.size[0]
            self.height = img.size[1]
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.use_base_size:
            layer_width = img.size[0]
            layer_height = img.size[1]
        else:
            layer_width = self.width
            layer_height = self.height
        
        pixels = np.full((layer_height, layer_width, 4), self.color, dtype=np.float32)
        
        img_settings = {
            'is_float': img.is_float,
            'colorspace_name': img.colorspace_settings.name
        }
        
        layer_label = self.layer_name if self.layer_name else "New Layer"
        utils.layer_create_layer(img, pixels, img_settings, None, custom_label=layer_label)
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'layer_name')
        layout.prop(self, 'use_base_size')
        if not self.use_base_size:
            layout.prop(self, 'width')
            layout.prop(self, 'height')
        layout.prop(self, 'color')

class IMAGE_EDIT_OT_crop(bpy.types.Operator):
    """Crop the image to the boundary of the selection"""
    bl_idname = "image_editor_plus.crop"
    bl_label = "Crop"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        else:
            target_pixels = pixels
        target_width, target_height = target_pixels.shape[1], target_pixels.shape[0]
        img.scale(target_width, target_height)
        utils.layer_write_pixels_to_image(img, target_pixels)
        if selection:
            img_props = img.imageeditorplus_properties
            layers = img_props.layers
            for layer in reversed(layers):
                layer_pos = layer.location
                layer_pos[0] -= selection[0][0]
                layer_pos[1] -= selection[0][1]
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_layer(bpy.types.Operator):
    bl_idname = "image_editor_plus.deselect_layer"
    bl_label = "Deselect Layer"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        img_props.selected_layer_index = -1
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_move_layer(bpy.types.Operator):
    """Move the layer"""
    bl_idname = "image_editor_plus.move_layer"
    bl_label = "Move Layer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_location = [0, 0]

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if event.type == 'MOUSEMOVE':
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            layer.location[0] = int(self.start_layer_location[0] + target_x - self.start_input_position[0])
            layer.location[1] = int(self.start_layer_location[1] - (target_y - self.start_input_position[1]))
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.location = self.start_layer_location
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_location = layer.location[:]
        area_session.layer_moving = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_delete_layer(bpy.types.Operator):
    bl_idname = "image_editor_plus.delete_layer"
    bl_label = "Delete Layer"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        layer = layers[selected_layer_index]
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            bpy.data.images.remove(layer_img)
        layers.remove(selected_layer_index)
        selected_layer_index = min(max(selected_layer_index, 0), len(layers) - 1)
        img_props.selected_layer_index = selected_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_edit_layer(bpy.types.Operator):
    """Toggle layer edit mode - paint directly on the selected layer"""
    bl_idname = "image_editor_plus.edit_layer"
    bl_label = "Edit Layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        img_props = img.imageeditorplus_properties
        # Allow if currently editing (to exit) or if a layer is selected (to enter)
        if img_props.editing_layer:
            return True
        if img_props.selected_layer_index >= 0 and img_props.selected_layer_index < len(img_props.layers):
            return True
        return False

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        
        img_props = img.imageeditorplus_properties
        
        # Toggle mode
        if img_props.editing_layer:
            # Exit edit mode
            if utils.layer_exit_edit_mode(context):
                self.report({'INFO'}, 'Exited layer edit mode')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'Failed to exit layer edit mode')
                return {'CANCELLED'}
        else:
            # Enter edit mode
            layer = utils.layer_get_active_layer(context)
            if layer and layer.locked:
                self.report({'WARNING'}, 'Layer is locked')
                return {'CANCELLED'}
            
            if utils.layer_enter_edit_mode(context):
                self.report({'INFO'}, 'Editing layer - paint directly on layer image')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'No layer selected')
                return {'CANCELLED'}

class IMAGE_EDIT_OT_duplicate_layer(bpy.types.Operator):

    """Duplicate the selected layer"""
    bl_idname = "image_editor_plus.duplicate_layer"
    bl_label = "Duplicate Layer"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(layer_img)
        img_settings = {'is_float': layer_img.is_float, 'colorspace_name': layer_img.colorspace_settings.name}
        layer_settings = {'rotation': layer.rotation, 'scale': list(layer.scale), 'custom_data': layer.custom_data}
        
        utils.layer_create_layer(img, pixels, img_settings, layer_settings, custom_label=layer.label + " Copy")
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Layer duplicated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_lock_all_layers(bpy.types.Operator):
    """Lock all layers"""
    bl_idname = "image_editor_plus.lock_all_layers"
    bl_label = "Lock All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.locked = True
        self.report({'INFO'}, 'All layers locked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_unlock_all_layers(bpy.types.Operator):
    """Unlock all layers"""
    bl_idname = "image_editor_plus.unlock_all_layers"
    bl_label = "Unlock All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.locked = False
        self.report({'INFO'}, 'All layers unlocked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_hide_all_layers(bpy.types.Operator):
    """Hide all layers"""
    bl_idname = "image_editor_plus.hide_all_layers"
    bl_label = "Hide All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.hide = True
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers hidden.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_show_all_layers(bpy.types.Operator):
    """Show all layers"""
    bl_idname = "image_editor_plus.show_all_layers"
    bl_label = "Show All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.hide = False
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers shown.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_all_layers(bpy.types.Operator):
    """Delete all layers"""
    bl_idname = "image_editor_plus.delete_all_layers"
    bl_label = "Delete All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        
        for layer in layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
        
        layers.clear()
        img_props.selected_layer_index = -1
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_update_layer_previews(bpy.types.Operator):
    """Update all layer preview thumbnails"""
    bl_idname = "image_editor_plus.update_layer_previews"
    bl_label = "Update Layer Previews"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        
        for layer in img_props.layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                layer_img.update()
                if layer_img.preview:
                    layer_img.preview.reload()
        
        img.update()
        if img.preview:
            img.preview.reload()
        
        context.area.tag_redraw()
        self.report({'INFO'}, 'Layer previews updated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_select_all_layers(bpy.types.Operator):
    """Select all layers"""
    bl_idname = "image_editor_plus.select_all_layers"
    bl_label = "Select All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.checked = True
        self.report({'INFO'}, 'All layers selected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_all_layers(bpy.types.Operator):
    """Deselect all layers"""
    bl_idname = "image_editor_plus.deselect_all_layers"
    bl_label = "Deselect All Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.checked = False
        self.report({'INFO'}, 'All layers deselected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_invert_layer_selection(bpy.types.Operator):
    """Invert layer selection"""
    bl_idname = "image_editor_plus.invert_layer_selection"
    bl_label = "Invert Layer Selection"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        for layer in img_props.layers:
            layer.checked = not layer.checked
        self.report({'INFO'}, 'Layer selection inverted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_selected_layers(bpy.types.Operator):
    """Delete all selected (checked) layers"""
    bl_idname = "image_editor_plus.delete_selected_layers"
    bl_label = "Delete Selected Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        
        indices_to_remove = []
        for i, layer in enumerate(layers):
            if layer.checked and not layer.locked:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            self.report({'WARNING'}, 'No unlocked layers selected.')
            return {'CANCELLED'}
        
        for i in reversed(indices_to_remove):
            layer = layers[i]
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
            layers.remove(i)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{len(indices_to_remove)} layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_selected_layers(bpy.types.Operator):
    """Merge all selected (checked) layers"""
    bl_idname = "image_editor_plus.merge_selected_layers"
    bl_label = "Merge Selected Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        
        checked_layers = [(i, layer) for i, layer in enumerate(layers) if layer.checked]
        
        if len(checked_layers) < 2:
            self.report({'WARNING'}, 'Select at least 2 layers to merge.')
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(img)
        
        merged_count = 0
        indices_to_remove = []
        
        for i, layer in reversed(checked_layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            
            bpy.data.images.remove(layer_img)
            indices_to_remove.append(i)
            merged_count += 1
        
        for i in sorted(indices_to_remove, reverse=True):
            layers.remove(i)
        
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{merged_count} layers merged.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_change_image_layer_order(bpy.types.Operator):
    bl_idname = "image_editor_plus.change_image_layer_order"
    bl_label = "Change Image Layer Order"
    up: bpy.props.BoolProperty()

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        if (self.up and selected_layer_index == 0) or (not self.up and selected_layer_index >= len(layers) - 1):
            return {'CANCELLED'}
        new_layer_index = selected_layer_index + (-1 if self.up else 1)
        layers.move(selected_layer_index, new_layer_index)
        img_props.selected_layer_index = new_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_layers(bpy.types.Operator):
    """Merge all layers"""
    bl_idname = "image_editor_plus.merge_layers"
    bl_label = "Merge Layers"

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        img_props = img.imageeditorplus_properties
        layers = img_props.layers
        for layer in reversed(layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            bpy.data.images.remove(layer_img)
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        layers.clear()
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_flip_layer(bpy.types.Operator):
    """Flip the layer"""
    bl_idname = "image_editor_plus.flip_layer"
    bl_label = "Flip Layer"
    is_vertically: bpy.props.BoolProperty(name="Vertically", default=False)

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        if self.is_vertically:
            layer.scale[1] *= -1.0
        else:
            layer.scale[0] *= -1.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer(bpy.types.Operator):
    """Rotate the layer"""
    bl_idname = "image_editor_plus.rotate_layer"
    bl_label = "Rotate Layer"
    is_left: bpy.props.BoolProperty(name="Left", default=False)

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer.rotation += math.pi / 2.0 if self.is_left else -math.pi / 2.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer_arbitrary(bpy.types.Operator):
    """Rotate the image by a specified angle"""
    bl_idname = "image_editor_plus.rotate_layer_arbitrary"
    bl_label = "Rotate Layer Arbitrary"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_angle = 0

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
        if event.type == 'MOUSEMOVE':
            center_x = layer.location[0] + layer_width / 2.0
            center_y = height - layer.location[1] - layer_height / 2.0
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            angle1 = math.atan2(self.start_input_position[1] - center_y, self.start_input_position[0] - center_x)
            angle2 = math.atan2(target_y - center_y, target_x - center_x)
            layer.rotation = self.start_layer_angle + angle2 - angle1
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.rotation = self.start_layer_angle
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_angle = layer.rotation
        area_session.layer_rotating = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_scale_layer(bpy.types.Operator):
    """Scale the layer"""
    bl_idname = "image_editor_plus.scale_layer"
    bl_label = "Scale Layer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_scale_x = 1.0
        self.start_layer_scale_y = 1.0

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
        if event.type == 'MOUSEMOVE':
            center_x = layer.location[0] + layer_width / 2.0
            center_y = height - layer.location[1] - layer_height / 2.0
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            dist1 = math.hypot(self.start_input_position[0] - center_x, self.start_input_position[1] - center_y)
            dist2 = math.hypot(target_x - center_x, target_y - center_y)
            layer.scale[0] = self.start_layer_scale_x * dist2 / dist1
            layer.scale[1] = self.start_layer_scale_y * dist2 / dist1
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_scale_x = layer.scale[0]
        self.start_layer_scale_y = layer.scale[1]
        area_session.layer_scaling = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class IMAGE_EDIT_OT_sculpt_image(bpy.types.Operator):
    """Sculpt the image with brush-based pixel warping"""
    bl_idname = "image_editor_plus.sculpt_image"
    bl_label = "Image Sculpt"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.area.type == 'IMAGE_EDITOR' and 
                context.area.spaces.active.image is not None)

    def modal(self, context, event):
        import numpy as np
        
        if event.type == 'MOUSEMOVE':
            if self.last_pos is not None:
                region_pos = (event.mouse_region_x, event.mouse_region_y)
                view_x, view_y = context.region.view2d.region_to_view(*region_pos)
                curr_x = int(view_x * self.width)
                curr_y = int(view_y * self.height)
                
                wm = context.window_manager
                props = wm.imageeditorplus_properties
                
                dx = curr_x - self.last_pos[0]
                dy = curr_y - self.last_pos[1]
                
                if dx != 0 or dy != 0:
                    # Apply to working buffer
                    self._apply_to_buffer(curr_x, curr_y, dx, dy, 
                                         props.sculpt_mode, props.sculpt_radius, props.sculpt_strength)
                    self.modified = True
                    self.frame_count += 1
                    
                    # Real-time feedback: update image every few frames for performance
                    if self.frame_count % 3 == 0:
                        self.img.pixels.foreach_set(self.working.ravel())
                        self.img.update()
                        context.area.tag_redraw()
                
                self.last_pos = [curr_x, curr_y]
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Final write on release
            if self.modified:
                # Push undo state BEFORE finalizing (save the original pixels efficiently)
                utils.ImageUndoStack.get().push_state_from_numpy(self.img, self.original_pixels)
                
                # Now apply the final result
                self.img.pixels.foreach_set(self.working.ravel())
                self.img.update()
            context.area.tag_redraw()
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self.modified:
                self.img.pixels.foreach_set(self.original_pixels)
                self.img.update()
            context.area.tag_redraw()
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        import numpy as np
        
        self.img = context.area.spaces.active.image
        if not self.img:
            return {'CANCELLED'}
        
        self.width, self.height = self.img.size[0], self.img.size[1]
        self.modified = False
        self.frame_count = 0
        
        # Read pixels once - use foreach_get for speed
        pixel_count = self.width * self.height * 4
        self.original_pixels = np.empty(pixel_count, dtype=np.float32)
        self.img.pixels.foreach_get(self.original_pixels)
        
        # Working copy reshaped for manipulation
        self.working = self.original_pixels.reshape((self.height, self.width, 4)).copy()
        
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.last_pos = [int(view_x * self.width), int(view_y * self.height)]
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _apply_to_buffer(self, cx, cy, dx, dy, mode, radius, strength):
        """MLS Rigid deformation - O(k) per pixel, highly optimized."""
        import numpy as np
        import bpy
        
        # Get falloff preset from properties
        props = bpy.context.window_manager.imageeditorplus_properties
        falloff_preset = props.sculpt_falloff_preset
        
        # Bounding box
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.width, cx + radius)
        y2 = min(self.height, cy + radius)
        
        if x1 >= x2 or y1 >= y2:
            return
        
        # Coordinate grids (float32 for speed)
        py = np.arange(y1, y2, dtype=np.float32)
        px = np.arange(x1, x2, dtype=np.float32)
        gx, gy = np.meshgrid(px, py)
        
        # Distance from brush center
        rel_x = gx - cx
        rel_y = gy - cy
        dist_sq = rel_x * rel_x + rel_y * rel_y
        radius_sq = float(radius * radius)
        
        # Mask for valid region
        mask = dist_sq < radius_sq
        if not np.any(mask):
            return
        
        # Calculate normalized distance t (0 at center, 1 at edge)
        dist = np.sqrt(dist_sq)
        t = np.clip(dist / radius, 0.0, 1.0)
        
        # Calculate falloff based on preset (vectorized for performance)
        if falloff_preset == 'SMOOTH':
            # Hermite smoothstep: 3t² - 2t³
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        elif falloff_preset == 'SMOOTHER':
            # Perlin smootherstep: 6t⁵ - 15t⁴ + 10t³
            weights = (1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)) * strength
        elif falloff_preset == 'SPHERE':
            # Spherical: sqrt(1 - t²)
            weights = np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0)) * strength
        elif falloff_preset == 'ROOT':
            # Root: 1 - sqrt(t)
            weights = (1.0 - np.sqrt(t)) * strength
        elif falloff_preset == 'SHARP':
            # Sharp: (1 - t)²
            weights = ((1.0 - t)**2) * strength
        elif falloff_preset == 'LINEAR':
            # Linear: 1 - t
            weights = (1.0 - t) * strength
        elif falloff_preset == 'CONSTANT':
            # Constant: no falloff (hard edge)
            weights = np.where(mask, strength, 0.0)
        elif falloff_preset == 'CUSTOM':
            # Use brush curve - evaluate per-pixel (slower but accurate)
            brush = bpy.context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                curve = brush.curve_distance_falloff
                weights = np.zeros_like(t)
                # Vectorized curve evaluation using pre-sampled LUT for performance
                lut_size = 256
                lut = np.array([curve.evaluate(curve.curves[0], i / (lut_size - 1)) 
                               for i in range(lut_size)], dtype=np.float32)
                t_indices = np.clip((t * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
                weights = lut[t_indices] * strength
            else:
                # Fallback to smooth
                weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        else:
            # Default to smooth
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        
        # Apply mask
        weights = np.where(mask, weights, 0.0)
        
        # Control point: brush center
        p_x, p_y = float(cx), float(cy)
        
        # MLS Rigid transformation based on mode
        if mode == 'GRAB':
            # Weighted translation
            offset_x = dx * weights
            offset_y = dy * weights
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'PINCH':
            # Scale toward center
            scale = 1.0 + weights * 0.5
            src_x = p_x + rel_x * scale
            src_y = p_y + rel_y * scale
        elif mode == 'DRIP':
            # Realistic paint drip effect - teardrop shape
            # Distance from vertical center line (for tapering)
            dist_from_center = np.abs(rel_x)
            
            # Taper factor: 1 at center, 0 at edges (creates narrow drip shape)
            taper = np.maximum(0, 1 - dist_from_center / (radius * 0.3))
            taper = taper ** 0.5  # Softer taper
            
            # Drip amount: stronger in center, combines with gaussian weight
            drip_amount = weights * taper * radius * strength
            
            # Create elongation: pixels stretch downward from where brush is
            # Sample from above to create the stretching effect
            src_x = gx
            src_y = gy - drip_amount
            
            # Add bulge at bottom: pixels near brush center get extra stretch
            center_boost = np.exp(-dist_from_center**2 / (radius * 0.15)**2)
            src_y = src_y - center_boost * weights * radius * 0.2
            
        elif mode == 'WAVE':
            # Ripple/wave distortion - concentric rings from center
            dist = np.sqrt(dist_sq) + 0.001  # avoid division by zero
            
            # Wave parameters
            frequency = 0.3  # waves per pixel
            amplitude = weights * radius * 0.15
            
            # Radial wave displacement
            wave = np.sin(dist * frequency * 2 * np.pi) * amplitude
            
            # Displace perpendicular to radius (creates ripple effect)
            norm_x = rel_x / dist
            norm_y = rel_y / dist
            src_x = gx + norm_x * wave
            src_y = gy + norm_y * wave
            
        elif mode == 'JITTER':
            # Turbulence/noise displacement
            # Use hash-based pseudo-random for deterministic jitter
            hash_x = np.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
            hash_y = np.sin(gx * 78.233 + gy * 12.9898) * 43758.5453
            noise_x = (hash_x - np.floor(hash_x)) * 2 - 1  # -1 to 1
            noise_y = (hash_y - np.floor(hash_y)) * 2 - 1
            
            # Apply weighted displacement
            jitter_amount = weights * radius * 0.2
            src_x = gx + noise_x * jitter_amount
            src_y = gy + noise_y * jitter_amount
            
        elif mode == 'HAZE':
            # Heat haze - vertical refractive shimmer
            # Use position-based sine for shimmer pattern
            phase = gy * 0.3 + gx * 0.1
            shimmer_x = np.sin(phase) * weights * radius * 0.1
            shimmer_y = np.sin(phase * 1.3 + 1.0) * weights * radius * 0.05
            
            src_x = gx + shimmer_x
            src_y = gy + shimmer_y
            
        elif mode == 'ERODE':
            # Edge breaking - displace based on local contrast/gradient
            # Sample luminance gradient using neighbors
            patch = self.working[y1:y2, x1:x2]
            lum = 0.299 * patch[:,:,0] + 0.587 * patch[:,:,1] + 0.114 * patch[:,:,2]
            
            # Compute gradient using Sobel-like kernel
            grad_x = np.zeros_like(lum)
            grad_y = np.zeros_like(lum)
            if lum.shape[0] > 2 and lum.shape[1] > 2:
                grad_x[1:-1, 1:-1] = lum[1:-1, 2:] - lum[1:-1, :-2]
                grad_y[1:-1, 1:-1] = lum[2:, 1:-1] - lum[:-2, 1:-1]
            
            # Displace along gradient (erodes edges)
            erode_strength = weights * radius * 0.3
            src_x = gx + grad_x * erode_strength
            src_y = gy + grad_y * erode_strength
            
        elif mode == 'CREASE':
            # Sharp linear deformation - crease along drag direction
            # Project onto drag direction for sharp line effect
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx = dx / drag_len
            drag_ny = dy / drag_len
            
            # Distance along drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            
            # Sharp crease using tanh for step-like transition
            crease = np.tanh(proj * 0.2) * weights * radius * 0.2
            
            # Displace perpendicular to drag
            src_x = gx - drag_ny * crease
            src_y = gy + drag_nx * crease
            
        elif mode == 'BRISTLE':
            # Directional streaks and striations
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Parallel striations 
            striation = np.sin((rel_x * drag_ny - rel_y * drag_nx) * 0.5) * 0.5 + 0.5
            offset_x = dx * weights * striation * 0.3
            offset_y = dy * weights * striation * 0.3
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'DRYPULL':
            # Broken dry-brush skipped pixels
            skip = ((gx.astype(np.int32) + gy.astype(np.int32)) % 3 != 0).astype(np.float32)
            src_x = gx - dx * weights * skip * 0.3
            src_y = gy - dy * weights * skip * 0.3
            
        elif mode == 'BLOOM':
            # Soft expanding overlap like petals
            dist = np.sqrt(dist_sq) + 0.001
            expand = weights * radius * 0.2
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * expand
            src_y = gy - norm_y * expand
            
        elif mode == 'INFLATE':
            # Organic uneven bulging with noise
            dist = np.sqrt(dist_sq) + 0.001
            noise = np.sin(gx * 0.1 + gy * 0.13) * np.cos(gx * 0.07 - gy * 0.11)
            bulge = weights * radius * 0.15 * (1 + noise * 0.5)
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * bulge
            src_y = gy - norm_y * bulge
            
        elif mode == 'LIQUIFY':
            # Fluid-like warp deformation
            dist = np.sqrt(dist_sq) + 0.001
            fluid = np.sin(dist * 0.1) * weights * radius * 0.15
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - dx * weights * 0.3 + norm_y * fluid
            src_y = gy - dy * weights * 0.3 - norm_x * fluid
            
        elif mode == 'SPIRAL':
            # Swirling vortex distortion
            dist = np.sqrt(dist_sq) + 0.001
            spiral_angle = weights * 2.0  # Strong spiral
            cos_s = np.cos(spiral_angle)
            sin_s = np.sin(spiral_angle)
            src_x = p_x + rel_x * cos_s - rel_y * sin_s
            src_y = p_y + rel_x * sin_s + rel_y * cos_s
            
        elif mode == 'STRETCH':
            # Directional elongation
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Project onto drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            stretch = weights * proj * 0.3
            src_x = gx - drag_nx * stretch
            src_y = gy - drag_ny * stretch
            
        elif mode == 'PIXELATE':
            # Pixelated mosaic effect - snap to grid
            grid_size = max(2, int(radius * 0.1 * strength) + 1)
            grid_x = (gx // grid_size) * grid_size + grid_size / 2
            grid_y = (gy // grid_size) * grid_size + grid_size / 2
            blend = weights
            src_x = gx * (1 - blend) + grid_x * blend
            src_y = gy * (1 - blend) + grid_y * blend
            
        elif mode == 'GLITCH':
            # Digital scan line displacement
            line_height = 3
            line_idx = (gy / line_height).astype(np.int32)
            offset = np.sin(line_idx * 3.7) * weights * radius * 0.3
            src_x = gx + offset
            src_y = gy
            
        else:
            return
        
        # Clamp source coordinates
        src_x = np.clip(src_x, 0, self.width - 1.001)
        src_y = np.clip(src_y, 0, self.height - 1.001)
        
        # Bilinear interpolation for quality
        x0 = src_x.astype(np.int32)
        y0 = src_y.astype(np.int32)
        x1i = np.minimum(x0 + 1, self.width - 1)
        y1i = np.minimum(y0 + 1, self.height - 1)
        
        fx = (src_x - x0)[:, :, np.newaxis]
        fy = (src_y - y0)[:, :, np.newaxis]
        
        # Sample 4 corners
        p00 = self.working[y0, x0]
        p10 = self.working[y0, x1i]
        p01 = self.working[y1i, x0]
        p11 = self.working[y1i, x1i]
        
        # Bilinear interpolate
        sampled = (p00 * (1 - fx) * (1 - fy) +
                   p10 * fx * (1 - fy) +
                   p01 * (1 - fx) * fy +
                   p11 * fx * fy)
        
        # Apply with mask
        patch = self.working[y1:y2, x1:x2]
        mask_3d = mask[:, :, np.newaxis]
        patch[:] = np.where(mask_3d, sampled, patch)


# ============================================================
# Lattice Deformation Tool
# ============================================================

class IMAGE_EDIT_OT_lattice_deform(bpy.types.Operator):
    """Deform image using lattice grid with perspective or mesh mode"""
    bl_idname = "image_editor_plus.lattice_deform"
    bl_label = "Lattice Deform"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _original_pixels = None
    _working_pixels = None
    _width = 0
    _height = 0
    
    # Control points: grid [row][col] = [x, y]
    _control_points = []
    _original_grid = []
    
    # Interaction state
    _active_point = None
    _is_dragging = False
    _initialized = False
    
    HANDLE_SIZE = 10
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _init_grid(self, context):
        """Initialize control point grid based on resolution."""
        props = context.window_manager.imageeditorplus_properties
        mode = props.lattice_mode
        
        if mode == 'PERSPECTIVE':
            res_u, res_v = 2, 2
        else:
            res_u = props.lattice_resolution_u
            res_v = props.lattice_resolution_v
        
        self._control_points = []
        self._original_grid = []
        
        for j in range(res_v):
            row, orig_row = [], []
            for i in range(res_u):
                x = (i / (res_u - 1)) * (self._width - 1) if res_u > 1 else self._width / 2
                y = (j / (res_v - 1)) * (self._height - 1) if res_v > 1 else self._height / 2
                row.append([x, y])
                orig_row.append([x, y])
            self._control_points.append(row)
            self._original_grid.append(orig_row)
    
    def _screen_to_image(self, context, mx, my):
        """Convert screen to image pixel coordinates."""
        view2d = context.region.view2d
        img_x, img_y = view2d.region_to_view(mx, my)
        return img_x * self._width, img_y * self._height
    
    def _image_to_screen(self, context, ix, iy):
        """Convert image to screen coordinates."""
        nx = ix / self._width if self._width > 0 else 0
        ny = iy / self._height if self._height > 0 else 0
        return context.region.view2d.view_to_region(nx, ny)
    
    def _hit_test(self, context, mx, my):
        """Check if mouse is over a control point."""
        ix, iy = self._screen_to_image(context, mx, my)
        hs = self.HANDLE_SIZE * 2
        for j, row in enumerate(self._control_points):
            for i, pt in enumerate(row):
                if abs(ix - pt[0]) < hs and abs(iy - pt[1]) < hs:
                    return (j, i)
        return None
    
    def _find_homography(self, src, dst):
        """Compute 3x3 homography using DLT algorithm."""
        import numpy as np
        n = src.shape[0]
        A = np.zeros((2*n, 9), dtype=np.float64)
        for i in range(n):
            x, y = src[i]
            xp, yp = dst[i]
            A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]
    
    def _apply_perspective_warp(self):
        """Apply 4-point perspective transformation."""
        import numpy as np
        
        src = np.array([self._original_grid[0][0], self._original_grid[0][-1],
                        self._original_grid[-1][-1], self._original_grid[-1][0]], dtype=np.float32)
        dst = np.array([self._control_points[0][0], self._control_points[0][-1],
                        self._control_points[-1][-1], self._control_points[-1][0]], dtype=np.float32)
        
        H = self._find_homography(src, dst)
        H_inv = np.linalg.inv(H)
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        coords = np.stack([x, y, np.ones_like(x)], axis=-1)
        src_coords = np.einsum('ij,...j->...i', H_inv, coords)
        
        w = src_coords[..., 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        src_x = src_coords[..., 0] / w[..., 0]
        src_y = src_coords[..., 1] / w[..., 0]
        
        self._bilinear_sample(src_x, src_y)
    
    def _apply_mesh_warp(self):
        """Apply mesh deformation with bilinear cell interpolation."""
        import numpy as np
        
        rows, cols = len(self._control_points), len(self._control_points[0])
        if rows < 2 or cols < 2:
            return
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        src_x, src_y = np.copy(x), np.copy(y)
        
        for j in range(rows - 1):
            for i in range(cols - 1):
                o_tl, o_tr = self._original_grid[j][i], self._original_grid[j][i+1]
                o_bl, o_br = self._original_grid[j+1][i], self._original_grid[j+1][i+1]
                d_tl, d_tr = self._control_points[j][i], self._control_points[j][i+1]
                d_bl, d_br = self._control_points[j+1][i], self._control_points[j+1][i+1]
                
                mask = (x >= o_tl[0]) & (x < o_tr[0]) & (y >= o_tl[1]) & (y < o_bl[1])
                if not np.any(mask):
                    continue
                
                cw, ch = o_tr[0] - o_tl[0], o_bl[1] - o_tl[1]
                if cw < 1 or ch < 1:
                    continue
                
                u = (x[mask] - o_tl[0]) / cw
                v = (y[mask] - o_tl[1]) / ch
                
                src_x[mask] = (1-u)*(1-v)*d_tl[0] + u*(1-v)*d_tr[0] + (1-u)*v*d_bl[0] + u*v*d_br[0]
                src_y[mask] = (1-u)*(1-v)*d_tl[1] + u*(1-v)*d_tr[1] + (1-u)*v*d_bl[1] + u*v*d_br[1]
        
        self._bilinear_sample(src_x, src_y)
    
    def _bilinear_sample(self, src_x, src_y):
        """Sample original pixels with bilinear interpolation."""
        import numpy as np
        src_x = np.clip(src_x, 0, self._width - 1.001)
        src_y = np.clip(src_y, 0, self._height - 1.001)
        
        x0, y0 = src_x.astype(np.int32), src_y.astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, self._width - 1), np.minimum(y0 + 1, self._height - 1)
        fx, fy = (src_x - x0)[:,:,np.newaxis], (src_y - y0)[:,:,np.newaxis]
        
        self._working_pixels[:] = (self._original_pixels[y0, x0] * (1-fx) * (1-fy) +
                                    self._original_pixels[y0, x1] * fx * (1-fy) +
                                    self._original_pixels[y1, x0] * (1-fx) * fy +
                                    self._original_pixels[y1, x1] * fx * fy)
    
    def _update_preview(self, context):
        """Apply deformation and update image preview."""
        props = context.window_manager.imageeditorplus_properties
        if props.lattice_mode == 'PERSPECTIVE':
            self._apply_perspective_warp()
        else:
            self._apply_mesh_warp()
        self._image.pixels.foreach_set(self._working_pixels.ravel())
        self._image.update()
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            utils.ImageUndoStack.get().push_state_from_numpy(self._image, self._original_pixels)
            self._image.pixels.foreach_set(self._working_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            self.report({'INFO'}, "Lattice deformation applied")
            return {'FINISHED'}
        
        if event.type == 'ESC' and event.value == 'PRESS':
            self._image.pixels.foreach_set(self._original_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            return {'CANCELLED'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hit = self._hit_test(context, mx, my)
                if hit:
                    self._active_point = hit
                    self._is_dragging = True
            elif event.value == 'RELEASE':
                self._is_dragging = False
                self._active_point = None
        
        elif event.type == 'MOUSEMOVE' and self._is_dragging and self._active_point:
            row, col = self._active_point
            ix, iy = self._screen_to_image(context, mx, my)
            ix = max(0, min(self._width - 1, ix))
            iy = max(0, min(self._height - 1, iy))
            self._control_points[row][col] = [ix, iy]
            self._update_preview(context)
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        import numpy as np
        
        self._image = context.space_data.image
        self._width, self._height = self._image.size
        
        if self._width < 2 or self._height < 2:
            self.report({'ERROR'}, "Image too small")
            return {'CANCELLED'}
        
        pixels = np.zeros(self._width * self._height * 4, dtype=np.float32)
        self._image.pixels.foreach_get(pixels)
        self._original_pixels = pixels.reshape((self._height, self._width, 4))
        self._working_pixels = np.copy(self._original_pixels)
        
        self._init_grid(context)
        self._initialized = True
        
        self._draw_handler = context.space_data.draw_handler_add(
            draw_lattice_overlay, (self, context), 'WINDOW', 'POST_PIXEL')
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def _cleanup(self, context):
        if self._draw_handler:
            context.space_data.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._initialized = False


def draw_lattice_overlay(op, context):
    """Draw lattice grid and control points."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    
    if not op._initialized or not op._control_points:
        return
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(1.5)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    rows = len(op._control_points)
    cols = len(op._control_points[0]) if rows else 0
    
    # Grid lines
    lines = []
    for j in range(rows):
        for i in range(cols - 1):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j][i+1])
            lines.extend([s1, s2])
    for j in range(rows - 1):
        for i in range(cols):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j+1][i])
            lines.extend([s1, s2])
    
    if lines:
        shader.uniform_float("color", (0.4, 0.7, 1.0, 0.7))
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
    
    # Control points
    pts = [op._image_to_screen(context, *op._control_points[j][i]) 
           for j in range(rows) for i in range(cols)]
    if pts:
        gpu.state.point_size_set(10.0)
        shader.uniform_float("color", (0.2, 0.5, 0.9, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
        gpu.state.point_size_set(6.0)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
    
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)
