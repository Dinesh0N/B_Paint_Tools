import os
import array
import platform
import math
import bpy
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from gpu.types import GPUOffScreen


# ----------------------------
# Fonts
# ----------------------------

def get_custom_font_dirs():
    """Get custom font directories from addon preferences."""
    custom_dirs = []
    try:
        # Need to get the addon package name
        package_name = __package__ if __package__ else "TextTex"
        if package_name in bpy.context.preferences.addons:
            prefs = bpy.context.preferences.addons[package_name].preferences
            if hasattr(prefs, 'custom_font_paths'):
                for item in prefs.custom_font_paths:
                    if item.path and os.path.exists(bpy.path.abspath(item.path)):
                        custom_dirs.append(bpy.path.abspath(item.path))
    except Exception:
        pass
    return custom_dirs

def load_custom_fonts_to_blender():
    """Scan custom directories and load fonts into bpy.data.fonts so they appear in the UI."""
    stats = {"loaded": 0, "existing": 0, "failed": 0}
    font_dirs = get_custom_font_dirs()
    
    existing_paths = {f.filepath for f in bpy.data.fonts}
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, _, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        full_path = os.path.join(root, f)
                        if full_path in existing_paths:
                            stats["existing"] += 1
                            continue
                            
                        try:
                            bpy.data.fonts.load(full_path, check_existing=True)
                            stats["loaded"] += 1
                        except Exception as e:
                            print(f"Failed to load font {full_path}: {e}")
                            stats["failed"] += 1
    
    print(f"[TextTool] Custom fonts: {stats['loaded']} loaded, {stats['existing']} skipped, {stats['failed']} failed.")
    return stats


# ----------------------------
# Loaded Font Cache (blf font IDs)
# ----------------------------
_blf_font_cache = {}  # font_path -> font_id

def _get_blf_font_id(font_path):
    """Get or load a font using blf, returns font_id."""
    global _blf_font_cache
    
    if font_path in _blf_font_cache:
        return _blf_font_cache[font_path]
    
    font_id = 0  # Default font
    if font_path and os.path.exists(font_path):
        try:
            font_id = blf.load(font_path)
            if font_id == -1:
                font_id = 0
        except Exception as e:
            print(f"[TextTool] Failed to load font {font_path}: {e}")
            font_id = 0
    
    _blf_font_cache[font_path] = font_id
    return font_id


def reset_font_cache():
    """Clear the font cache so fonts are reloaded on next use."""
    global _blf_font_cache
    _blf_font_cache.clear()


# ----------------------------
# Gradient Node Storage
# ----------------------------
def get_gradient_node(create_if_missing=True):
    """Get or create a hidden node tree with a Color Ramp node.
    
    Args:
        create_if_missing: If False, returns None if node doesn't exist.
                          Set to False when called from draw callbacks.
    """
    tree_name = ".TextTool_Gradient_Storage"
    node_name = "Gradient_Ramp"
    
    # Check if tree exists
    if tree_name not in bpy.data.node_groups:
        if not create_if_missing:
            return None
        # Create new node group (shader type)
        tree = bpy.data.node_groups.new(tree_name, 'ShaderNodeTree')
        tree.use_fake_user = True  # Ensure it persists
    else:
        tree = bpy.data.node_groups[tree_name]
    
    # Check if node exists
    if node_name not in tree.nodes:
        if not create_if_missing:
            return None
        # Find existing color ramp node by type (in case name wasn't set)
        for node in tree.nodes:
            if node.type == 'VALTORGB':
                return node
        # Create new node
        node = tree.nodes.new('ShaderNodeValToRGB')
        node.name = node_name
        node.label = "Gradient"
    else:
        node = tree.nodes[node_name]
        
    return node


def get_gradient_lut(node, samples=256):
    """Evaluate a Color Ramp node into a LUT of RGBA tuples."""
    if not node or not hasattr(node, "color_ramp"):
        return []
    
    ramp = node.color_ramp
    lut = []
    # Optimization: If ramp has only 1 element, just return that.
    # But usually it has 2.
    
    # Evaluate at regular intervals
    step = 1.0 / (samples - 1)
    for i in range(samples):
        # clamp position 0-1
        pos = min(1.0, i * step)
        # evaluate returns a Color object (r,g,b,a)
        # We need tuple
        c = ramp.evaluate(pos)
        lut.append((c[0], c[1], c[2], c[3]))
        
    return lut


# ----------------------------
# Native Blender Font Manager
# ----------------------------
class FontManager:
    @staticmethod
    def create_text_image(text, font_path, font_size, color, width=None, height=None, rotation_degrees=0.0, gradient_lut=None, outline_info=None, alignment='CENTER', line_spacing=1.2):
        """Render text to pixel buffer using Blender's native blf and GPUOffScreen.
        
        Args:
            text: Text string to render (can contain newlines for multi-line)
            font_path: Path to the font file
            font_size: Font size in pixels
            color: Base RGBA color tuple (used when gradient_lut is None)
            width: Optional canvas width
            height: Optional canvas height
            rotation_degrees: Rotation angle in degrees
            gradient_lut: Optional list of RGBA tuples (Look-Up Table) for gradient.
            outline_info: Optional dict with outline settings.
            alignment: Text alignment: 'LEFT', 'CENTER', 'RIGHT', or 'JUSTIFY'
            line_spacing: Line spacing multiplier (1.0 = normal, 1.5 = 150%)
        
        Returns: (pixels_list, (width, height)) or (None, None) on failure
        """
        if not text:
            return None, None
        
        try:
            font_id = _get_blf_font_id(font_path)
            
            # Set font size
            blf.size(font_id, font_size)
            
            # Split text into lines for multi-line support
            lines = text.split('\n')
            
            # Calculate dimensions for each line and find max width
            line_heights = []
            line_widths = []
            for line in lines:
                if line.strip():  # Non-empty line
                    w, h = blf.dimensions(font_id, line)
                else:  # Empty line - use height of a space character
                    _, h = blf.dimensions(font_id, " ")
                    w = 0
                line_widths.append(w)
                line_heights.append(h)
            
            # Use max width and sum of heights with line spacing
            single_line_height = max(line_heights) if line_heights else font_size
            text_width = max(line_widths) if line_widths else 0
            text_height = single_line_height * line_spacing * len(lines)
            
            # Store for later use in rendering
            _lines_data = {
                'lines': lines,
                'line_widths': line_widths,
                'single_line_height': single_line_height,
                'line_spacing': line_spacing,
                'alignment': alignment,
                'text_width': text_width
            }
            
            # Add padding - extra for outline if enabled
            outline_size = 0
            if outline_info and outline_info.get('enabled'):
                outline_size = outline_info.get('size', 2)
            
            padding = 10 + outline_size
            base_width = int(text_width + padding * 2)
            base_height = int(text_height + padding * 2)
            
            # For rotation, we need a larger canvas to fit rotated text
            if rotation_degrees != 0.0:
                angle_rad = math.radians(abs(rotation_degrees))
                # Calculate bounding box of rotated rectangle
                cos_a = abs(math.cos(angle_rad))
                sin_a = abs(math.sin(angle_rad))
                rotated_width = int(base_width * cos_a + base_height * sin_a) + padding * 2
                rotated_height = int(base_width * sin_a + base_height * cos_a) + padding * 2
                canvas_width = max(rotated_width, base_width)
                canvas_height = max(rotated_height, base_height)
            else:
                canvas_width = base_width
                canvas_height = base_height
            
            # Ensure minimum size
            canvas_width = max(2, canvas_width)
            canvas_height = max(2, canvas_height)
            
            # Create offscreen buffer
            offscreen = GPUOffScreen(canvas_width, canvas_height)
            
            text_pixels = []
            outline_pixels = []
            has_outline = outline_info and outline_info.get('enabled')
            
            with offscreen.bind():
                # Get framebuffer
                fb = gpu.state.active_framebuffer_get()
                
                # Setup 2D orthographic projection
                from mathutils import Matrix
                sx = 2.0 / canvas_width
                sy = 2.0 / canvas_height
                proj = Matrix((
                    (sx, 0, 0, -1),
                    (0, sy, 0, -1),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1)
                ))
                
                gpu.matrix.push()
                gpu.matrix.push_projection()
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(proj)
                gpu.state.blend_set('ALPHA')
                
                # Calculate geometry
                cx, cy = canvas_width / 2, canvas_height / 2
                if rotation_degrees != 0.0:
                    angle_rad = math.radians(rotation_degrees)
                    blf.enable(font_id, blf.ROTATION)
                    blf.rotation(font_id, angle_rad)
                    
                    cos_r = math.cos(angle_rad)
                    sin_r = math.sin(angle_rad)
                    offset_x = text_width / 2
                    offset_y = text_height / 2
                    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
                    rotated_offset_y = offset_x * sin_r + offset_y * cos_r
                    base_x = cx - rotated_offset_x
                    base_y = cy - rotated_offset_y
                else:
                    base_x = (canvas_width - text_width) / 2
                    base_y = (canvas_height - text_height) / 2
                
                # --- PASS 1: Outline (if enabled) ---
                # We render to a separate buffer first? Or simpler: 
                # If we want gradient on text but NOT outline, we must process text separately.
                # So let's render text first to get the main shape for gradient.
                # Then render outline to get outline pixels.
                # Then composite.
                
                # --- PASS 1: Text Body ---
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                
                # Set text color
                if gradient_lut:
                    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
                else:
                    r, g, b = color[0], color[1], color[2]
                    a = color[3] if len(color) > 3 else 1.0
                    blf.color(font_id, r, g, b, a)
                
                # Draw each line (multi-line support)
                line_height = _lines_data['single_line_height'] * _lines_data['line_spacing']
                total_text_width = _lines_data['text_width']
                align = _lines_data['alignment']
                
                for i, line in enumerate(_lines_data['lines']):
                    if not line.strip():
                        continue  # Skip empty lines (but they still take vertical space)
                    
                    # Calculate Y position for this line (top line first)
                    line_y = base_y + text_height - (i + 1) * line_height + line_height * 0.2
                    
                    # Calculate X position based on alignment
                    line_w = _lines_data['line_widths'][i]
                    if align == 'LEFT':
                        line_x = base_x
                    elif align == 'RIGHT':
                        line_x = base_x + total_text_width - line_w
                    elif align == 'JUSTIFY' and len(_lines_data['lines']) > 1 and i < len(_lines_data['lines']) - 1:
                        # Justify: stretch to fill width (except last line)
                        line_x = base_x  # Start from left, word spacing handled by blf
                    else:  # CENTER (default)
                        line_x = base_x + (total_text_width - line_w) / 2
                    
                    blf.position(font_id, line_x, line_y, 0)
                    blf.draw(font_id, line)
                
                buffer_text = fb.read_color(0, 0, canvas_width, canvas_height, 4, 0, 'FLOAT')
                # Convert to flat list
                for row in buffer_text:
                    for pixel in row:
                        text_pixels.extend(pixel)
                
                # --- PASS 2: Outline (only if enabled) ---
                if has_outline:
                    fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                    
                    outline_color = outline_info.get('color', (0, 0, 0, 1))
                    outline_sz = outline_info.get('size', 2)
                    
                    or_, og, ob = outline_color[0], outline_color[1], outline_color[2]
                    oa = outline_color[3] if len(outline_color) > 3 else 1.0
                    blf.color(font_id, or_, og, ob, oa)
                    
                    # Draw outline for each line (multi-line support)
                    line_height = _lines_data['single_line_height'] * _lines_data['line_spacing']
                    total_text_width = _lines_data['text_width']
                    align = _lines_data['alignment']
                    
                    for i, line in enumerate(_lines_data['lines']):
                        if not line.strip():
                            continue
                        
                        line_y = base_y + text_height - (i + 1) * line_height + line_height * 0.2
                        line_w = _lines_data['line_widths'][i]
                        
                        # Calculate X position based on alignment
                        if align == 'LEFT':
                            line_x = base_x
                        elif align == 'RIGHT':
                            line_x = base_x + total_text_width - line_w
                        elif align == 'JUSTIFY' and len(_lines_data['lines']) > 1 and i < len(_lines_data['lines']) - 1:
                            line_x = base_x
                        else:  # CENTER (default)
                            line_x = base_x + (total_text_width - line_w) / 2
                        
                        # Draw outline circular pattern
                        for angle in range(0, 360, 30):
                            rad = math.radians(angle)
                            ox = math.cos(rad) * outline_sz
                            oy = math.sin(rad) * outline_sz
                            blf.position(font_id, line_x + ox, line_y + oy, 0)
                            blf.draw(font_id, line)
                        
                        # Cardinal directions
                        for ox, oy in [(outline_sz, 0), (-outline_sz, 0), (0, outline_sz), (0, -outline_sz)]:
                            blf.position(font_id, line_x + ox, line_y + oy, 0)
                            blf.draw(font_id, line)
                        
                    buffer_outline = fb.read_color(0, 0, canvas_width, canvas_height, 4, 0, 'FLOAT')
                    for row in buffer_outline:
                        for pixel in row:
                            outline_pixels.extend(pixel)

                # Cleanup
                if rotation_degrees != 0.0:
                    blf.disable(font_id, blf.ROTATION)
                
                gpu.state.blend_set('NONE')
                gpu.matrix.pop_projection()
                gpu.matrix.pop()
            
            offscreen.free()
            
            # --- Post-Processing ---
            
            # 1. Apply Gradient to Text Body
            if gradient_lut:
                text_pixels = FontManager._apply_gradient(
                    text_pixels, canvas_width, canvas_height, gradient_lut
                )
            
            # 2. Composite (Text over Outline)
            if has_outline and outline_pixels:
                final_pixels = FontManager._composite_layers(text_pixels, outline_pixels)
            else:
                final_pixels = text_pixels
            
            return final_pixels, (canvas_width, canvas_height)
            
        except Exception as e:
            print(f"[TextTool] Render error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    @staticmethod
    def _composite_layers(fg_pixels, bg_pixels):
        """Composite foreground (text) over background (outline)."""
        # Both are flat lists of RGBA floats
        count = len(fg_pixels)
        if len(bg_pixels) != count:
            return fg_pixels 
            
        result = [0.0] * count
        
        # Simple Over operator:
        # out = fg + bg * (1 - fg.a)
        # Note: Interpreting pixels as straight alpha (Blender default)
        
        for i in range(0, count, 4):
            fr, fg, fb, fa = fg_pixels[i], fg_pixels[i+1], fg_pixels[i+2], fg_pixels[i+3]
            br, bg, bb, ba = bg_pixels[i], bg_pixels[i+1], bg_pixels[i+2], bg_pixels[i+3]
            
            inv_fa = 1.0 - fa
            
            result[i]   = fr * fa + br * inv_fa   # Using premul logic or straight?
            # Actually fb.read_color returns straight alpha usually unless we clear with 0.
            # But standard composition equation: 
            #   out_a = src_a + dst_a * (1 - src_a)
            #   out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1 - src_a)) / out_a
            # Simpler approximation for now (standard mix):
            
            result[i]   = fr * fa + br * ba * inv_fa # Premultiplied output attempt?
            # Let's stick to standard straight alpha mix:
            # Output should ideally be straight RGB and A.
            
            # Standard "Over":
            # out_a = fa + ba * (1 - fa)
            out_a = fa + ba * inv_fa
            
            # Avoid divide by zero
            if out_a > 0:
                # out_rgb = (fr * fa + br * ba * inv_fa) / out_a
                # Wait, if we assume straight inputs:
                # contributions:
                # fg contributes (fr, fg, fb) * fa
                # bg contributes (br, bg, bb) * ba * (1 - fa)
                
                result[i]   = (fr * fa + br * ba * inv_fa) / out_a
                result[i+1] = (fg * fa + bg * ba * inv_fa) / out_a
                result[i+2] = (fb * fa + bb * ba * inv_fa) / out_a
                result[i+3] = out_a
            else:
                result[i] = 0.0
                result[i+1] = 0.0
                result[i+2] = 0.0
                result[i+3] = 0.0
                
        return result

    
    @staticmethod
    def _apply_gradient(pixels, width, height, gradient_data):
        """Apply gradient colors to rendered text pixels.
        
        Args:
            pixels: Flat list of RGBA pixel values
            width: Image width
            height: Image height
            gradient_data: Dict with {'type': str, 'lut': list of RGBA, 'angle': float, 'font_rotation': float}
        """
        gradient_type = gradient_data.get('type', 'LINEAR')
        lut = gradient_data.get('lut', [])
        gradient_angle = gradient_data.get('angle', 0.0)
        font_rotation = gradient_data.get('font_rotation', 0.0)
        lut_len = len(lut)
        
        if lut_len < 2:
            return pixels 
        
        # --- Step 1: Find actual text bounding box (non-transparent pixels) ---
        min_x, max_x = width, 0
        min_y, max_y = height, 0
        
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                if pixels[idx + 3] > 0:  # Has alpha
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
        
        # If no text pixels found, return unchanged
        if max_x < min_x or max_y < min_y:
            return pixels
            
        # Text bounding box center (in canvas coordinates)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        # --- Step 2: Precompute rotation constants ---
        # To transform canvas coordinates back to "unrotated text" coordinates,
        # we rotate by the NEGATIVE of font_rotation
        font_rad = math.radians(-font_rotation)
        font_cos = math.cos(font_rad)
        font_sin = math.sin(font_rad)
        
        # Now we need to find the text bounds in the UNROTATED space
        # by transforming all text pixel positions
        local_min_x, local_max_x = float('inf'), float('-inf')
        local_min_y, local_max_y = float('inf'), float('-inf')
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                idx = (y * width + x) * 4
                if pixels[idx + 3] > 0:
                    # Transform to local (unrotated) coordinates
                    dx = x - cx
                    dy = y - cy
                    local_x = dx * font_cos - dy * font_sin
                    local_y = dx * font_sin + dy * font_cos
                    
                    if local_x < local_min_x: local_min_x = local_x
                    if local_x > local_max_x: local_max_x = local_x
                    if local_y < local_min_y: local_min_y = local_y
                    if local_y > local_max_y: local_max_y = local_y
        
        # Local text dimensions
        local_width = local_max_x - local_min_x if local_max_x > local_min_x else 1.0
        local_height = local_max_y - local_min_y if local_max_y > local_min_y else 1.0
        local_cx = (local_min_x + local_max_x) / 2
        local_cy = (local_min_y + local_max_y) / 2
        
        # --- Step 3: Calculate gradient parameters in LOCAL space ---
        # Gradient angle is relative to the text's local coordinate system
        grad_rad = math.radians(gradient_angle)
        grad_cos = math.cos(grad_rad)
        grad_sin = math.sin(grad_rad)
        
        # For linear gradient, project local corners onto the gradient axis
        hw = local_width / 2
        hh = local_height / 2
        corners = [
            (-hw * grad_cos - -hh * grad_sin),
            ( hw * grad_cos - -hh * grad_sin),
            ( hw * grad_cos -  hh * grad_sin),
            (-hw * grad_cos -  hh * grad_sin),
        ]
        min_p = min(corners)
        max_p = max(corners)
        span = max_p - min_p if (max_p - min_p) > 0.001 else 1.0

        # Radial max dist
        max_dist = math.sqrt(hw * hw + hh * hh) if (hw > 0 and hh > 0) else 1.0

        # --- Step 4: Apply gradient ---
        result = list(pixels)
        
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                
                alpha = result[idx + 3]
                if alpha <= 0:
                    continue
                
                # Transform pixel to LOCAL coordinates (undo font rotation)
                dx = x - cx
                dy = y - cy
                local_x = dx * font_cos - dy * font_sin
                local_y = dx * font_sin + dy * font_cos
                
                # Offset from local center
                lx = local_x - local_cx
                ly = local_y - local_cy
                
                # Calculate gradient factor
                if gradient_type == 'LINEAR':
                    rot_x = lx * grad_cos + ly * grad_sin
                    t = (rot_x - min_p) / span
                else:  # RADIAL
                    dist = math.sqrt(lx * lx + ly * ly)
                    t = dist / max_dist
                
                # Clamp t
                t = max(0.0, min(1.0, t))
                
                # Sample LUT
                lut_index = int(t * (lut_len - 1))
                color = lut[lut_index]
                
                # Apply gradient color while preserving luminance and alpha from original
                orig_lum = result[idx]  # Original was rendered white
                
                result[idx] = color[0] * orig_lum
                result[idx + 1] = color[1] * orig_lum
                result[idx + 2] = color[2] * orig_lum
                # Alpha stays the same
        
        return result



def blend_pixel(pixels, idx, tr, tg, tb, ta, mode):
    """
    Blend source pixel (tr, tg, tb, ta) onto destination pixels[idx] using 'mode'.
    Modes correspond to Blender Brush Blend modes.
    """
    # Destination pixel
    dr = pixels[idx]
    dg = pixels[idx + 1]
    db = pixels[idx + 2]
    da = pixels[idx + 3]

    # Pre-calculations
    # For standard alpha blending ('MIX'), we typically use:
    # out = src * src_a + dst * (1 - src_a)
    # But here 'tr, tg, tb' are straight color, 'ta' is alpha. 
    # 'pixels' is presumably premultiplied or straight? 
    # Blender Internal Images are generally Straight Alpha (unassociated).
    # So we do mixing in straight space usually.
    
    out_r, out_g, out_b, out_a = dr, dg, db, da

    if mode == 'MIX':
        # Standard Alpha Blending (Source Over)
        # out = col * alpha + dst * (1 - alpha)
        # alpha = src_a + dst_a * (1 - src_a)
        inv_ta = 1.0 - ta
        out_r = tr * ta + dr * inv_ta
        out_g = tg * ta + dg * inv_ta
        out_b = tb * ta + db * inv_ta
        out_a = ta + da * inv_ta

    elif mode == 'ADD':
        # Additive blending
        # out = dst + src * alpha
        out_r = min(1.0, dr + tr * ta)
        out_g = min(1.0, dg + tg * ta)
        out_b = min(1.0, db + tb * ta)
        out_a = min(1.0, da + ta) # ? Actually usually alpha just accumulates or stays max.

    elif mode == 'SUBTRACT':
        # Subtractive
        out_r = max(0.0, dr - tr * ta)
        out_g = max(0.0, dg - tg * ta)
        out_b = max(0.0, db - tb * ta)
        out_a = da # Alpha generally unaffected or mixed? Let's keep destination alpha.

    elif mode == 'MULTIPLY':
        # Multiply
        # out = dst * (1 - alpha) + (dst * src) * alpha
        # = dst * (1 - alpha + src * alpha)
        # Standard multiply blend in composition:
        inv_ta = 1.0 - ta
        out_r = dr * (inv_ta + tr * ta)
        out_g = dg * (inv_ta + tg * ta)
        out_b = db * (inv_ta + tb * ta)
        out_a = da # Usually alpha shouldn't reduce? 'Mix' logic for alpha: out_a = ta + da*(1-ta)
        out_a = ta + da * inv_ta

    elif mode == 'LIGHTEN':
        # Lighten: max(dst, src)
        # Mixed with alpha
        target_r = max(dr, tr)
        target_g = max(dg, tg)
        target_b = max(db, tb)
        inv_ta = 1.0 - ta
        out_r = target_r * ta + dr * inv_ta
        out_g = target_g * ta + dg * inv_ta
        out_b = target_b * ta + db * inv_ta
        out_a = ta + da * inv_ta

    elif mode == 'DARKEN':
        # Darken: min(dst, src)
        target_r = min(dr, tr)
        target_g = min(dg, tg)
        target_b = min(db, tb)
        inv_ta = 1.0 - ta
        out_r = target_r * ta + dr * inv_ta
        out_g = target_g * ta + dg * inv_ta
        out_b = target_b * ta + db * inv_ta
        out_a = ta + da * inv_ta
        
    elif mode == 'ERASE_ALPHA':
        # Erase Alpha: Reduce dst alpha by src alpha
        out_a = max(0.0, da - ta)
        # Color remains destination color (but will be invisible if alpha 0)
    
    elif mode == 'ADD_ALPHA':
        # Add Alpha
        out_a = min(1.0, da + ta)
    
    else:
        # Fallback to MIX
        inv_ta = 1.0 - ta
        out_r = tr * ta + dr * inv_ta
        out_g = tg * ta + dg * inv_ta
        out_b = tb * ta + db * inv_ta
        out_a = ta + da * inv_ta

    pixels[idx] = out_r
    pixels[idx + 1] = out_g
    pixels[idx + 2] = out_b
    pixels[idx + 3] = out_a

# ----------------------------
# Texture Refresh Helper
# ----------------------------
def force_texture_refresh(context, image):
    """Force the 3D viewport to refresh the texture after modification."""
    if not image:
        return

    image.update()
    
    # Toggle image node property to force material update
    # This acts as a 'touch' to tell the dependency graph something changed on the node
    # which usually triggers a GPU texture reload for the viewport.
    
    # Strategy: Find materials using this image and toggle interpolation
    # We scan all materials in data to cover all objects.
    # Note: Scanning all materials might be slow in huge scenes.
    # Optimization: Scan only materials on visible objects or just the active object if possible.
    # But for Image Editor tool, we don't know which object uses it.
    # Let's try to find usage in current context if possible, or scan common usage.
    
    # A safer/faster approach for specific context:
    # If in 3D Viewport, we know the active object.
    # If in Image Editor, we assume user wants to see it on objects in the scene.
    
    # Let's iterate over visible objects in the current view layer.
    if context.view_layer:
        for obj in context.view_layer.objects.values():
            if obj.type == 'MESH' and obj.visible_get():
                if obj.active_material and obj.active_material.use_nodes:
                    mat = obj.active_material
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image == image:
                            current = node.interpolation
                            # Toggle
                            node.interpolation = 'Closest' if current != 'Closest' else 'Linear'
                            node.interpolation = current
                            break  # Found the node, move to next object
    
    # Tag all 3D viewports for redraw
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

# ----------------------------
# Shared State
# ----------------------------
cursor_pos = None
show_cursor = False
cursor_pixel_scale = 1.0

# Gradient Tool preview state
gradient_preview_start = None  # (x, y) screen coordinates
gradient_preview_end = None    # (x, y) screen coordinates

# Crop Tool preview state
crop_preview_start = None  # (x, y) screen coordinates
crop_preview_end = None    # (x, y) screen coordinates

# Clone Tool state
clone_source_pos = None    # (x, y) screen coordinates - source point
clone_cursor_pos = None    # (x, y) screen coordinates - current brush position
clone_source_set = False   # Whether source has been set



# ----------------------------
# Image Undo/Redo Stack
# ----------------------------
class ImageUndoStack:
    """Manages undo/redo history for image pixel modifications.
    
    Blender's built-in undo system does NOT track direct pixel modifications,
    so we need a custom undo stack for text paint operations.
    """
    
    _instance = None
    
    def __init__(self, max_undo_steps=20):
        # Dictionary: image_name -> {"undo": [...], "redo": [...]}
        self._stacks = {}
        self.max_undo_steps = max_undo_steps
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ImageUndoStack()
        return cls._instance
    
    def _get_stack(self, image_name):
        """Get or create the undo/redo stack for an image."""
        if image_name not in self._stacks:
            self._stacks[image_name] = {"undo": [], "redo": []}
        return self._stacks[image_name]
    
    def push_state(self, image):
        """Save current image state before modification."""
        if image is None:
            return
        
        stack = self._get_stack(image.name)
        
        # Save current pixels as an array (much faster than list)
        width, height = image.size
        num_pixels = width * height * 4
        
        # Create float array
        pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(pixels)
        
        stack["undo"].append({
            "pixels": pixels,
            "size": (width, height)
        })
        
        # Clear redo stack when new action is performed
        stack["redo"].clear()
        
        # Limit undo history size
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def push_state_from_array(self, image, pixels_array):
        """Save pre-cached pixel array as undo state (for realtime preview)."""
        if image is None or pixels_array is None:
            return
        
        stack = self._get_stack(image.name)
        width, height = image.size
        
        # Make a copy of the array
        pixels_copy = array.array('f', pixels_array)
        
        stack["undo"].append({
            "pixels": pixels_copy,
            "size": (width, height)
        })
        
        stack["redo"].clear()
        
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def push_state_from_numpy(self, image, np_array):
        """Save numpy array as undo state efficiently (avoids slow .tolist())."""
        if image is None or np_array is None:
            return
        
        stack = self._get_stack(image.name)
        width, height = image.size
        
        # Efficient copy: use numpy's buffer interface
        # Flatten if needed and ensure contiguous float32
        flat = np_array.ravel().astype('float32', copy=False)
        pixels_copy = array.array('f')
        pixels_copy.frombytes(flat.tobytes())
        
        stack["undo"].append({
            "pixels": pixels_copy,
            "size": (width, height)
        })
        
        stack["redo"].clear()
        
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def undo(self, image):
        """Restore previous image state (including size changes from crop)."""
        if image is None:
            return False
        
        stack = self._get_stack(image.name)
        
        if not stack["undo"]:
            return False
        
        # Save current state to redo stack
        width, height = image.size
        num_pixels = width * height * 4
        current_pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(current_pixels)
        
        stack["redo"].append({
            "pixels": current_pixels,
            "size": (width, height)
        })
        
        # Restore previous state
        state = stack["undo"].pop()
        old_width, old_height = state["size"]
        
        # Handle size change (from crop operations)
        if (old_width, old_height) != (width, height):
            image.scale(old_width, old_height)
        
        # Fast restore
        image.pixels.foreach_set(state["pixels"])
        image.update()
        return True
    
    def redo(self, image):
        """Restore next image state (including size changes from crop)."""
        if image is None:
            return False
        
        stack = self._get_stack(image.name)
        
        if not stack["redo"]:
            return False
        
        # Save current state to undo stack
        width, height = image.size
        num_pixels = width * height * 4
        current_pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(current_pixels)
        
        stack["undo"].append({
            "pixels": current_pixels,
            "size": (width, height)
        })
        
        # Restore next state
        state = stack["redo"].pop()
        new_width, new_height = state["size"]
        
        # Handle size change (from crop operations)
        if (new_width, new_height) != (width, height):
            image.scale(new_width, new_height)
        
        image.pixels.foreach_set(state["pixels"])
        image.update()
        return True
    
    def can_undo(self, image):
        if image is None:
            return False
        return len(self._get_stack(image.name)["undo"]) > 0
    
    def can_redo(self, image):
        if image is None:
            return False
        return len(self._get_stack(image.name)["redo"]) > 0
    
    def clear(self, image=None):
        if image is None:
            self._stacks.clear()
        elif image.name in self._stacks:
            del self._stacks[image.name]


# ----------------------------
# Text Preview Cache
# ----------------------------
class TextPreviewCache:
    """Cache for text preview to avoid re-rendering on every frame."""
    
    _instance = None
    PREVIEW_IMAGE_NAME = "_TextTool_Preview_"
    
    def __init__(self):
        self.blender_image = None
        self.gpu_texture = None
        self.preview_width = 0
        self.preview_height = 0
        # Cache key to detect when settings change
        self.cache_key = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = TextPreviewCache()
        return cls._instance
    
    def _make_cache_key(self, text, font_path, font_size, color, rotation, gradient_data):
        """Create a hashable key from current settings."""
        # Color is a tuple/list, convert to tuple for hashing
        color_key = tuple(round(c, 3) for c in color)
        
        # Hash gradient data
        grad_key = None
        if gradient_data:
            grad_key = (
                gradient_data.get('type', 'LINEAR'),
                tuple(tuple(round(c, 3) for c in col) for col in gradient_data.get('lut', [])),
                gradient_data.get('angle', 0.0)
            )
        
        return (text, font_path, font_size, color_key, round(rotation, 2), grad_key)
    
    def update_preview(self, text, font_path, font_size, color, rotation, gradient_data=None):
        """Update the preview texture if settings have changed."""
        if not text:
            return False
        
        new_key = self._make_cache_key(text, font_path, font_size, color, rotation, gradient_data)
        
        # Check if we need to regenerate
        if new_key == self.cache_key and self.gpu_texture is not None:
            return True  # Already up to date
        
        # Generate new preview image using native rendering
        rotation_degrees = math.degrees(rotation) if isinstance(rotation, float) else rotation
        
        pixels, size = FontManager.create_text_image(
            text, font_path, font_size, color, 
            rotation_degrees=rotation_degrees,
            gradient_lut=gradient_data
        )
        
        if pixels is None or size is None:
            self.invalidate()
            return False
        
        self.preview_width, self.preview_height = size
        
        # Create or update Blender image
        try:
            # Remove old preview image if exists with different size
            if self.PREVIEW_IMAGE_NAME in bpy.data.images:
                old_img = bpy.data.images[self.PREVIEW_IMAGE_NAME]
                if old_img.size[0] != self.preview_width or old_img.size[1] != self.preview_height:
                    bpy.data.images.remove(old_img)
                    self.blender_image = None
                    self.gpu_texture = None
            
            # Create new image if needed
            if self.PREVIEW_IMAGE_NAME not in bpy.data.images:
                self.blender_image = bpy.data.images.new(
                    self.PREVIEW_IMAGE_NAME,
                    width=self.preview_width,
                    height=self.preview_height,
                    alpha=True
                )
                self.blender_image.colorspace_settings.name = 'sRGB'
            else:
                self.blender_image = bpy.data.images[self.PREVIEW_IMAGE_NAME]
            
            # Set pixels directly (already in correct format from GPUOffScreen)
            # GPUOffScreen reads bottom-up which matches Blender's image format
            self.blender_image.pixels.foreach_set(pixels)
            self.blender_image.update()
            
            # Force recreate GPU texture after every pixel update
            # The texture must be recreated to reflect updated pixel data
            self.gpu_texture = gpu.texture.from_image(self.blender_image)
            
            self.cache_key = new_key
            return True
            
        except Exception as e:
            print(f"[TextTool] Failed to create preview texture: {e}")
            import traceback
            traceback.print_exc()
            self.invalidate()
            return False
    
    def invalidate(self):
        """Clear the cache."""
        self.gpu_texture = None
        self.preview_width = 0
        self.preview_height = 0
        self.cache_key = None
        # Don't remove the Blender image here to avoid issues during drawing
    
    def get_texture_and_size(self):
        """Return the GPU texture and its size."""
        return self.gpu_texture, self.preview_width, self.preview_height
    
    def cleanup(self):
        """Remove the preview image from Blender. Call on addon unregister."""
        if self.PREVIEW_IMAGE_NAME in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[self.PREVIEW_IMAGE_NAME])
        self.blender_image = None
        self.gpu_texture = None

# ============================================================
# Layer System
# ============================================================

import blf
import numpy as np
from . ui_renderer import UIRenderer

class LayerAreaSession:
    """Session state for a single Image Editor area."""
    def __init__(self):
        self._selections = []  # List of [[x1, y1], [x2, y2]] rectangles
        self._ellipses = []    # List of [[x1, y1], [x2, y2]] ellipses (bounding boxes)
        self._lassos = []      # List of [[x1,y1], [x2,y2], ...] polygon point lists
        # Negation shapes (for subtract mode - affects ALL selection types)
        self._neg_rects = []     # Rectangles to subtract
        self._neg_ellipses = []  # Ellipses to subtract
        self._neg_lassos = []    # Lassos to subtract
        self.selection_region = None
        self.ellipse_region = None  # Current drag region for ellipse tool
        self.lasso_points = None    # Current lasso points being drawn
        self.selecting = False
        self.selection_mode = 'SET'  # 'SET', 'ADD', 'SUBTRACT'
        self.selection_mask = None  # Numpy boolean mask for paint restriction
        self.layer_moving = False
        self.layer_rotating = False
        self.layer_scaling = False
        self.prevent_layer_update_event = False
        self.prev_image = None
        self.prev_image_width = 0
        self.prev_image_height = 0
        # Selection undo/redo history
        self._selection_history = []  # Stack of previous selection states
        self._selection_redo = []     # Stack of redo states
        self._max_history = 50        # Max undo history size
    
    def _get_selection_state(self):
        """Get current selection state as a dictionary."""
        import copy
        return {
            'selections': copy.deepcopy(self._selections),
            'ellipses': copy.deepcopy(self._ellipses),
            'lassos': copy.deepcopy(self._lassos),
            'neg_rects': copy.deepcopy(self._neg_rects),
            'neg_ellipses': copy.deepcopy(self._neg_ellipses),
            'neg_lassos': copy.deepcopy(self._neg_lassos),
        }
    
    def _set_selection_state(self, state):
        """Restore selection state from a dictionary."""
        import copy
        self._selections = copy.deepcopy(state.get('selections', []))
        self._ellipses = copy.deepcopy(state.get('ellipses', []))
        self._lassos = copy.deepcopy(state.get('lassos', []))
        self._neg_rects = copy.deepcopy(state.get('neg_rects', []))
        self._neg_ellipses = copy.deepcopy(state.get('neg_ellipses', []))
        self._neg_lassos = copy.deepcopy(state.get('neg_lassos', []))
        self.selection_mask = None
    
    def push_undo(self):
        """Push current selection state to undo history."""
        state = self._get_selection_state()
        self._selection_history.append(state)
        # Limit history size
        if len(self._selection_history) > self._max_history:
            self._selection_history.pop(0)
        # Clear redo stack when new action is performed
        self._selection_redo.clear()
    
    def undo_selection(self):
        """Undo last selection change. Returns True if successful."""
        if not self._selection_history:
            return False
        # Save current state to redo stack
        current = self._get_selection_state()
        self._selection_redo.append(current)
        # Restore previous state
        prev_state = self._selection_history.pop()
        self._set_selection_state(prev_state)
        return True
    
    def redo_selection(self):
        """Redo last undone selection change. Returns True if successful."""
        if not self._selection_redo:
            return False
        # Save current state to undo stack
        current = self._get_selection_state()
        self._selection_history.append(current)
        # Restore redo state
        redo_state = self._selection_redo.pop()
        self._set_selection_state(redo_state)
        return True
    
    @property
    def selection(self):
        """Get bounding box of all selections (backward compatibility)."""
        if not self._selections:
            return None
        # Compute bounding box of all selection rectangles
        min_x = min(s[0][0] for s in self._selections)
        min_y = min(s[0][1] for s in self._selections)
        max_x = max(s[1][0] for s in self._selections)
        max_y = max(s[1][1] for s in self._selections)
        return [[min_x, min_y], [max_x, max_y]]
    
    @selection.setter
    def selection(self, value):
        """Set selection (backward compatibility - replaces all selections)."""
        if value is None:
            self._selections = []
            self.selection_mask = None
        else:
            self._selections = [value]
    
    @property
    def selections(self):
        """Get list of all selection rectangles."""
        return self._selections
    
    def add_selection(self, rect):
        """Add a rectangle to selections (extend mode)."""
        self._selections.append(rect)
        self.selection_mask = None  # Invalidate mask
    
    def subtract_selection(self, rect):
        """Subtract a rectangle from selections.
        
        This computes the geometric difference between existing selections
        and the subtracted rectangle, potentially splitting rectangles.
        """
        if not self._selections:
            return
        
        sub_x1, sub_y1 = rect[0]
        sub_x2, sub_y2 = rect[1]
        
        new_selections = []
        for sel in self._selections:
            sel_x1, sel_y1 = sel[0]
            sel_x2, sel_y2 = sel[1]
            
            # Check if rectangles overlap
            if sub_x1 >= sel_x2 or sub_x2 <= sel_x1 or sub_y1 >= sel_y2 or sub_y2 <= sel_y1:
                # No overlap, keep original
                new_selections.append(sel)
            else:
                # Overlap - split into up to 4 rectangles
                # Top portion (above subtracted area)
                if sub_y1 > sel_y1:
                    new_selections.append([[sel_x1, sel_y1], [sel_x2, sub_y1]])
                # Bottom portion (below subtracted area)
                if sub_y2 < sel_y2:
                    new_selections.append([[sel_x1, sub_y2], [sel_x2, sel_y2]])
                # Left portion (within vertical overlap)
                top = max(sel_y1, sub_y1)
                bottom = min(sel_y2, sub_y2)
                if sub_x1 > sel_x1 and bottom > top:
                    new_selections.append([[sel_x1, top], [sub_x1, bottom]])
                # Right portion (within vertical overlap)
                if sub_x2 < sel_x2 and bottom > top:
                    new_selections.append([[sub_x2, top], [sel_x2, bottom]])
        
        self._selections = new_selections
        self.selection_mask = None
    
    @property
    def ellipses(self):
        """Get list of all ellipse selections (bounding boxes)."""
        return self._ellipses

    def add_ellipse(self, bbox):
        """Add an ellipse to selections (extend mode)."""
        self._ellipses.append(bbox)
        self.selection_mask = None

    @property
    def lassos(self):
        """Get list of all lasso selections (point lists)."""
        return self._lassos

    def add_lasso(self, points):
        """Add a lasso polygon to selections (extend mode)."""
        self._lassos.append(points)
        self.selection_mask = None

    def clear_selections(self):
        """Clear all selections."""
        self._selections = []
        self._ellipses = []
        self._lassos = []
        self._neg_rects = []
        self._neg_ellipses = []
        self._neg_lassos = []
        self.selection_mask = None

class LayerSession:
    """Global layer session state."""
    def __init__(self):
        self.icons = None
        self.keymaps = []
        self.ui_renderer = None
        self.copied_image_pixels = None
        self.copied_image_settings = None
        self.copied_layer_settings = None
        self.areas = {}

# Global layer session
_layer_session = LayerSession()

def layer_get_session():
    """Get the global layer session."""
    global _layer_session
    return _layer_session

def layer_get_area_session(context):
    """Get or create the area session for an Image Editor area."""
    global _layer_session
    area_session = _layer_session.areas.get(context.area, None)
    if not area_session:
        area_session = LayerAreaSession()
        _layer_session.areas[context.area] = area_session
    return area_session

def layer_draw_handler():
    """Draw handler for layer rendering in Image Editor."""
    global _layer_session
    context = bpy.context
    area_session = layer_get_area_session(context)
    info_text = None
    width, height, img_name = 0, 0, ''
    img = context.area.spaces.active.image
    if img:
        width, height = img.size[0], img.size[1]

    if img and (area_session.selections or area_session.selection_region or area_session.ellipses or area_session.ellipse_region or area_session.lassos or area_session.lasso_points):
        if not _layer_session.ui_renderer:
            _layer_session.ui_renderer = UIRenderer()
        
        # Draw current drag rectangle if selecting
        if area_session.selection_region and area_session.selecting:
            region_pos1, region_pos2 = area_session.selection_region
            region_size = [region_pos2[0] - region_pos1[0], region_pos2[1] - region_pos1[1]]
            _layer_session.ui_renderer.render_selection_frame(region_pos1, region_size)
        
        # Draw current drag ellipse if selecting
        if area_session.ellipse_region and area_session.selecting:
            region_pos1, region_pos2 = area_session.ellipse_region
            # Normalize for width/height
            x = min(region_pos1[0], region_pos2[0])
            y = min(region_pos1[1], region_pos2[1])
            w = abs(region_pos2[0] - region_pos1[0])
            h = abs(region_pos2[1] - region_pos1[1])
            _layer_session.ui_renderer.render_ellipse_selection((x, y), (w, h))
        
        # Draw current lasso points if selecting
        if area_session.lasso_points and area_session.selecting:
            _layer_session.ui_renderer.render_lasso_preview(area_session.lasso_points)
        
        # Draw ALL committed selections as unified merged contour (Krita-style)
        screen_rects = []
        screen_ellipses = []
        screen_lassos = []
        
        if area_session.selections:
            for selection in area_session.selections:
                view_x1 = selection[0][0] / width
                view_y1 = selection[0][1] / height
                view_x2 = selection[1][0] / width
                view_y2 = selection[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                screen_rects.append((region_pos1[0], region_pos1[1], region_pos2[0], region_pos2[1]))

        if area_session.ellipses:
            for selection in area_session.ellipses:
                view_x1 = selection[0][0] / width
                view_y1 = selection[0][1] / height
                view_x2 = selection[1][0] / width
                view_y2 = selection[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                
                x1 = min(region_pos1[0], region_pos2[0])
                y1 = min(region_pos1[1], region_pos2[1])
                x2 = max(region_pos1[0], region_pos2[0])
                y2 = max(region_pos1[1], region_pos2[1])
                screen_ellipses.append((x1, y1, x2, y2))
        
        if area_session.lassos:
            for lasso in area_session.lassos:
                screen_pts = []
                for pt in lasso:
                    vx = pt[0] / width
                    vy = pt[1] / height
                    sx, sy = context.region.view2d.view_to_region(vx, vy, clip=False)
                    screen_pts.append((sx, sy))
                if len(screen_pts) >= 3:
                    screen_lassos.append(screen_pts)
        
        # Collect negation shapes for subtraction
        neg_screen_ellipses = []
        neg_screen_lassos = []
        
        if area_session._neg_ellipses:
            for sel in area_session._neg_ellipses:
                view_x1 = sel[0][0] / width
                view_y1 = sel[0][1] / height
                view_x2 = sel[1][0] / width
                view_y2 = sel[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                
                x1 = min(region_pos1[0], region_pos2[0])
                y1 = min(region_pos1[1], region_pos2[1])
                x2 = max(region_pos1[0], region_pos2[0])
                y2 = max(region_pos1[1], region_pos2[1])
                neg_screen_ellipses.append((x1, y1, x2, y2))
        
        neg_screen_rects = []
        if area_session._neg_rects:
            for sel in area_session._neg_rects:
                view_x1 = sel[0][0] / width
                view_y1 = sel[0][1] / height
                view_x2 = sel[1][0] / width
                view_y2 = sel[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                neg_screen_rects.append((region_pos1[0], region_pos1[1], region_pos2[0], region_pos2[1]))
        
        if area_session._neg_lassos:
            for lasso in area_session._neg_lassos:
                screen_pts = []
                for pt in lasso:
                    vx = pt[0] / width
                    vy = pt[1] / height
                    sx, sy = context.region.view2d.view_to_region(vx, vy, clip=False)
                    screen_pts.append((sx, sy))
                if len(screen_pts) >= 3:
                    neg_screen_lassos.append(screen_pts)
        
        if screen_rects or screen_ellipses or screen_lassos or neg_screen_rects or neg_screen_ellipses or neg_screen_lassos:
            _layer_session.ui_renderer.render_merged_all(screen_rects, screen_ellipses, screen_lassos, neg_screen_rects, neg_screen_ellipses, neg_screen_lassos)

    if img:
        if not _layer_session.ui_renderer:
            _layer_session.ui_renderer = UIRenderer()
        img_props = img.imageeditorplus_properties
        selected_layer_index = img_props.selected_layer_index
        layers = img_props.layers
        # Blend mode string to integer mapping
        BLEND_MODE_MAP = {
            'MIX': 0, 'DARKEN': 1, 'MULTIPLY': 2, 'COLOR_BURN': 3,
            'LIGHTEN': 4, 'SCREEN': 5, 'COLOR_DODGE': 6, 'ADD': 7,
            'OVERLAY': 8, 'SOFT_LIGHT': 9, 'LINEAR_LIGHT': 10,
            'DIFFERENCE': 11, 'EXCLUSION': 12, 'SUBTRACT': 13, 'DIVIDE': 14,
            'HUE': 15, 'SATURATION': 16, 'COLOR': 17, 'VALUE': 18
        }
        
        for i, layer in reversed(list(enumerate(layers))):
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                layer_width, layer_height = layer_img.size[0], layer_img.size[1]
                layer_pos = layer.location
                layer_pos1 = [layer_pos[0], layer_pos[1] + layer_height]
                layer_pos2 = [layer_pos[0] + layer_width, layer_pos[1]]
                layer_view_x1 = layer_pos1[0] / width
                layer_view_y1 = 1.0 - layer_pos1[1] / height
                layer_region_pos1 = context.region.view2d.view_to_region(layer_view_x1, layer_view_y1, clip=False)
                layer_view_x2 = layer_pos2[0] / width
                layer_view_y2 = 1.0 - layer_pos2[1] / height
                layer_region_pos2 = context.region.view2d.view_to_region(layer_view_x2, layer_view_y2, clip=False)
                layer_region_size = [layer_region_pos2[0] - layer_region_pos1[0], layer_region_pos2[1] - layer_region_pos1[1]]
                if not layer.hide:
                    blend_mode_int = BLEND_MODE_MAP.get(layer.blend_mode, 0)
                    _layer_session.ui_renderer.render_image(layer_img, layer_region_pos1, layer_region_size, layer.rotation, layer.scale, layer.opacity, blend_mode_int)
                if i == selected_layer_index:
                    _layer_session.ui_renderer.render_selection_frame(layer_region_pos1, layer_region_size, layer.rotation, layer.scale)

    if img:
        if area_session.selection or area_session.selection_region:
            if area_session.prev_image:
                if img != area_session.prev_image:
                    layer_cancel_selection(context)
                elif width != area_session.prev_image_width or height != area_session.prev_image_height:
                    layer_crop_selection(context)

    area_session.prev_image = img
    area_session.prev_image_width = width
    area_session.prev_image_height = height

    if area_session.layer_moving or area_session.layer_rotating or area_session.layer_scaling:
        info_text = "LMB: Perform   RMB: Cancel"

    area_width = context.area.width
    if info_text:
        ui_scale = context.preferences.system.ui_scale
        _layer_session.ui_renderer.render_info_box((0, 0), (area_width, 20 * ui_scale))
        blf.position(0, 8 * ui_scale, 6 * ui_scale, 0)
        blf.size(0, 11 * ui_scale) if bpy.app.version >= (3, 6) else blf.size(0, 11 * ui_scale, 72)
        blf.color(0, 0.7, 0.7, 0.7, 1.0)
        blf.draw(0, info_text)

def layer_get_active_layer(context):
    """Get the currently active layer."""
    img = context.area.spaces.active.image
    if not img:
        return None
    img_props = img.imageeditorplus_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return None
    return layers[selected_layer_index]

def layer_get_target_image(context):
    """Get the target image (layer image or base image)."""
    layer = layer_get_active_layer(context)
    if layer:
        return bpy.data.images.get(layer.name, None)
    else:
        return context.area.spaces.active.image

def layer_enter_edit_mode(context):
    """Enter layer edit mode - swap to layer image for painting."""
    img = context.area.spaces.active.image
    if not img:
        return False
    
    img_props = img.imageeditorplus_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return False
    
    layer = layers[selected_layer_index]
    layer_img = bpy.data.images.get(layer.name, None)
    if not layer_img:
        return False
    
    # Store base image info on the layer image so we can get back
    layer_img_props = layer_img.imageeditorplus_properties
    layer_img_props.base_image_name = img.name
    layer_img_props.editing_layer = True
    
    # Swap to layer image
    context.area.spaces.active.image = layer_img
    return True

def layer_exit_edit_mode(context):
    """Exit layer edit mode - swap back to base image."""
    img = context.area.spaces.active.image
    if not img:
        return False
    
    img_props = img.imageeditorplus_properties
    
    if not img_props.editing_layer:
        return False
    
    base_img_name = img_props.base_image_name
    base_img = bpy.data.images.get(base_img_name, None)
    if not base_img:
        return False
    
    # Clear editing state
    img_props.editing_layer = False
    img_props.base_image_name = ''
    
    # Update the layer image
    img.update()
    if img.preview:
        img.preview.reload()
    
    # Swap back to base image
    context.area.spaces.active.image = base_img
    
    # Trigger layer refresh
    layer_rebuild_image_layers_nodes(base_img)
    
    return True

def layer_is_editing(context):
    """Check if currently in layer edit mode."""
    img = context.area.spaces.active.image
    if not img:
        return False
    return img.imageeditorplus_properties.editing_layer

def layer_convert_selection(context, mode='SET'):
    """Convert screen-space selection to image-space.
    
    Args:
        context: Blender context
        mode: 'SET' (replace), 'ADD' (extend), or 'SUBTRACT'
    """
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    selection_region = area_session.selection_region
    if not selection_region:
        return
    
    # Convert region coordinates to image coordinates
    x1, y1 = context.region.view2d.region_to_view(*selection_region[0])
    x2, y2 = context.region.view2d.region_to_view(*selection_region[1])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = round(x1 * width)
    y1 = round(y1 * height)
    x2 = round(x2 * width)
    y2 = round(y2 * height)
    
    # Don't clamp here - allow out-of-bounds selections
    # Clamping happens in paint mask and other operations that need it
    
    # Ensure minimum size
    if x2 - x1 <= 0:
        x2 = x1 + 1
    if y2 - y1 <= 0:
        y2 = y1 + 1
    
    new_rect = [[x1, y1], [x2, y2]]
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        # Replace all selections with new one (clear both boxes, ellipses, lassos, and negations)
        area_session._selections = [new_rect]
        area_session._ellipses = []
        area_session._lassos = []
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        # Add to existing selections
        area_session.add_selection(new_rect)
    elif mode == 'SUBTRACT':
        # Add to negation list (subtracts from ALL selection types)
        area_session._neg_rects.append(new_rect)
        area_session.selection_mask = None
    
    area_session.selection_mode = mode

def layer_convert_ellipse_selection(context, mode='SET'):
    """Convert screen-space ellipse selection to image-space."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    ellipse_region = area_session.ellipse_region
    if not ellipse_region:
        return
    
    # Convert region coordinates to image coordinates
    x1, y1 = context.region.view2d.region_to_view(*ellipse_region[0])
    x2, y2 = context.region.view2d.region_to_view(*ellipse_region[1])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = round(x1 * width)
    y1 = round(y1 * height)
    x2 = round(x2 * width)
    y2 = round(y2 * height)
    
    # Ensure minimum size
    if x2 - x1 <= 0:
        x2 = x1 + 1
    if y2 - y1 <= 0:
        y2 = y1 + 1
    
    new_bbox = [[x1, y1], [x2, y2]]
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        area_session._selections = []
        area_session._ellipses = [new_bbox]
        area_session._lassos = []
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        area_session.add_ellipse(new_bbox)
    elif mode == 'SUBTRACT':
        # Add to negation list for subtract
        area_session._neg_ellipses.append(new_bbox)
        area_session.selection_mask = None

    area_session.selection_mode = mode

def layer_convert_lasso_selection(context, mode='SET'):
    """Convert screen-space lasso selection to image-space."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    lasso_points = area_session.lasso_points
    if not lasso_points or len(lasso_points) < 3:
        return
    
    # Convert region coordinates to image coordinates
    image_points = []
    for point in lasso_points:
        vx, vy = context.region.view2d.region_to_view(point[0], point[1])
        ix = round(vx * width)
        iy = round(vy * height)
        image_points.append([ix, iy])
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        area_session._selections = []
        area_session._ellipses = []
        area_session._lassos = [image_points]
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        area_session.add_lasso(image_points)
    elif mode == 'SUBTRACT':
        # Add to negation list for subtract
        area_session._neg_lassos.append(image_points)
        area_session.selection_mask = None

    area_session.selection_mode = mode

def layer_crop_selection(context):
    """Clamp all selections to image bounds."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    if not area_session._selections:
        return
    
    # Clamp all selection rectangles
    cropped = []
    for sel in area_session._selections:
        [x1, y1], [x2, y2] = sel
        x1 = max(min(x1, width), 0)
        y1 = max(min(y1, height), 0)
        x2 = max(min(x2, width), 0)
        y2 = max(min(y2, height), 0)
        if x2 - x1 <= 0:
            if x2 < width:
                x2 = x2 + 1
            else:
                x1 = x1 - 1
        if y2 - y1 <= 0:
            if y2 < height:
                y2 = y2 + 1
            else:
                y1 = y1 - 1
        cropped.append([[x1, y1], [x2, y2]])
    area_session._selections = cropped

def layer_cancel_selection(context):
    """Cancel the current selection."""
    area_session = layer_get_area_session(context)
    # Push to undo history so clear can be undone
    area_session.push_undo()
    area_session.clear_selections()
    area_session.selection_region = None
    area_session.ellipse_region = None
    area_session.lasso_points = None
    area_session.selecting = False
    layer_clear_paint_mask(context)

# Paint mask data storage
_layer_paint_mask_data = {
    'enabled': False,
    'image_name': None,
    'selections': [],  # List of valid selection rectangles
    'full_cached': None,  # Cached original image
    'img_size': None,
    'timer': None
}

def layer_build_selection_mask(width, height, selections, subtract_rect=None):
    """Build numpy boolean mask from selection rectangles.
    
    Args:
        width, height: Image dimensions
        selections: List of [[x1, y1], [x2, y2]] rectangles
        subtract_rect: Optional rectangle to subtract from mask
    
    Returns:
        Numpy boolean array (height, width) - True = inside selection
    """
    # Start with all False (nothing selected)
    mask = np.zeros((height, width), dtype=bool)
    
    # Add all selection rectangles
    for sel in selections:
        [[x1, y1], [x2, y2]] = sel
        # Clamp to image bounds
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    
    # Subtract rectangle if provided
    if subtract_rect:
        [[x1, y1], [x2, y2]] = subtract_rect
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = False
    
    return mask

def _compute_gap_rectangles(width, height, selections):
    """Precompute rectangles that need to be restored (gaps between selections).
    
    Returns list of (y1, y2, x1, x2) tuples for efficient slicing.
    """
    if not selections:
        return [(0, height, 0, width)]  # Entire image
    
    # Clamp selections to image bounds
    clamped = []
    for sel in selections:
        x1 = max(0, min(sel[0][0], width))
        y1 = max(0, min(sel[0][1], height))
        x2 = max(0, min(sel[1][0], width))
        y2 = max(0, min(sel[1][1], height))
        if x2 > x1 and y2 > y1:
            clamped.append([[x1, y1], [x2, y2]])
    
    if not clamped:
        return [(0, height, 0, width)]
    
    selections = clamped
    
    # Compute bounding box
    min_x = min(s[0][0] for s in selections)
    min_y = min(s[0][1] for s in selections)
    max_x = max(s[1][0] for s in selections)
    max_y = max(s[1][1] for s in selections)
    
    gap_rects = []
    
    # Add outer edges (outside bounding box)
    if min_y > 0:
        gap_rects.append((0, min_y, 0, width))  # Top
    if max_y < height:
        gap_rects.append((max_y, height, 0, width))  # Bottom
    if min_x > 0:
        gap_rects.append((min_y, max_y, 0, min_x))  # Left
    if max_x < width:
        gap_rects.append((min_y, max_y, max_x, width))  # Right
    
    # Find ALL interior gaps between selection pairs (works for any count)
    for i, sel1 in enumerate(selections):
        for sel2 in selections[i+1:]:
            # Horizontal gap: selections side by side
            y_overlap_start = max(sel1[0][1], sel2[0][1])
            y_overlap_end = min(sel1[1][1], sel2[1][1])
            if y_overlap_start < y_overlap_end:
                if sel1[1][0] < sel2[0][0]:
                    gap_rects.append((y_overlap_start, y_overlap_end, sel1[1][0], sel2[0][0]))
                elif sel2[1][0] < sel1[0][0]:
                    gap_rects.append((y_overlap_start, y_overlap_end, sel2[1][0], sel1[0][0]))
            
            # Vertical gap: selections stacked
            x_overlap_start = max(sel1[0][0], sel2[0][0])
            x_overlap_end = min(sel1[1][0], sel2[1][0])
            if x_overlap_start < x_overlap_end:
                if sel1[1][1] < sel2[0][1]:
                    gap_rects.append((sel1[1][1], sel2[0][1], x_overlap_start, x_overlap_end))
                elif sel2[1][1] < sel1[0][1]:
                    gap_rects.append((sel2[1][1], sel1[0][1], x_overlap_start, x_overlap_end))
    
    return gap_rects

def layer_apply_selection_as_paint_mask(context):
    """Create paint mask from multi-selection regions."""
    global _layer_paint_mask_data
    
    area_session = layer_get_area_session(context)
    selections = area_session.selections
    ellipses = area_session.ellipses
    lassos = area_session.lassos
    neg_rects = area_session._neg_rects
    neg_ellipses = area_session._neg_ellipses
    neg_lassos = area_session._neg_lassos
    
    if not selections and not ellipses and not lassos:
        return
    
    img = context.area.spaces.active.image
    if not img:
        return
    
    width, height = img.size
    
    # Cache the full image (this is the guaranteed correct approach)
    pixels = layer_read_pixels_from_image(img)
    
    # Clamp and validate selections
    valid_selections = []
    if selections:
        for sel in selections:
            x1 = max(0, min(sel[0][0], width))
            y1 = max(0, min(sel[0][1], height))
            x2 = max(0, min(sel[1][0], width))
            y2 = max(0, min(sel[1][1], height))
            if x2 > x1 and y2 > y1:
                valid_selections.append([[x1, y1], [x2, y2]])
    
    valid_ellipses = []
    if ellipses:
        for sel in ellipses:
            # Store original bbox for ellipse equation
            orig_x1, orig_y1 = sel[0][0], sel[0][1]
            orig_x2, orig_y2 = sel[1][0], sel[1][1]
            
            # Clamp only the pixel access region
            x1 = max(0, min(orig_x1, width))
            y1 = max(0, min(orig_y1, height))
            x2 = max(0, min(orig_x2, width))
            y2 = max(0, min(orig_y2, height))
            
            if x2 > x1 and y2 > y1:
                # Store: clamped region for pixel access, original bbox for ellipse math
                valid_ellipses.append({
                    'region': [[x1, y1], [x2, y2]],
                    'original': [[orig_x1, orig_y1], [orig_x2, orig_y2]]
                })
    
    # Validate lassos (just store points, clamping happens during paste)
    valid_lassos = []
    if lassos:
        for lasso in lassos:
            if len(lasso) >= 3:
                valid_lassos.append(lasso)
                
    if not valid_selections and not valid_ellipses and not valid_lassos:
        return
    
    # Validate negation rectangles
    valid_neg_rects = []
    if neg_rects:
        for sel in neg_rects:
            x1 = max(0, min(sel[0][0], width))
            y1 = max(0, min(sel[0][1], height))
            x2 = max(0, min(sel[1][0], width))
            y2 = max(0, min(sel[1][1], height))
            if x2 > x1 and y2 > y1:
                valid_neg_rects.append([[x1, y1], [x2, y2]])
    
    # Validate negation ellipses
    valid_neg_ellipses = []
    if neg_ellipses:
        for sel in neg_ellipses:
            orig_x1, orig_y1 = sel[0][0], sel[0][1]
            orig_x2, orig_y2 = sel[1][0], sel[1][1]
            x1 = max(0, min(orig_x1, width))
            y1 = max(0, min(orig_y1, height))
            x2 = max(0, min(orig_x2, width))
            y2 = max(0, min(orig_y2, height))
            if x2 > x1 and y2 > y1:
                valid_neg_ellipses.append({
                    'region': [[x1, y1], [x2, y2]],
                    'original': [[orig_x1, orig_y1], [orig_x2, orig_y2]]
                })
    
    # Validate negation lassos
    valid_neg_lassos = []
    if neg_lassos:
        for lasso in neg_lassos:
            if len(lasso) >= 3:
                valid_neg_lassos.append(lasso)
    
    # Get invert mask setting
    wm = context.window_manager
    invert_mask = False
    if hasattr(wm, 'imageeditorplus_properties'):
        invert_mask = wm.imageeditorplus_properties.invert_mask
    
    _layer_paint_mask_data['enabled'] = True
    _layer_paint_mask_data['image_name'] = img.name
    _layer_paint_mask_data['selections'] = valid_selections
    _layer_paint_mask_data['ellipses'] = valid_ellipses
    _layer_paint_mask_data['lassos'] = valid_lassos
    _layer_paint_mask_data['neg_rects'] = valid_neg_rects
    _layer_paint_mask_data['neg_ellipses'] = valid_neg_ellipses
    _layer_paint_mask_data['neg_lassos'] = valid_neg_lassos
    _layer_paint_mask_data['invert_mask'] = invert_mask
    _layer_paint_mask_data['full_cached'] = pixels.copy()
    _layer_paint_mask_data['img_size'] = (width, height)
    
    # Register timer
    if _layer_paint_mask_data['timer'] is None:
        _layer_paint_mask_data['timer'] = bpy.app.timers.register(
            _layer_paint_mask_timer, 
            first_interval=0.8,
            persistent=True
        )

def _layer_paint_mask_timer():
    """Paint mask timer - guaranteed correct algorithm.
    
    Starts with cached image, pastes back only selection regions.
    This ensures paint NEVER escapes outside selections.
    """
    global _layer_paint_mask_data
    
    if not _layer_paint_mask_data['enabled']:
        _layer_paint_mask_data['timer'] = None
        return None
    
    img_name = _layer_paint_mask_data['image_name']
    if not img_name:
        return 0.8
    
    img = bpy.data.images.get(img_name)
    if not img:
        _layer_paint_mask_data['enabled'] = False
        _layer_paint_mask_data['timer'] = None
        return None
    
    # Skip if image hasn't changed
    if not img.is_dirty:
        return 0.8
    
    cached = _layer_paint_mask_data.get('full_cached')
    selections = _layer_paint_mask_data.get('selections', [])
    ellipses = _layer_paint_mask_data.get('ellipses', [])
    lassos = _layer_paint_mask_data.get('lassos', [])
    neg_rects = _layer_paint_mask_data.get('neg_rects', [])
    neg_ellipses = _layer_paint_mask_data.get('neg_ellipses', [])
    neg_lassos = _layer_paint_mask_data.get('neg_lassos', [])
    invert_mask = _layer_paint_mask_data.get('invert_mask', False)
    
    if cached is None or (not selections and not ellipses and not lassos):
        return 0.8
    
    try:
        # Read current painted image
        if img.size[0] != cached.shape[1] or img.size[1] != cached.shape[0]:
             # Image resized, abort mask
             _layer_paint_mask_data['enabled'] = False
             return None

        current = layer_read_pixels_from_image(img)
        
        if not np.array_equal(current.shape, cached.shape):
             _layer_paint_mask_data['enabled'] = False
             return None

        width, height = _layer_paint_mask_data['img_size']
        
        if invert_mask:
            # INVERTED: Start with current (keeps all paint outside selections)
            result = current.copy()
            
            # Restore cached INSIDE selection regions (removes paint inside)
            for sel in selections:
                x1, y1 = sel[0]
                x2, y2 = sel[1]
                result[y1:y2, x1:x2] = cached[y1:y2, x1:x2]
        else:
            # NORMAL: Start with cached (removes ALL paint)
            result = cached.copy()
            
            # Paste back current ONLY inside selection regions (keeps paint inside)
            for sel in selections:
                x1, y1 = sel[0]
                x2, y2 = sel[1]
                result[y1:y2, x1:x2] = current[y1:y2, x1:x2]
        
        # Paste back ellipses
        for ell in ellipses:
            # Get clamped region for pixel access
            region = ell['region']
            x1, y1 = region[0]
            x2, y2 = region[1]
            
            # Get original bbox for ellipse math
            orig = ell['original']
            orig_x1, orig_y1 = orig[0]
            orig_x2, orig_y2 = orig[1]
            
            # Original ellipse dimensions
            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            
            if orig_w <= 0 or orig_h <= 0:
                continue
            
            # Ellipse center and radii from original bbox
            center_x = orig_x1 + orig_w / 2
            center_y = orig_y1 + orig_h / 2
            radius_x = orig_w / 2
            radius_y = orig_h / 2
            
            # Region dimensions (what we're accessing)
            region_h = y2 - y1
            region_w = x2 - x1
            
            if region_w <= 0 or region_h <= 0:
                continue
            
            # Create mask for the clamped region using ORIGINAL ellipse equation
            cy, cx = np.ogrid[0:region_h, 0:region_w]
            px = x1 + cx
            py = y1 + cy
            
            mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
            
            patch_result = result[y1:y2, x1:x2]
            if invert_mask:
                # INVERTED: Restore cached inside ellipse (prevents paint inside)
                patch_cached = cached[y1:y2, x1:x2]
                patch_result[mask] = patch_cached[mask]
            else:
                # NORMAL: Paste current inside ellipse (keeps paint inside)
                patch_current = current[y1:y2, x1:x2]
                patch_result[mask] = patch_current[mask]
            result[y1:y2, x1:x2] = patch_result
        
        # Paste back lassos using polygon scanline fill
        for lasso in lassos:
            if len(lasso) < 3:
                continue
            
            # Compute bounding box of lasso (clamped to image)
            lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
            ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
            lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
            ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
            
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            
            # Create polygon mask using scanline fill
            poly_h = ly2 - ly1
            poly_w = lx2 - lx1
            poly_mask = np.zeros((poly_h, poly_w), dtype=bool)
            
            for y in range(poly_h):
                abs_y = ly1 + y
                intersections = []
                for i in range(len(lasso)):
                    p1 = lasso[i]
                    p2 = lasso[(i + 1) % len(lasso)]
                    if (p1[1] <= abs_y < p2[1]) or (p2[1] <= abs_y < p1[1]):
                        if p2[1] != p1[1]:
                            xi = p1[0] + (abs_y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                            intersections.append(int(xi))
                
                intersections.sort()
                for i in range(0, len(intersections) - 1, 2):
                    x1_fill = max(0, min(intersections[i] - lx1, poly_w))
                    x2_fill = max(0, min(intersections[i + 1] - lx1, poly_w))
                    if x2_fill > x1_fill:
                        poly_mask[y, x1_fill:x2_fill] = True
            
            # Apply polygon mask
            patch_result = result[ly1:ly2, lx1:lx2]
            if invert_mask:
                # INVERTED: Restore cached inside lasso (prevents paint inside)
                patch_cached = cached[ly1:ly2, lx1:lx2]
                patch_result[poly_mask] = patch_cached[poly_mask]
            else:
                # NORMAL: Paste current inside lasso (keeps paint inside)
                patch_current = current[ly1:ly2, lx1:lx2]
                patch_result[poly_mask] = patch_current[poly_mask]
            result[ly1:ly2, lx1:lx2] = patch_result
        
        # Apply negation rectangles - RESTORE cached pixels (prevent paint)
        for sel in neg_rects:
            x1, y1 = sel[0]
            x2, y2 = sel[1]
            result[y1:y2, x1:x2] = cached[y1:y2, x1:x2]
        
        # Apply negation ellipses - RESTORE cached pixels (prevent paint)
        for ell in neg_ellipses:
            region = ell['region']
            x1, y1 = region[0]
            x2, y2 = region[1]
            
            orig = ell['original']
            orig_x1, orig_y1 = orig[0]
            orig_x2, orig_y2 = orig[1]
            
            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            
            if orig_w <= 0 or orig_h <= 0:
                continue
            
            center_x = orig_x1 + orig_w / 2
            center_y = orig_y1 + orig_h / 2
            radius_x = orig_w / 2
            radius_y = orig_h / 2
            
            region_h = y2 - y1
            region_w = x2 - x1
            
            if region_w <= 0 or region_h <= 0:
                continue
            
            cy, cx = np.ogrid[0:region_h, 0:region_w]
            px = x1 + cx
            py = y1 + cy
            
            mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
            
            # Restore cached (subtract - no paint allowed here)
            patch_cached = cached[y1:y2, x1:x2]
            patch_result = result[y1:y2, x1:x2]
            patch_result[mask] = patch_cached[mask]
            result[y1:y2, x1:x2] = patch_result
        
        # Apply negation lassos - RESTORE cached pixels (prevent paint)
        for lasso in neg_lassos:
            if len(lasso) < 3:
                continue
            
            lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
            ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
            lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
            ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
            
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            
            poly_h = ly2 - ly1
            poly_w = lx2 - lx1
            poly_mask = np.zeros((poly_h, poly_w), dtype=bool)
            
            for y in range(poly_h):
                abs_y = ly1 + y
                intersections = []
                for i in range(len(lasso)):
                    p1 = lasso[i]
                    p2 = lasso[(i + 1) % len(lasso)]
                    if (p1[1] <= abs_y < p2[1]) or (p2[1] <= abs_y < p1[1]):
                        if p2[1] != p1[1]:
                            xi = p1[0] + (abs_y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                            intersections.append(int(xi))
                
                intersections.sort()
                for i in range(0, len(intersections) - 1, 2):
                    x1_fill = max(0, min(intersections[i] - lx1, poly_w))
                    x2_fill = max(0, min(intersections[i + 1] - lx1, poly_w))
                    if x2_fill > x1_fill:
                        poly_mask[y, x1_fill:x2_fill] = True
            
            # Restore cached (subtract - no paint allowed here)
            patch_cached = cached[ly1:ly2, lx1:lx2]
            patch_result = result[ly1:ly2, lx1:lx2]
            patch_result[poly_mask] = patch_cached[poly_mask]
            result[ly1:ly2, lx1:lx2] = patch_result

        # Only write if actually different
        if not np.array_equal(current, result):
            layer_write_pixels_to_image(img, result)
    except Exception as e:
        print(f"Paint mask error: {e}")
        pass
    
    return 0.8

def layer_clear_paint_mask(context):
    """Disable paint mask and clean up."""
    global _layer_paint_mask_data
    
    _layer_paint_mask_data['enabled'] = False
    _layer_paint_mask_data['image_name'] = None
    _layer_paint_mask_data['selections'] = []
    _layer_paint_mask_data['ellipses'] = []
    _layer_paint_mask_data['lassos'] = []
    _layer_paint_mask_data['full_cached'] = None
    _layer_paint_mask_data['img_size'] = None

def layer_get_selection(context):
    """Get current selection."""
    area_session = layer_get_area_session(context)
    return area_session.selection

def layer_get_target_selection(context):
    """Get selection if no layer is selected."""
    area_session = layer_get_area_session(context)
    selection = area_session.selection
    if not selection:
        return None
    img = context.area.spaces.active.image
    if not img:
        return selection
    img_props = img.imageeditorplus_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return selection
    return None

def layer_refresh_image(context):
    """Refresh the image in the editor."""
    wm = context.window_manager
    img = context.area.spaces.active.image
    if not img:
        return
    context.area.spaces.active.image = img
    img.update()
    if not hasattr(wm, 'imagelayersnode_api') or wm.imagelayersnode_api.VERSION < (1, 1, 0):
        return
    wm.imagelayersnode_api.update_pasted_layer_nodes(img)

def layer_apply_layer_transform(img, rot, scale):
    """Apply rotation and scale to a layer image."""
    global _layer_session
    if not _layer_session.ui_renderer:
        _layer_session.ui_renderer = UIRenderer()
    buff, width, height = _layer_session.ui_renderer.render_image_offscreen(img, rot, scale)
    pixels = np.array([[pixel for pixel in row] for row in buff], np.float32) / 255.0
    layer_convert_colorspace(pixels, 'Linear', 'Linear' if img.is_float else img.colorspace_settings.name)
    return pixels, width, height

def layer_create_layer(base_img, pixels, img_settings, layer_settings, custom_label=None):
    """Create a new layer from pixels."""
    base_width, base_height = base_img.size
    target_width, target_height = pixels.shape[1], pixels.shape[0]
    layer_img_prefix = '#layer'
    layer_img_name = base_img.name + layer_img_prefix
    layer_img = bpy.data.images.new(layer_img_name, width=target_width, height=target_height, alpha=True, float_buffer=base_img.is_float)
    layer_img.colorspace_settings.name = base_img.colorspace_settings.name
    pixels = pixels.copy()
    layer_convert_colorspace(pixels, 'Linear' if img_settings['is_float'] else img_settings['colorspace_name'], 'Linear' if base_img.is_float else base_img.colorspace_settings.name)
    layer_write_pixels_to_image(layer_img, pixels)
    layer_img.use_fake_user = True
    layer_img.pack()
    img_props = base_img.imageeditorplus_properties
    layers = img_props.layers
    layer = layers.add()
    layer.name = layer_img.name
    layer.location = [int((base_width - target_width) / 2.0), int((base_height - target_height) / 2.0)]
    
    # Set layer label
    if custom_label:
        layer.label = custom_label
    else:
        layer_img_postfix = layer_img.name[layer_img.name.rfind(layer_img_prefix) + len(layer_img_prefix):]
        if layer_img_postfix:
            layer.label = 'Layer ' + layer_img_postfix
        else:
            layer.label = 'Layer'
    
    if layer_settings:
        layer.rotation = layer_settings['rotation']
        layer.scale = layer_settings['scale']
        layer.custom_data = layer_settings['custom_data']
    layers.move(len(layers) - 1, 0)
    img_props.selected_layer_index = 0
    layer_rebuild_image_layers_nodes(base_img)

def layer_rebuild_image_layers_nodes(img):
    """Rebuild layer nodes for the image."""
    wm = bpy.context.window_manager
    if not hasattr(wm, 'imagelayersnode_api') or wm.imagelayersnode_api.VERSION < (1, 1, 0):
        return
    wm.imagelayersnode_api.rebuild_image_layers_nodes(img)

def layer_cleanup_scene():
    """Cleanup layer node groups."""
    node_group = bpy.data.node_groups.get('imageeditorplus')
    if node_group:
        bpy.data.node_groups.remove(node_group)

@bpy.app.handlers.persistent
def layer_save_pre_handler(args):
    """Handler called before saving to pack dirty images."""
    layer_cleanup_scene()
    for img in bpy.data.images:
        if img.source != 'VIEWER':
            if img.is_dirty:
                if img.packed_files or not img.filepath:
                    img.pack()
                else:
                    img.save()

# Layer-specific pixel read/write functions
def layer_read_pixels_from_image(img):
    """Read pixels from image as numpy array."""
    width, height = img.size[0], img.size[1]
    pixels = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(pixels)
    return np.reshape(pixels, (height, width, 4))

def layer_write_pixels_to_image(img, pixels):
    """Write pixels to image from numpy array."""
    img.pixels.foreach_set(np.reshape(pixels, -1))
    if img.preview:
        img.preview.reload()

def layer_convert_colorspace(pixels, src_colorspace, dest_colorspace):
    """Convert pixels between color spaces."""
    if src_colorspace == dest_colorspace:
        return
    if src_colorspace == 'Linear' and dest_colorspace == 'sRGB':
        pixels[:, :, 0:3] = pixels[:, :, :3] ** (1.0 / 2.2)
    elif src_colorspace == 'sRGB' and dest_colorspace == 'Linear':
        pixels[:, :, 0:3] = pixels[:, :, :3] ** 2.2

