# nyx/core/spatial/map_visualization.py

import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from nyx.core.spatial.spatial_mapper import CognitiveMap, SpatialObject, SpatialRegion, SpatialRoute

logger = logging.getLogger(__name__)

class MapVisualization:
    """
    Provides visualization capabilities for cognitive maps.
    Can generate SVG, ASCII art, or data for interactive visualizations.
    """
    
    @staticmethod
    def generate_svg(cognitive_map: CognitiveMap, width: int = 800, height: int = 600, 
                   highlight_landmark_ids: List[str] = None,
                   highlight_route_id: Optional[str] = None,
                   highlight_object_ids: List[str] = None) -> str:
        """
        Generate an SVG visualization of a cognitive map
        
        Args:
            cognitive_map: The cognitive map to visualize
            width: Width of the SVG
            height: Height of the SVG
            highlight_landmark_ids: Optional list of landmark IDs to highlight
            highlight_route_id: Optional route ID to highlight
            highlight_object_ids: Optional list of object IDs to highlight
            
        Returns:
            SVG string representation of the map
        """
        # Default empty lists
        highlight_landmark_ids = highlight_landmark_ids or []
        highlight_object_ids = highlight_object_ids or []
        
        # Find bounds of the map
        min_x, max_x, min_y, max_y = MapVisualization._find_map_bounds(cognitive_map)
        
        # Add padding
        padding = 0.1  # 10% padding
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        min_x -= x_range * padding
        max_x += x_range * padding
        min_y -= y_range * padding
        max_y += y_range * padding
        
        # Scaling factors
        scale_x = width / (max_x - min_x) if max_x > min_x else 1
        scale_y = height / (max_y - min_y) if max_y > min_y else 1
        
        # Start SVG
        svg = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add definitions for markers and patterns
        svg += MapVisualization._generate_svg_defs()
        
        # Add title
        svg += f'  <title>{cognitive_map.name}</title>\n'
        
        # Add background
        svg += f'  <rect width="{width}" height="{height}" fill="#f0f0f0" />\n'
        
        # Draw regions
        for region_id, region in cognitive_map.regions.items():
            svg += MapVisualization._generate_svg_region(region, min_x, min_y, scale_x, scale_y)
        
        # Draw routes
        for route_id, route in cognitive_map.routes.items():
            is_highlighted = route_id == highlight_route_id
            svg += MapVisualization._generate_svg_route(
                route, cognitive_map, min_x, min_y, scale_x, scale_y, is_highlighted
            )
        
        # Draw objects
        for obj_id, obj in cognitive_map.spatial_objects.items():
            is_landmark = obj_id in cognitive_map.landmarks
            is_highlighted_landmark = obj_id in highlight_landmark_ids
            is_highlighted_object = obj_id in highlight_object_ids
            
            svg += MapVisualization._generate_svg_object(
                obj, min_x, min_y, scale_x, scale_y, 
                is_landmark, is_highlighted_landmark, is_highlighted_object
            )
        
        # Add legend
        svg += MapVisualization._generate_svg_legend(width, height)
        
        # Close SVG
        svg += '</svg>'
        
        return svg
    
    @staticmethod
    def generate_ascii_map(cognitive_map: CognitiveMap, width: int = 80, height: int = 40) -> str:
        """
        Generate an ASCII art visualization of a cognitive map
        
        Args:
            cognitive_map: The cognitive map to visualize
            width: Width of the ASCII art in characters
            height: Height of the ASCII art in characters
            
        Returns:
            ASCII string representation of the map
        """
        # Find bounds of the map
        min_x, max_x, min_y, max_y = MapVisualization._find_map_bounds(cognitive_map)
        
        # Add padding
        padding = 0.1  # 10% padding
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        min_x -= x_range * padding
        max_x += x_range * padding
        min_y -= y_range * padding
        max_y += y_range * padding
        
        # Scaling factors
        scale_x = width / (max_x - min_x) if max_x > min_x else 1
        scale_y = height / (max_y - min_y) if max_y > min_y else 1
        
        # Create empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw regions (as boundaries)
        for region_id, region in cognitive_map.regions.items():
            if len(region.boundary_points) < 3:
                continue
                
            # Draw region outline
            for i in range(len(region.boundary_points)):
                p1 = region.boundary_points[i]
                p2 = region.boundary_points[(i + 1) % len(region.boundary_points)]
                
                # Convert to grid coordinates
                x1 = int((p1.x - min_x) * scale_x)
                y1 = int((p1.y - min_y) * scale_y)
                x2 = int((p2.x - min_x) * scale_x)
                y2 = int((p2.y - min_y) * scale_y)
                
                # Clamp to grid bounds
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                
                # Draw line
                MapVisualization._draw_ascii_line(grid, x1, y1, x2, y2, '.')
        
        # Draw routes
        for route_id, route in cognitive_map.routes.items():
            waypoints = route.waypoints
            if len(waypoints) < 2:
                continue
                
            # Draw route path
            for i in range(len(waypoints) - 1):
                p1 = waypoints[i]
                p2 = waypoints[i + 1]
                
                # Convert to grid coordinates
                x1 = int((p1.x - min_x) * scale_x)
                y1 = int((p1.y - min_y) * scale_y)
                x2 = int((p2.x - min_x) * scale_x)
                y2 = int((p2.y - min_y) * scale_y)
                
                # Clamp to grid bounds
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                
                # Draw line
                MapVisualization._draw_ascii_line(grid, x1, y1, x2, y2, '-')
        
        # Draw objects
        for obj_id, obj in cognitive_map.spatial_objects.items():
            # Convert to grid coordinates
            x = int((obj.coordinates.x - min_x) * scale_x)
            y = int((obj.coordinates.y - min_y) * scale_y)
            
            # Clamp to grid bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Choose symbol based on object type
            symbol = 'o'  # Default
            
            if obj_id in cognitive_map.landmarks:
                symbol = '*'  # Landmark
            elif obj.object_type == "door":
                symbol = 'D'
            elif obj.object_type == "chair":
                symbol = 'c'
            elif obj.object_type == "table":
                symbol = 'T'
            elif obj.object_type == "wall":
                symbol = '#'
            elif obj.object_type == "window":
                symbol = 'W'
            
            # Place on grid
            grid[y][x] = symbol
            
            # Add first letter of name if space available
            if x + 1 < width and obj.name:
                grid[y][x + 1] = obj.name[0].lower()
        
        # Convert grid to string
        ascii_map = '\n'.join([''.join(row) for row in grid])
        
        # Add title and legend
        title = f" {cognitive_map.name} ".center(width, '-')
        legend = (
            f"Legend: * = Landmark, o = Object, - = Route, . = Region boundary\n"
            f"Map completeness: {cognitive_map.completeness:.2f}, Objects: {len(cognitive_map.spatial_objects)}, "
            f"Regions: {len(cognitive_map.regions)}"
        )
        
        return f"{title}\n{ascii_map}\n{'-' * width}\n{legend}"
    
    @staticmethod
    def generate_map_data(cognitive_map: CognitiveMap) -> Dict[str, Any]:
        """
        Generate structured data representation of a cognitive map
        for use in interactive visualizations
        
        Args:
            cognitive_map: The cognitive map to generate data for
            
        Returns:
            Dictionary with structured map data
        """
        # Find bounds of the map
        min_x, max_x, min_y, max_y = MapVisualization._find_map_bounds(cognitive_map)
        
        # Prepare data structure
        map_data = {
            "id": cognitive_map.id,
            "name": cognitive_map.name,
            "description": cognitive_map.description,
            "type": cognitive_map.map_type,
            "reference_frame": cognitive_map.reference_frame,
            "bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y
            },
            "metadata": {
                "creation_date": cognitive_map.creation_date,
                "last_updated": cognitive_map.last_updated,
                "accuracy": cognitive_map.accuracy,
                "completeness": cognitive_map.completeness
            },
            "objects": [],
            "landmarks": [],
            "regions": [],
            "routes": []
        }
        
        # Add objects
        for obj_id, obj in cognitive_map.spatial_objects.items():
            obj_data = {
                "id": obj_id,
                "name": obj.name,
                "type": obj.object_type,
                "position": {
                    "x": obj.coordinates.x,
                    "y": obj.coordinates.y
                },
                "is_landmark": obj_id in cognitive_map.landmarks,
                "observation_count": obj.observation_count,
                "connections": obj.connections
            }
            
            # Add Z coordinate if available
            if obj.coordinates.z is not None:
                obj_data["position"]["z"] = obj.coordinates.z
            
            # Add size if available
            if obj.size:
                obj_data["size"] = obj.size
            
            # Add to appropriate list
            if obj_id in cognitive_map.landmarks:
                map_data["landmarks"].append(obj_data)
            
            map_data["objects"].append(obj_data)
        
        # Add regions
        for region_id, region in cognitive_map.regions.items():
            region_data = {
                "id": region_id,
                "name": region.name,
                "type": region.region_type,
                "is_navigable": region.is_navigable,
                "confidence": region.confidence,
                "boundary": [],
                "contained_objects": region.contained_objects,
                "adjacent_regions": region.adjacent_regions
            }
            
            # Add boundary points
            for point in region.boundary_points:
                point_data = {"x": point.x, "y": point.y}
                if point.z is not None:
                    point_data["z"] = point.z
                region_data["boundary"].append(point_data)
            
            map_data["regions"].append(region_data)
        
        # Add routes
        for route_id, route in cognitive_map.routes.items():
            route_data = {
                "id": route_id,
                "name": route.name or f"Route from {route.start_id} to {route.end_id}",
                "start_id": route.start_id,
                "end_id": route.end_id,
                "distance": route.distance,
                "estimated_time": route.estimated_time,
                "usage_count": route.usage_count,
                "waypoints": []
            }
            
            # Add waypoints
            for point in route.waypoints:
                point_data = {"x": point.x, "y": point.y}
                if point.z is not None:
                    point_data["z"] = point.z
                route_data["waypoints"].append(point_data)
            
            map_data["routes"].append(route_data)
        
        return map_data
    
    # --- Private helper methods ---
    
    @staticmethod
    def _find_map_bounds(cognitive_map: CognitiveMap) -> Tuple[float, float, float, float]:
        """Find the min/max coordinates of the map"""
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        # Check objects
        for obj_id, obj in cognitive_map.spatial_objects.items():
            min_x = min(min_x, obj.coordinates.x)
            max_x = max(max_x, obj.coordinates.x)
            min_y = min(min_y, obj.coordinates.y)
            max_y = max(max_y, obj.coordinates.y)
        
        # Check region boundaries
        for region_id, region in cognitive_map.regions.items():
            for point in region.boundary_points:
                min_x = min(min_x, point.x)
                max_x = max(max_x, point.x)
                min_y = min(min_y, point.y)
                max_y = max(max_y, point.y)
        
        # Check route waypoints
        for route_id, route in cognitive_map.routes.items():
            for point in route.waypoints:
                min_x = min(min_x, point.x)
                max_x = max(max_x, point.x)
                min_y = min(min_y, point.y)
                max_y = max(max_y, point.y)
        
        # Handle empty map
        if min_x == float('inf'):
            min_x, max_x = 0, 100
            min_y, max_y = 0, 100
        
        return min_x, max_x, min_y, max_y
    
    @staticmethod
    def _generate_svg_defs() -> str:
        """Generate SVG definitions for markers and patterns"""
        defs = '  <defs>\n'
        
        # Arrowhead marker for routes
        defs += '''    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#3366FF" />
    </marker>\n'''
        
        # Pattern for regions
        defs += '''    <pattern id="region-pattern" width="10" height="10" patternUnits="userSpaceOnUse">
      <rect width="10" height="10" fill="#E6E6E6" />
      <circle cx="5" cy="5" r="1" fill="#CCCCCC" />
    </pattern>\n'''
        
        # Close defs
        defs += '  </defs>\n'
        
        return defs
    
    @staticmethod
    def _generate_svg_region(region: SpatialRegion, min_x: float, min_y: float, 
                          scale_x: float, scale_y: float) -> str:
        """Generate SVG for a region"""
        if len(region.boundary_points) < 3:
            return ''
        
        # Create polygon points
        points = []
        for point in region.boundary_points:
            x = (point.x - min_x) * scale_x
            y = (point.y - min_y) * scale_y
            points.append(f"{x},{y}")
        
        # Choose color based on region type
        fill = "url(#region-pattern)"
        stroke = "#666666"
        stroke_width = 1
        opacity = 0.7
        
        if region.region_type == "room":
            fill = "#E6F2FF"
        elif region.region_type == "corridor":
            fill = "#F2F2F2"
        elif region.region_type == "outside":
            fill = "#E6FFE6"
        
        if not region.is_navigable:
            fill = "#FFE6E6"
            opacity = 0.8
        
        # Create polygon element
        svg = f'  <polygon points="{" ".join(points)}" fill="{fill}" stroke="{stroke}" '
        svg += f'stroke-width="{stroke_width}" opacity="{opacity}">\n'
        
        # Add title
        svg += f'    <title>{region.name} ({region.region_type})</title>\n'
        
        # Close polygon
        svg += '  </polygon>\n'
        
        # Add text label
        # Calculate centroid for label
        avg_x = sum(point.x for point in region.boundary_points) / len(region.boundary_points)
        avg_y = sum(point.y for point in region.boundary_points) / len(region.boundary_points)
        
        label_x = (avg_x - min_x) * scale_x
        label_y = (avg_y - min_y) * scale_y
        
        svg += f'  <text x="{label_x}" y="{label_y}" text-anchor="middle" '
        svg += f'font-family="Arial" font-size="12" fill="#333333">{region.name}</text>\n'
        
        return svg
    
    @staticmethod
    def _generate_svg_route(route: SpatialRoute, cognitive_map: CognitiveMap, 
                         min_x: float, min_y: float, scale_x: float, scale_y: float,
                         is_highlighted: bool = False) -> str:
        """Generate SVG for a route"""
        waypoints = route.waypoints
        if len(waypoints) < 2:
            return ''
        
        # Create path data
        path_data = f'M {(waypoints[0].x - min_x) * scale_x} {(waypoints[0].y - min_y) * scale_y}'
        
        for i in range(1, len(waypoints)):
            path_data += f' L {(waypoints[i].x - min_x) * scale_x} {(waypoints[i].y - min_y) * scale_y}'
        
        # Choose style
        stroke = "#3366FF" if not is_highlighted else "#FF3366"
        stroke_width = 2 if not is_highlighted else 3
        stroke_dasharray = "none" if not is_highlighted else "none"
        opacity = 0.8 if not is_highlighted else 1.0
        
        # Create path element
        svg = f'  <path d="{path_data}" fill="none" stroke="{stroke}" '
        svg += f'stroke-width="{stroke_width}" stroke-dasharray="{stroke_dasharray}" '
        svg += f'opacity="{opacity}" marker-end="url(#arrowhead)">\n'
        
        # Add title
        start_name = "Unknown"
        end_name = "Unknown"
        
        if route.start_id in cognitive_map.spatial_objects:
            start_name = cognitive_map.spatial_objects[route.start_id].name
        elif route.start_id in cognitive_map.regions:
            start_name = cognitive_map.regions[route.start_id].name
            
        if route.end_id in cognitive_map.spatial_objects:
            end_name = cognitive_map.spatial_objects[route.end_id].name
        elif route.end_id in cognitive_map.regions:
            end_name = cognitive_map.regions[route.end_id].name
            
        route_name = route.name or f"Route from {start_name} to {end_name}"
        distance_info = f", Distance: {route.distance:.1f}" if route.distance > 0 else ""
        time_info = f", Time: {route.estimated_time:.1f}s" if route.estimated_time else ""
        
        svg += f'    <title>{route_name}{distance_info}{time_info}</title>\n'
        
        # Close path
        svg += '  </path>\n'
        
        return svg
    
    @staticmethod
    def _generate_svg_object(obj: SpatialObject, min_x: float, min_y: float, 
                          scale_x: float, scale_y: float, is_landmark: bool = False,
                          is_highlighted_landmark: bool = False,
                          is_highlighted_object: bool = False) -> str:
        """Generate SVG for a spatial object"""
        # Calculate position
        x = (obj.coordinates.x - min_x) * scale_x
        y = (obj.coordinates.y - min_y) * scale_y
        
        # Choose symbol based on object type
        symbol_type = "circle"  # Default
        
        if obj.object_type == "door":
            symbol_type = "rect"
        elif obj.object_type == "chair":
            symbol_type = "rect"
        elif obj.object_type == "table":
            symbol_type = "rect"
        elif obj.object_type == "window":
            symbol_type = "rect"
        
        # Choose size based on object type and landmark status
        size = 5
        if is_landmark:
            size = 8
        
        # Choose colors
        fill = "#3366FF"
        stroke = "#000000"
        
        if is_landmark:
            fill = "#FFD700"  # Gold for landmarks
            
        if is_highlighted_landmark:
            fill = "#FF9900"  # Orange for highlighted landmarks
            size += 2
            
        if is_highlighted_object:
            fill = "#FF3366"  # Pink for highlighted objects
            size += 2
        
        # Create element
        svg = ""
        if symbol_type == "circle":
            svg = f'  <circle cx="{x}" cy="{y}" r="{size}" fill="{fill}" '
            svg += f'stroke="{stroke}" stroke-width="1">\n'
        elif symbol_type == "rect":
            svg = f'  <rect x="{x - size/2}" y="{y - size/2}" width="{size}" height="{size}" '
            svg += f'fill="{fill}" stroke="{stroke}" stroke-width="1">\n'
        
        # Add title
        landmark_info = " (Landmark)" if is_landmark else ""
        observation_info = f", Observations: {obj.observation_count}" if obj.observation_count > 1 else ""
        
        svg += f'    <title>{obj.name} ({obj.object_type}){landmark_info}{observation_info}</title>\n'
        
        # Close element
        if symbol_type == "circle":
            svg += '  </circle>\n'
        elif symbol_type == "rect":
            svg += '  </rect>\n'
        
        # Add text label
        label_y = y + size + 12
        
        svg += f'  <text x="{x}" y="{label_y}" text-anchor="middle" '
        svg += f'font-family="Arial" font-size="10" fill="#333333">{obj.name}</text>\n'
        
        return svg
    
    @staticmethod
    def _generate_svg_legend(width: int, height: int) -> str:
        """Generate SVG legend"""
        legend_x = 10
        legend_y = height - 50
        legend_width = 300
        legend_height = 40
        
        # Create legend box
        svg = f'  <rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" '
        svg += f'fill="white" stroke="#CCCCCC" stroke-width="1" rx="5" ry="5" opacity="0.8" />\n'
        
        # Add legend items
        item_x = legend_x + 10
        item_y = legend_y + 15
        
        # Landmark
        svg += f'  <circle cx="{item_x}" cy="{item_y}" r="5" fill="#FFD700" stroke="#000000" stroke-width="1" />\n'
        svg += f'  <text x="{item_x + 10}" y="{item_y + 5}" font-family="Arial" font-size="10">Landmark</text>\n'
        
        # Object
        item_x += 80
        svg += f'  <circle cx="{item_x}" cy="{item_y}" r="5" fill="#3366FF" stroke="#000000" stroke-width="1" />\n'
        svg += f'  <text x="{item_x + 10}" y="{item_y + 5}" font-family="Arial" font-size="10">Object</text>\n'
        
        # Route
        item_x += 70
        svg += f'  <line x1="{item_x}" y1="{item_y}" x2="{item_x + 20}" y2="{item_y}" '
        svg += f'stroke="#3366FF" stroke-width="2" />\n'
        svg += f'  <text x="{item_x + 30}" y="{item_y + 5}" font-family="Arial" font-size="10">Route</text>\n'
        
        # Region
        item_x += 70
        svg += f'  <rect x="{item_x}" y="{item_y - 5}" width="10" height="10" '
        svg += f'fill="#E6F2FF" stroke="#666666" stroke-width="1" opacity="0.7" />\n'
        svg += f'  <text x="{item_x + 15}" y="{item_y + 5}" font-family="Arial" font-size="10">Region</text>\n'
        
        return svg
    
    @staticmethod
    def _draw_ascii_line(grid: List[List[str]], x1: int, y1: int, x2: int, y2: int, char: str) -> None:
        """Draw a line on ASCII grid using Bresenham's line algorithm"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        
        while True:
            # Check bounds
            if 0 <= y1 < height and 0 <= x1 < width:
                grid[y1][x1] = char
                
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
