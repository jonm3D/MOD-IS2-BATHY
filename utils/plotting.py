import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing

def create_combined_plot(data, cmap, legend_labels):
    """Create a combined interactive plot with raw data in top plot, classifications in middle plot, and confidence in bottom plot"""
    colors_hex = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in cmap.colors]
    
    # Create subplots with clear titles - now with 3 rows
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=("Raw Data (Reference)", "Classification with Surface Model", "Confidence Values"),
        vertical_spacing=0.12,  # Reduced vertical spacing to accommodate third plot
        shared_xaxes=True,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
        horizontal_spacing=0.03
    )
    
    # Adjust subplot title position
    fig.update_annotations(yshift=10, font_size=16)
    
    # Compute data ranges for consistent zooming
    x_min, x_max = data.x_ph.min(), data.x_ph.max()
    y_min, y_max = data.z_ph.min(), data.z_ph.max()
    
    # Track scatter trace indices for slider updates
    scatter_trace_indices = []
    
    # Add raw data to top plot (all points, single color)
    fig.add_trace(
        go.Scattergl(
            x=data.x_ph,
            y=data.z_ph,
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(100, 100, 100, 0.5)',  # Gray color for all points
                opacity=0.5
            ),
            name='Raw Data',
            hovertemplate='x: %{x:.2f}<br>z: %{y:.2f}',
            showlegend=True
        ),
        row=1, col=1
    )
    scatter_trace_indices.append(0)  # Add index of scatter trace for slider control
    
    # Add classifications to middle plot
    unique_classes = sorted(data.classification.unique())
    for class_val in unique_classes:
        mask = data.classification == class_val
        fig.add_trace(
            go.Scattergl(
                x=data.loc[mask, 'x_ph'],
                y=data.loc[mask, 'z_ph'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors_hex[class_val],
                    opacity=0.5
                ),
                name=legend_labels[class_val],
                hovertemplate='x: %{x:.2f}<br>z: %{y:.2f}<br>class: ' + legend_labels[class_val],
                legendgroup=f'class_{class_val}',
                showlegend=True
            ),
            row=2, col=1
        )
        scatter_trace_indices.append(len(fig.data) - 1)
    
    # Add surface model line to middle plot only
    fig.add_trace(
        go.Scatter(
            x=data.x_ph,
            y=data.surf_mu,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Surface Model',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Add confidence values to bottom plot
    # Use a continuous color scale for confidence values
    fig.add_trace(
        go.Scattergl(
            x=data.x_ph,
            y=data.z_ph,
            mode='markers',
            marker=dict(
                size=5,
                color=data.confidence,  # Use confidence values for color
                colorscale='Viridis',   # Use Viridis colorscale
                colorbar=dict(
                    title="Confidence",
                    thickness=15,
                    len=0.3,
                    y=0.15,             # Position vertically centered relative to the 3rd plot
                    yanchor="middle",   # Anchor at middle for vertical centering
                    xpad=10,            # Add some padding to the right
                    x=1.02,             # Position slightly to the right of the plot
                    xanchor="left"      # Anchor at left side of colorbar
                ),
                opacity=0.5,
                cmin=0,  # Set color range for confidence values
                cmax=1,
                showscale=True
            ),
            name='Confidence',
            hovertemplate='x: %{x:.2f}<br>z: %{y:.2f}<br>confidence: %{marker.color:.3f}',
            showlegend=False
        ),
        row=3, col=1
    )
    scatter_trace_indices.append(len(fig.data) - 1)  # Add index of confidence scatter for slider control
    
    # Create slider steps for point size
    size_steps = []
    for point_size in range(1, 11):
        step = dict(
            method='update',
            args=[{'marker.size': [point_size if i in scatter_trace_indices else None for i in range(len(fig.data))]}],
            label=str(point_size)
        )
        size_steps.append(step)
    
    # Create slider steps for opacity
    opacity_steps = []
    for opacity_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        step = dict(
            method='update',
            args=[{'marker.opacity': [opacity_val if i in scatter_trace_indices else None for i in range(len(fig.data))]}],
            label=str(opacity_val)
        )
        opacity_steps.append(step)
    
    # Create sliders
    sliders = [
        dict(
            active=4,
            currentvalue={"prefix": "Point Size: "},
            pad={"t": 50, "b": 10},
            len=0.9,
            x=0.1,
            y=0.02,
            steps=size_steps
        ),
        dict(
            active=4,
            currentvalue={"prefix": "Opacity: "},
            pad={"t": 50, "b": 10},
            len=0.9,
            x=0.1,
            y=-0.1,
            steps=opacity_steps
        )
    ]
    
    # Update layout with sliders - increase height to accommodate three plots
    fig.update_layout(
        height=1200,  # Increased from 950 to accommodate third plot
        width=1200,
        title_text="ICESat-2 Bathymetry Classification",
        title=dict(
            y=0.98,
            xanchor='center',
            x=0.5,
            font=dict(size=20)
        ),
        template="plotly_white",
        hovermode="closest",
        uirevision="same",
        sliders=sliders,
        margin=dict(t=100, b=150, r=250),
        showlegend=True,
        legend=dict(
            title="<b>Classification</b>",
            yanchor="middle",
            y=0.5,
            xanchor="center",
            x=1.15,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            itemsizing="constant"
        )
    )
    
    # Update axes for all three plots
    # Top plot
    fig.update_xaxes(
        title_text="", # No x-title for top plot since axes are shared
        row=1, col=1,
        range=[x_min, x_max],
        matches='x2'
    )
    fig.update_yaxes(
        title_text="Z (elevation, m)", 
        row=1, col=1,
        range=[y_min, y_max]
    )
    
    # Middle plot
    fig.update_xaxes(
        title_text="", # No x-title for middle plot since axes are shared
        row=2, col=1,
        range=[x_min, x_max],
        matches='x3'
    )
    fig.update_yaxes(
        title_text="Z (elevation, m)", 
        row=2, col=1,
        range=[y_min, y_max],
        matches='y'
    )
    
    # Bottom plot
    fig.update_xaxes(
        title_text="X (along-track distance, m)", # Only bottom plot shows x-axis title
        row=3, col=1,
        range=[x_min, x_max]
    )
    fig.update_yaxes(
        title_text="Z (elevation, m)", 
        row=3, col=1,
        range=[y_min, y_max],
        matches='y'
    )
    
    # Add grid to all plots
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    return fig


def create_bathymetry_map(output_dir):
    """
    Generate a visualization of the bathymetry data with an ESRI World Imagery basemap.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing the processed bathymetry data
    save_format : str, optional
        Format to save the map ('html' for interactive, 'png' for static)
        
    Returns:
    --------
    None, saves the map to the output directory
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import box
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Generating bathymetry visualization map")
    
    # Load the combined bathymetry data
    gpkg_path = os.path.join(output_dir, "bathymetry.gpkg")
    if not os.path.exists(gpkg_path):
        logger.error(f"Bathymetry data not found at {gpkg_path}")
        return
    
    try:
        # Read the bathymetry data
        bathy_data = gpd.read_file(gpkg_path)
        
        if bathy_data.empty:
            logger.warning("Bathymetry dataset is empty")
            return
            
        # Ensure data is in Web Mercator for basemap compatibility
        if bathy_data.crs != "EPSG:3857":
            bathy_data = bathy_data.to_crs("EPSG:3857")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a colormap for depth values (z_ph_refr)
        if 'z_ph_refr' in bathy_data.columns:
            # Normalize values for colormap
            vmin = bathy_data['z_ph_refr'].min()
            # vmax = bathy_data['z_ph_refr'].max()
            vmax = 5
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            # cmap = cm.viridis_r
            # plasma
            cmap = cm.plasma_r
            
            # Plot the points
            scatter = bathy_data.plot(
                column='z_ph_refr',
                ax=ax,
                alpha=0.7,
                markersize=5,
                cmap=cmap,
                norm=norm,
                legend=True,
                legend_kwds={'label': 'Depth (m)'}
            )
        else:
            # Fallback if z_ph_refr not available
            bathy_data.plot(ax=ax, color='blue', alpha=0.7, markersize=3)
            logger.warning("z_ph_refr column not found, using default blue color")
            
        # Add basemap
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.Esri.WorldImagery,
                zoom='auto'
            )
        except Exception as e:
            logger.error(f"Failed to add basemap: {str(e)}")
            
        # Set title and layout
        plt.title('ICESat-2 Bathymetry Points')
        plt.tight_layout()
        
        # Save as static image
        png_path = os.path.join(output_dir, "bathymetry_map.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logger.info(f"Map image saved to {png_path}")
            
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating bathymetry map: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False