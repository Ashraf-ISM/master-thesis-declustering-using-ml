#!/bin/bash
echo "=== New Zealand Seismicity Map Generator ==="
echo "Processing earthquake catalog data..."

# Check if data file exists
if [ ! -f "nz.txt" ]; then
    echo "ERROR: nz.txt file not found!"
    echo "Please ensure your earthquake catalog is named 'nz.txt' and in the current directory."
    exit 1
fi

# Create output directories
mkdir -p figures
mkdir -p data_processed

# Count events and get basic statistics
TOTAL_EVENTS=$(tail -n +2 nz.txt | wc -l)
echo "Total events found: $TOTAL_EVENTS"

# Get data bounds
echo "Calculating data bounds..."
awk -F',' 'NR>1 {print $4, $3}' nz.txt | gmt info -I0.1 > bounds.txt
REGION=$(cat bounds.txt | awk '{print $1}')
echo "Region: $REGION"

# Process earthquake data for plotting
echo "Processing earthquake data for GMT..."
awk -F',' '
BEGIN {
    print "# Longitude Latitude Depth Magnitude Year"
}
NR>1 {
    split($1, date, "-")
    year = date[1]
    lon = $4
    lat = $3
    depth = $5
    mag = $6
    
    # Skip invalid data
    if (lon != "" && lat != "" && depth != "" && mag != "" && 
        lon >= 160 && lon <= 180 && lat >= -50 && lat <= -30 && 
        depth >= 0 && depth <= 700 && mag >= 0 && mag <= 10) {
        printf "%.4f %.4f %.1f %.2f %d\n", lon, lat, depth, mag, year
    }
}' nz.txt > data_processed/earthquakes.txt

# Count processed events
PROCESSED_EVENTS=$(tail -n +2 data_processed/earthquakes.txt | wc -l)
echo "Events after processing: $PROCESSED_EVENTS"

# Create depth categories file
awk 'NR>1 {
    if ($3 <= 70) print $1, $2, $4, "shallow"
    else if ($3 <= 300) print $1, $2, $4, "intermediate" 
    else print $1, $2, $4, "deep"
}' data_processed/earthquakes.txt > data_processed/depth_categories.txt

# Start GMT modern session
gmt begin nz_seismicity_map pdf,png

    # Set GMT defaults for publication quality
    # Set GMT defaults for publication quality
    gmt set MAP_FRAME_TYPE plain
    gmt set MAP_FRAME_WIDTH 2p
    gmt set MAP_FRAME_PEN 1p,black
    gmt set MAP_TICK_LENGTH_PRIMARY 4p
    gmt set MAP_TICK_LENGTH_SECONDARY 2p
    # Remove old GMT 5 keywords
    # gmt set MAP_ANNOT_FONT_PRIMARY 10p
    # gmt set MAP_LABEL_FONT 12p
    # gmt set MAP_TITLE_FONT 16p
    # Modern GMT 6 font settings
    gmt set FONT_ANNOT_PRIMARY 10p,Helvetica,black
    gmt set FONT_LABEL 12p,Helvetica,black
    gmt set FONT_TITLE 16p,Helvetica-Bold,black
    gmt set MAP_GRID_PEN_PRIMARY 0.5p,gray70
    gmt set MAP_GRID_PEN_SECONDARY 0.25p,gray80
    gmt set PS_MEDIA A4
    gmt set PS_PAGE_ORIENTATION portrait
    

    # Define projection and region
    # Using Mercator projection suitable for New Zealand
    PROJ="M15c"
    REG="160/180/-50/-30"
    
    echo "Creating base map with topography and coastlines..."
    
    # Download and plot topography (SRTM data)
    # Add base map with both X and Y axes labels
    gmt coast -R$REG -J$PROJ -Df -W0.5p,black -N1/1p,red -N2/0.5p,gray50 \
              -Bpxa5f1+l"Longitude (°E)" -Bpya2f1+l"Latitude (°S)" \
              -B+t"New Zealand Seismicity Map (1980-2024)" \
              -Lg170/-48+c-40+w200k+u+f
    
    
    # Add topographic relief
    gmt grdimage @earth_relief_01m -R$REG -J$PROJ -I+d -Cgeo -t50
    
    # Add coastlines again on top
    gmt coast -R$REG -J$PROJ -Df -W1p,black -N1/1p,red
    
    echo "Adding earthquake data by depth categories..."
    
    # Plot earthquakes by depth with different colors and sizes based on magnitude
    # Deep earthquakes (>300 km) - Purple circles
    awk '$4=="deep"' data_processed/depth_categories.txt | \
    awk '{size = 0.05 + ($3-2)*0.08; if(size<0.05) size=0.05; if(size>0.4) size=0.4; print $1, $2, size}' | \
    gmt plot -R$REG -J$PROJ -Sc -Gblue4 -W0.2p,white

    awk '$4>7' data_processed/depth_categories.txt | \
    awk '{
        size = 0.05 + ($4-7)*0.08; 
        if(size<0.05) size=0.05; 
        if(size>0.5) size=0.5; 
        print $1, $2, size
    }' > great_eqs.txt

    # Plot as stars
    gmt plot great_eqs.txt -R$REG -J$PROJ -Sa -Ggold -W0.3p,black

    
    # Intermediate earthquakes (70-300 km) - Orange circles  
    awk '$4=="intermediate"' data_processed/depth_categories.txt | \
    awk '{size = 0.05 + ($3-2)*0.08; if(size<0.05) size=0.05; if(size>0.4) size=0.4; print $1, $2, size}' | \
    gmt plot -R$REG -J$PROJ -Sc -Gorange2 -W0.2p,white
    
    # Shallow earthquakes (<=70 km) - Red circles
    awk '$4=="shallow"' data_processed/depth_categories.txt | \
    awk '{size = 0.05 + ($3-2)*0.08; if(size<0.05) size=0.05; if(size>0.4) size=0.4; print $1, $2, size}' | \
    gmt plot -R$REG -J$PROJ -Sc -Gred -W0.2p,white
    
    echo "Adding major cities and tectonic features..."
    
    # Add major cities
    cat << EOF > cities.txt
174.7762 -41.2865 Wellington
175.2793 -37.7870 Hamilton  
174.7633 -36.8485 Auckland
172.6362 -43.5321 Christchurch
170.5037 -45.8788 Dunedin
173.2597 -41.2706 Nelson
EOF
    
    gmt plot cities.txt -R$REG -J$PROJ -Sc0.15c -Gwhite -W1p,black
    gmt text cities.txt -R$REG -J$PROJ -F+f9p,Helvetica-Bold,black+jML -D0.2c/0c
    
    # Add plate boundary (simplified)
    cat << EOF > plate_boundary.txt
# Pacific-Australian plate boundary (simplified)
166.0 -46.0
167.5 -45.0
169.0 -44.0
170.5 -43.0
172.0 -42.0
173.5 -41.0
175.0 -40.0
176.5 -39.0
178.0 -38.0
179.0 -37.0
EOF
    
    gmt plot plate_boundary.txt -R$REG -J$PROJ -W2p,purple,- -L
    
    echo "Creating legends and annotations..."
    
    # Create magnitude legend
    gmt legend -R$REG -J$PROJ -DjTR+w4c+o0.5c -F+p1p+gwhite+s << EOF
H 11p,Helvetica-Bold Magnitude
S 0.3c c 0.08c red 0.2p,white 0.5c M 2.0
S 0.3c c 0.15c red 0.2p,white 0.5c M 4.0  
S 0.3c c 0.25c red 0.2p,white 0.5c M 6.0
S 0.3c c 0.35c red 0.2p,white 0.5c M 8.0
G 0.1c
H 11p,Helvetica-Bold Depth (km)
S 0.3c c 0.15c red 0.2p,white 0.5c 0-70 (Shallow)
S 0.3c c 0.15c orange2 0.2p,white 0.5c 70-300 (Intermediate)
S 0.3c c 0.15c blue4 0.2p,white 0.5c >300 (Deep)
G 0.1c
S 0.3c - 0.3c purple 2p 0.5c Plate Boundary
EOF

    # Add scale bar
    gmt basemap -R$REG -J$PROJ -Lx0.5c/-1.5c+w100k+u+f
    
    # Add north arrow
    gmt basemap -R$REG -J$PROJ -Tdg178/-32+w1c+f1
    
    # Add data source and statistics
    gmt text -R$REG -J$PROJ -F+f8p,Helvetica+jBL -Gwhite -W0.5p << EOF
161 -49.5 Events: $PROCESSED_EVENTS | Period: 1980-Present | Source: GeoNet
161 -49.8 Coordinate System: WGS84 | Projection: Mercator
EOF

gmt end show

echo ""
echo "=== Map Generation Complete ==="
echo "Output files created:"
echo "  - nz_seismicity_map.pdf (Vector format)"
echo "  - nz_seismicity_map.png (Raster format, 300 DPI)"
echo ""
echo "=== Creating Supplementary Analysis Plots ==="

# Create magnitude-frequency distribution
echo "Generating magnitude-frequency plot..."
gmt begin magnitude_frequency png

    # Extract magnitudes and create histogram data
    awk 'NR>1 {print $4}' data_processed/earthquakes.txt | \
    gmt histogram -T0.5/9/0.2 -Z1 -Gblue > mag_hist.txt
    
    gmt plot mag_hist.txt -R0.5/8/1/10000 -JX12c/8c -Sb0.15c -Gblue \
            -Bxa1f0.5+l"Magnitude" -Bya1pf3+l"Number of Events" \
            -BWsne+t"Magnitude-Frequency Distribution"
    
    # Add Gutenberg-Richter fit line (theoretical)
    gmt math -T0.5/8/0.1 T 4.5 SUB -1 MUL EXP10 30000 MUL = gr_line.txt
    gmt plot gr_line.txt -W2p,red,-

gmt end

# Create depth distribution
echo "Generating depth distribution plot..."
gmt begin depth_distribution png

    awk 'NR>1 {print $3}' data_processed/earthquakes.txt | \
    gmt histogram -T0/600/20 -Z1 -Ggreen > depth_hist.txt
    
    gmt plot depth_hist.txt -R0/600/1/1000 -JX12c/8c -Sb15 -Ggreen \
            -Bxa100f50+l"Depth (km)" -Bya1pf3+l"Number of Events" \
            -BWsne+t"Earthquake Depth Distribution"

gmt end

# Create temporal analysis
echo "Generating temporal analysis plot..."
gmt begin temporal_analysis png

    # Create yearly counts
    awk 'NR>1 {print $5}' data_processed/earthquakes.txt | \
    sort | uniq -c | awk '{print $2, $1}' > yearly_counts.txt
    
    gmt plot yearly_counts.txt -R1980/2025/0/15000 -JX15c/8c -W2p,blue \
            -Bxa5f1+l"Year" -Bya2000f1000+l"Number of Events" \
            -BWsne+t"Temporal Distribution of Seismic Activity"
            
    # Add points
    gmt plot yearly_counts.txt -Sc0.1c -Gblue

gmt end

echo ""
echo "=== Analysis Complete ==="
echo "Additional plots created:"
echo "  - magnitude_frequency.png"
echo "  - depth_distribution.png"  
echo "  - temporal_analysis.png"
echo ""
echo "=== Clean up temporary files ==="
rm -f bounds.txt cities.txt plate_boundary.txt 
rm -f mag_hist.txt depth_hist.txt yearly_counts.txt gr_line.txt

echo "=== Publication Notes ==="
echo "1. Maps generated at 300 DPI suitable for publication"
echo "2. PDF version available for vector graphics editing"
echo "3. Coordinate system: WGS84"
echo "4. Projection: Mercator"
echo "5. Data processing includes magnitude scaling and depth categorization"
echo "6. Color scheme: Red (shallow), Orange (intermediate), Blue (deep)"
echo "7. Symbol sizes scaled by earthquake magnitude"
echo ""
echo "=== Usage Instructions ==="
echo "To run this script:"
echo "1. Ensure GMT 6.3.0+ is installed"
echo "2. Place earthquake data as 'nz.txt' in current directory"
echo "3. Run: bash $(basename "$0")"
echo "4. Check 'figures/' directory for output files"

echo ""
echo "Script execution completed successfully!"