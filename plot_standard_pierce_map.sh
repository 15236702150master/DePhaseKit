#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "Usage: INPUT_FILE=/path/to/pierce_points_xy.txt OUTPUT_PREFIX=/path/to/output_prefix $0" >&2
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found: $INPUT_FILE" >&2
    exit 1
fi

REGION="${REGION:--32/-23/-61/-55}"
PROJ="${PROJ:-M15c}"
# Default to empty here; resolved to a local grid after GMT_HOME is set below.
# The "@earth_relief_01m" remote form makes GMT fetch the grid over the network,
# which stalls (SSL handshake failures / hangs) when the data server is
# unreachable, freezing the export.
RELIEF_GRID="${RELIEF_GRID:-}"
CUSTOM_CPT="${CUSTOM_CPT:-$SCRIPT_DIR/cpt/south_sandwich_reference.cpt}"
EVENT_LON="${EVENT_LON:-}"
EVENT_LAT="${EVENT_LAT:-}"

export GMT_HOME="${GMT_HOME:-$PROJECT_ROOT/opt/gmt}"
export PATH="$GMT_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GMT_HOME/lib:${LD_LIBRARY_PATH:-}"
export GMT_SHAREDIR="${GMT_HOME}/share"

# Resolve the relief grid: honor an explicit RELIEF_GRID, else use the bundled
# local file under share/custom/. Only fall back to the remote "@earth_relief"
# form if no local grid is present.
LOCAL_RELIEF_DEFAULT="$GMT_HOME/share/custom/earth_relief_01m.grd"
if [ -z "$RELIEF_GRID" ]; then
    if [ -f "$LOCAL_RELIEF_DEFAULT" ]; then
        RELIEF_GRID="$LOCAL_RELIEF_DEFAULT"
    else
        RELIEF_GRID="@earth_relief_01m"
    fi
fi

if [ ! -x "$GMT_HOME/bin/gmt" ] && ! command -v gmt >/dev/null 2>&1; then
    echo "gmt command not found. Check GMT_HOME: $GMT_HOME" >&2
    exit 1
fi

if [ ! -f "$CUSTOM_CPT" ]; then
    echo "Custom CPT not found: $CUSTOM_CPT" >&2
    exit 1
fi

OUTPUT_DIR=$(dirname -- "$OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

# If the relief grid is the bundled global grid, crop it to the plot region once
# and cache the small regional subset. Re-reading the 225 MB global grid on every
# render is slow; the cached crop is a few hundred KB and renders in seconds.
RELIEF_TO_USE="$RELIEF_GRID"
if [ -f "$RELIEF_GRID" ] && [ "$(stat -c%s "$RELIEF_GRID" 2>/dev/null || echo 0)" -gt 100000000 ]; then
    region_tag=$(echo "$REGION" | tr '/' '_')
    grid_hash=$(stat -c '%Y_%s' "$RELIEF_GRID" 2>/dev/null | tr '\n' '_')
    CACHED_CROP="${TMPDIR:-/tmp}/dpk_relief_${grid_hash}_${region_tag}.grd"
    if [ ! -f "$CACHED_CROP" ]; then
        gmt grdcut "$RELIEF_GRID" -R"$REGION" -G"$CACHED_CROP" 2>/dev/null || true
    fi
    if [ -f "$CACHED_CROP" ]; then
        RELIEF_TO_USE="$CACHED_CROP"
    fi
fi

gmt begin "$OUTPUT_PREFIX" png
    gmt set FORMAT_GEO_MAP ddd.xx
    gmt set MAP_FRAME_TYPE plain
    gmt set FONT_ANNOT_PRIMARY 10p,Helvetica
    gmt set FONT_LABEL 12p,Helvetica-Bold
    gmt set FONT_TITLE 15p,Helvetica-Bold

    if gmt grdimage "$RELIEF_TO_USE" -R"$REGION" -J"$PROJ" -C"$CUSTOM_CPT" -I+d+a45+nt0.2 2>/dev/null; then
        gmt coast -R"$REGION" -J"$PROJ" -Df -W0.6p,black -A1000 -N1/0.4p,gray30 2>/dev/null || true
    else
        gmt coast -R"$REGION" -J"$PROJ" -Df -Glightgray -Swhite -W0.8p,black -A1000 -N1/0.5p,gray
    fi

    gmt basemap -R"$REGION" -J"$PROJ" -Bxa1f0.5 -Bya1f0.5 -BWeSn

    # Batch-draw pierce points. Calling gmt plot once per point (200+ module
    # invocations for 100 traces) makes the export crawl because each call
    # round-trips through the PostScript pipe. Group points by their status color
    # and draw each color in a single gmt plot call instead.
    TMP_BASE="$OUTPUT_DIR/.dpk_pierce_base.tmp"
    : > "$TMP_BASE"
    declare -A CORE_BY_COLOR
    FLIP_COORDS="$OUTPUT_DIR/.dpk_pierce_flip.tmp"
    : > "$FLIP_COORDS"
    while read -r lon lat color is_flip; do
        [ -n "${lon:-}" ] || continue
        [ -n "${lat:-}" ] || continue
        point_color="${color:-#111111}"
        printf '%s %s\n' "$lon" "$lat" >> "$TMP_BASE"
        CORE_BY_COLOR["$point_color"]+="${lon} ${lat}"$'\n'
        [ "${is_flip:-0}" = "1" ] && printf '%s %s\n' "$lon" "$lat" >> "$FLIP_COORDS"
    done < "$INPUT_FILE"

    # Dark base anchors the true location (one call for all points).
    if [ -s "$TMP_BASE" ]; then
        gmt plot "$TMP_BASE" -R"$REGION" -J"$PROJ" -Sc0.26c -Gblack -W0.20p,white@35
    fi
    # Status-colored core, one call per distinct color.
    for point_color in "${!CORE_BY_COLOR[@]}"; do
        printf '%s' "${CORE_BY_COLOR[$point_color]}" | gmt plot -R"$REGION" -J"$PROJ" -Sc0.16c -G"$point_color" -W0.18p,white@25
    done
    if [ -s "$FLIP_COORDS" ]; then
        gmt plot "$FLIP_COORDS" -R"$REGION" -J"$PROJ" -Sc0.31c -W0.85p,#ff9f1c
    fi
    rm -f "$TMP_BASE" "$FLIP_COORDS"

    if [ -n "$EVENT_LON" ] && [ -n "$EVENT_LAT" ]; then
        printf '%s %s\n' "$EVENT_LON" "$EVENT_LAT" | gmt plot -R"$REGION" -J"$PROJ" -Sa0.50c -Gred -W0.8p,black
    fi

    gmt colorbar -C"$CUSTOM_CPT" -DJBC+w8.5c/0.45c+o0c/1.2c+h -Bxaf2000f1000+l"Elevation (m)"
gmt end

echo "${OUTPUT_PREFIX}.png"
