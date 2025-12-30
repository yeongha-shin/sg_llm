import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# =====================================================
# 0. 설정
# =====================================================
shp_path = "./data/N3L_F0010000_INC/N3L_F0010000_28.shp"
ELEV_COL = "CONT"          # 등고선 고도 컬럼
resolution = 100.0           # meter / pixel
output_png = "gradient_heightmap.png"

# =====================================================
# 1. SHP 로드
# =====================================================
gdf = gpd.read_file(shp_path)

print("컬럼 목록:", gdf.columns)

if ELEV_COL not in gdf.columns:
    raise ValueError(f"{ELEV_COL} 컬럼이 없습니다.")

# 고도 타입 보정 (문자열 → float)
gdf[ELEV_COL] = gdf[ELEV_COL].astype(float)

# 좌표계 보정 (lat/lon이면 미터 좌표계로 변환)
if gdf.crs is None:
    raise ValueError("CRS가 정의되지 않은 SHP입니다.")

if gdf.crs.is_geographic:
    print("지리 좌표계 감지 → EPSG:5179로 변환")
    gdf = gdf.to_crs(epsg=5179)

print("사용 CRS:", gdf.crs)

# =====================================================
# 2. 등고선 → 포인트 샘플링
# =====================================================
points = []
values = []

for geom, elev in zip(gdf.geometry, gdf[ELEV_COL]):

    if geom.geom_type == "LineString":
        coords = geom.coords

    elif geom.geom_type == "MultiLineString":
        coords = []
        for line in geom.geoms:
            coords.extend(line.coords)

    else:
        continue

    for x, y in coords:
        points.append((x, y))
        values.append(elev)

points = np.array(points)
values = np.array(values)

print(f"샘플 포인트 수: {len(points)}")

if len(points) == 0:
    raise RuntimeError("포인트 추출 실패")

# =====================================================
# 3. Raster Grid 생성
# =====================================================
xmin, ymin, xmax, ymax = gdf.total_bounds

xi, yi = np.meshgrid(
    np.arange(xmin, xmax, resolution),
    np.arange(ymin, ymax, resolution)
)

print("Raster 크기:", xi.shape)

# =====================================================
# 4. 보간 (안정적인 2단계 방식)
# =====================================================
# 1차: linear
zi = griddata(points, values, (xi, yi), method="linear")

# 2차: nearest (외곽 NaN 보정)
zi_nearest = griddata(points, values, (xi, yi), method="nearest")
zi = np.where(np.isnan(zi), zi_nearest, zi)

# =====================================================
# 5. 0~1 정규화
# =====================================================
z_min = zi.min()
z_max = zi.max()

if z_max - z_min == 0:
    raise RuntimeError("고도 변화가 없습니다.")

zi_norm = (zi - z_min) / (z_max - z_min)

# =====================================================
# 6. PNG 저장 (grayscale heightmap)
# =====================================================
png = (zi_norm * 255).astype(np.uint8)

plt.imsave(
    output_png,
    png,
    cmap="gray",
    origin="lower"
)

print(f"✅ {output_png} 생성 완료")
print(f"고도 범위: {z_min:.2f} m ~ {z_max:.2f} m")
