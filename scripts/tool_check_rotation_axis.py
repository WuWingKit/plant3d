"""检查旋转轴的位置"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:5]

print(f"=== 检查旋转轴的位置 ===")
print(f"分析前 5 帧")

# 计算每帧的质心
centroids = []
for i in range(min(5, len(pcd_files))):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    centroids.append(centroid)
    print(f"帧 {i} 质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

centroids = np.array(centroids)

# 计算质心的平均位置
avg_centroid = centroids.mean(axis=0)
print(f"\n质心平均位置: ({avg_centroid[0]:.1f}, {avg_centroid[1]:.1f}, {avg_centroid[2]:.1f})")

# 计算质心在 XZ 平面上的分布
print(f"\n质心在 XZ 平面上的分布:")
for i in range(len(centroids)):
    x, y, z = centroids[i]
    angle = np.degrees(np.arctan2(z, x))
    dist = np.sqrt(x**2 + z**2)
    print(f"  帧 {i}: X={x:.1f}, Z={z:.1f}, 角度={angle:.1f}°, 距离={dist:.1f}mm")

# 计算质心在 XZ 平面上的圆心
xz_centroids = centroids[:, [0, 2]]
# 使用最小二乘法拟合圆
def fit_circle(x, y):
    """拟合圆"""
    A = np.array([
        [2*x[0], 2*y[0], 1],
        [2*x[1], 2*y[1], 1],
        [2*x[2], 2*y[2], 1],
    ])
    b = np.array([x[0]**2 + y[0]**2, x[1]**2 + y[1]**2, x[2]**2 + y[2]**2])
    try:
        params = np.linalg.solve(A, b)
        cx, cy = params[0], params[1]
        r = np.sqrt(params[2] + cx**2 + cy**2)
        return cx, cy, r
    except:
        return None, None, None

if len(xz_centroids) >= 3:
    cx, cy, r = fit_circle(xz_centroids[:, 0], xz_centroids[:, 1])
    if cx is not None:
        print(f"\n拟合圆:")
        print(f"  圆心: ({cx:.1f}, {cy:.1f})")
        print(f"  半径: {r:.1f}mm")
