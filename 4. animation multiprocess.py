import os
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import concurrent.futures  # <<< 1. 引入并行处理库
from functools import partial # <<< 2. 引入 partial 功能，方便传递参数

def readtxt(filepath):
    text_here = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text_here.append(line.strip())
    return text_here
            
def process_text(filepath):
    text = readtxt(filepath)
    parsed_labels = []

    for line in text:
        parts = line.split('#')
        if len(parts) == 4:
            label_info = {
                "raw_text": parts[0],
                "value_1": float(parts[2]),
                "value_2": float(parts[3])
            }
            parsed_labels.append(label_info)
    return parsed_labels

def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
#   matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    global data
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
        ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    # 使用 ffmpeg 作为 writer，确保 profile 设置正确
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer='ffmpeg')
    plt.close(fig) # <<< 附带优化：明确关闭 figure 对象，避免内存泄漏

# <<< 3. 将原来 for 循环里的逻辑封装成一个函数
def process_single_file(npy_file, src_dir, tgt_ani_dir, txt_dir, kinematic_chain):
    """
    处理单个 npy 文件，加载数据、文本，并生成动画。
    这是原来 for 循环体内的全部逻辑。
    """
    try:
        data = np.load(pjoin(src_dir, npy_file))
        save_path = pjoin(tgt_ani_dir, npy_file[:-3] + 'mp4')
        text_path = pjoin(txt_dir, npy_file[:-3] + 'txt')
        text_dict = process_text(text_path)
        
        # 确保 text_dict 不为空
        if not text_dict:
            return f"Skipping {npy_file}: No text found."
            
        plot_3d_motion(save_path, kinematic_chain, data, title=text_dict[0]['raw_text'], fps=20, radius=4)
        return f"Successfully generated animation for {npy_file}"
    except Exception as e:
        # 捕获任何可能发生的错误，防止单个文件的失败导致整个程序中断
        return f"Failed to process {npy_file}: {e}"

if __name__ == "__main__":
    src_dir = './HumanML3D/new_joints/'
    tgt_ani_dir = "./HumanML3D/animations/"
    txt_dir = './HumanML3D/texts/'

    kinematic_chain = [
        [0, 2, 5, 8, 11], # 右腿 pelvis, right_hip, right_knee, right_ankle, right_foot
        [0, 1, 4, 7, 10], # 左腿 pelvis, left_hip, left_knee, left_ankle, left_foot
        [0, 3, 6, 9, 12, 15], # 躯干 pelvis, spine1, spine2, spine3, neck, head
        [9, 14, 17, 19, 21],  # 右臂 spine3, right_collar, right_shoulder, right_elbow, right_wrist
        [9, 13, 16, 18, 20]   # 左臂 spine3, left_collar, left_shoulder, left_elbow, left_wrist
    ]
    os.makedirs(tgt_ani_dir, exist_ok=True)

    npy_files = sorted(os.listdir(src_dir))
    
    # max_workers=None 会自动使用所有可用的 CPU 核心
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # 使用 partial 固定住那些在所有任务中都保持不变的参数
        task_func = partial(process_single_file, 
                              src_dir=src_dir, 
                              tgt_ani_dir=tgt_ani_dir, 
                              txt_dir=txt_dir, 
                              kinematic_chain=kinematic_chain)
        
        # 使用 executor.map 将文件列表分配给多个进程处理
        # tqdm 用于显示进度条
        results = list(tqdm(executor.map(task_func, npy_files), total=len(npy_files)))

    print("\n--- Processing Finished ---")
    # (可选) 打印所有处理结果，方便查看是否有失败的任务
    # for res in results:
    #     if "Failed" in res:
    #         print(res)
    print("All tasks have been processed.")