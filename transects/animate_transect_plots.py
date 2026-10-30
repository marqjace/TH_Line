import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np

def main():
    # List of image file paths in order
    image_paths = [
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect1_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect2_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect3_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect4_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect5_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect6_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect7_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect8_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect9_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\caloos\TH_line_sg266_transect10_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg686_sep_2026_anomaly_plots\caloos\TH_line_sg686_transect2_T_anom_caloos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg686_sep_2026_anomaly_plots\caloos\TH_line_sg686_transect3_T_anom_caloos.png',
    ]

    # Load images
    images = [np.array(Image.open(p)) for p in image_paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    im = ax.imshow(images[0])

    def update(frame):
        im.set_data(images[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images),
                                interval=2000,    # ms between frames
                                repeat=True)

    plt.tight_layout()
    ani.save(r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\TH_line_T_anom_caloos.gif', writer='pillow', fps=1)

    # List of image file paths in order
    image_paths = [
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect1_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect2_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect3_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect4_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect5_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect6_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect7_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect8_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect9_T_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect10_T_anom_nanoos.png',
    ]

    # Load images
    images = [np.array(Image.open(p)) for p in image_paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    im = ax.imshow(images[0])

    def update(frame):
        im.set_data(images[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images),
                                interval=2000,    # ms between frames
                                repeat=True)

    plt.tight_layout()
    ani.save(r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_T_anom_nanoos.gif', writer='pillow', fps=1)

    # List of image file paths in order
    image_paths = [
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect1_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect2_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect3_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect4_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect5_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect6_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect7_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect8_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect9_S_anom_nanoos.png',
        r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_transect10_S_anom_nanoos.png',
    ]

    # Load images
    images = [np.array(Image.open(p)) for p in image_paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    im = ax.imshow(images[0])

    def update(frame):
        im.set_data(images[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images),
                                interval=2000,    # ms between frames
                                repeat=True)

    plt.tight_layout()
    ani.save(r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures\sg266_mar_2026_anomaly_plots\nanoos\TH_line_sg266_S_anom_nanoos.gif', writer='pillow', fps=1)

if __name__ == "__main__":
    main()