import os
import numpy as np
from skimage import morphology, measure
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plt_fig(
    test_img, scores, img_scores, gts, threshold, cls_threshold, save_dir, class_name
):
    num = len(scores)
    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()
        heat_map = scores[i] * 255

        # --- Thresholding & Morphology ---
        mask = scores[i].copy()
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        # --- Get Bounding Boxes from mask ---
        label_mask = measure.label(mask)
        regions = measure.regionprops(label_mask)
        bboxes = []
        for region in regions:
            if region.area > 10:  # filter out noise
                minr, minc, maxr, maxc = region.bbox
                bboxes.append((minc, minr, maxc - minc, maxr - minr))  # (x, y, w, h)

        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")

        fig_img, ax_img = plt.subplots(
            1, 4, figsize=(9, 3), gridspec_kw={"width_ratios": [4, 4, 4, 3]}
        )
        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        # --- Draw All Panels ---
        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Input image")

        ax_img[1].imshow(gt, cmap="gray")
        ax_img[1].title.set_text("GroundTruth")

        ax_img[2].imshow(heat_map, cmap="jet", norm=norm, interpolation="none")
        ax_img[2].imshow(img, cmap="gray", alpha=0.7, interpolation="none")
        ax_img[2].title.set_text("Segmentation")

        # --- Draw bounding boxes on segmentation image ---
        for bbox in bboxes:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            ax_img[2].add_patch(rect)

        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[3].imshow(black_mask, cmap="gray")
        ax = plt.gca()

        cls_result = "nok" if img_scores[i] > cls_threshold else "ok"

        # --- Classification Panel Text ---
        ax_img[3].title.set_text("Classification")
        ax.text(0.05, 0.89, "Detected anomalies", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.79, "------------------------", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.72, "Results", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.67, "------------------------", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.59, f"'{cls_result}'", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="r", family="sans-serif"))
        ax.text(0.05, 0.47, f"Anomaly scores: {img_scores[i]:.2f}", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.37, "------------------------", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.30, "Thresholds", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.25, "------------------------", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.17, f"Classification: {cls_threshold:.2f}", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))
        ax.text(0.05, 0.07, f"Segmentation: {threshold:.2f}", transform=ax.transAxes,
                fontdict=dict(fontsize=8, color="w", family="sans-serif"))

        # --- Save the figure ---
        fig_img.savefig(
            os.path.join(save_dir, f"{class_name}_{i}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
