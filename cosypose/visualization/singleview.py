import numpy as np

from cosypose.visualization.plotter import Plotter

import cv2
import os
from bokeh.io import export_png, show
import torch
from pathlib import Path
from bokeh.plotting import figure, output_file, save
# import chromedriver_binary
# from matplotlib.ppyplot import 
from matplotlib import pyplot as plt

from cosypose.config import LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer

from cosypose.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from cosypose.datasets.augmentations import CropResizeToAspectAugmentation


def filter_predictions(preds, scene_id, view_id=None, th=None):
    mask = preds.infos['scene_id'] == scene_id
    if view_id is not None:
        mask = np.logical_and(mask, preds.infos['view_id'] == view_id)
    if th is not None:
        mask = np.logical_and(mask, preds.infos['score'] >= th)
    keep_ids = np.where(mask)[0]
    preds = preds[keep_ids]
    return preds


def render_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
        )
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])[0]['rgb']
    return rgb_rendered

def make_singleview_prediction_matplotlib(scene_ds, renderer, predictions, detections=None, resolution=(640, 480)):
    scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=resolution)
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]

    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=state['camera'])
    return rgb_input, pred_rendered


def make_singleview_prediction_plots(scene_ds, renderer, predictions, detections=None, resolution=(640, 480)):
    plotter = Plotter()

    scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=resolution)
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]

    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    if detections is not None:
        fig_dets = plotter.plot_image(rgb_input)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections)
        figures['detections'] = fig_dets

    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=state['camera'])
    # print("pred_rendered", type(pred_rendered), pred_rendered.shape)
    cv2.imshow("test", pred_rendered)

    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)
    return figures

class VisualizeSingleView():
    def __init__(self, ds_name, result_id) -> None:
        self.pred_keys = ['pix2pose_detections/refiner/iteration=1','pix2pose_detections/refiner/iteration=2']
        self.result_id =  result_id
        self.ds_name = ds_name
        self.urdf_ds_name = 'bracket_assembly'
        results = LOCAL_DATA_DIR / 'results' / result_id / 'results.pth.tar'
        self.scene_ds = make_scene_dataset(ds_name)
        self.results = torch.load(results)['predictions']
        LOCAL_DATA = Path('/home/ubuntu/synthetic_pose_estimation/cosypose/local_data')
        self.VIZ_DATA_DIR = LOCAL_DATA / 'visualizations' / self.result_id
        self.renderer = BulletSceneRenderer(self.urdf_ds_name)
        if not os.path.exists(self.VIZ_DATA_DIR):
            os.mkdir(self.VIZ_DATA_DIR)

    def visualize_single(self, scene_id, view_id):
        plt.figure()
        f, ax = plt.subplots(2,2) 
        for pred_id, pred_key in enumerate(self.pred_keys):
            this_preds = filter_predictions(self.results[pred_key], scene_id, view_id)
            rgb_input, pred_rendered = make_singleview_prediction_matplotlib(self.scene_ds, self.renderer, this_preds)
            ax[pred_id][1].title.set_text(pred_key)
            ax[pred_id][0].imshow(rgb_input)
            ax[pred_id][1].imshow(pred_rendered)
        plt.savefig(self.VIZ_DATA_DIR / f'viz_{scene_id}_{view_id}.png')


if __name__ == '__main__':
    #baseline
    # viz = VisualizeSingleView('bracket_assembly', 'bracket_assembly-n_views=1--3211443897')
    #
    viz = VisualizeSingleView('bracket_assembly_nut', 'bracket_assembly_nut-n_views=1--684291998')
    import random
    #Generate 5 random numbers between 10 and 30
    random.seed(0)
    random_scenes = random.choices([0,1,2], k=5)
    random_views = random.sample(range(0, 100), 60)
    print(random_scenes, random_views)

    # scene_id, view_id = 2, 26
    # viz.visualize_single(scene_id, view_id)
    for scene_id in random_scenes:
        for view_id in random_views:
            # print(scene_id, view_id)
            try:
                viz.visualize_single(scene_id, view_id)
                print("Sucess",scene_id, view_id)
            except Exception as e:
                print("Failed",scene_id, view_id, e)