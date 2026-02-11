import copy
import json
import sys
from contextlib import contextmanager
import torch

import numpy as np

from PIL import Image
from pathlib import Path

from pytorch3d.ops import box3d_overlap

from pyquaternion import Quaternion as Q

from .simbev_utils import *

from ..dataset import DatasetTemplate

from ...utils.box_utils import boxes_to_corners_3d


CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

OBJECT_CLASSES = {
    12: 'pedestrian',
    14: 'car',
    15: 'truck',
    16: 'bus',
    18: 'motorcycle',
    19: 'bicycle'
}


class SimBEVDataset(DatasetTemplate):
    '''
    This class serves as the API for experiments on the SimBEV dataset.

    Args:
        dataset_cfg: configuration used for the dataset.
        class_names: list of class names.
        training: whether the dataset is used for training or evaluation.
        root_path: path to the root directory of the dataset.
        logger: logger object for logging information.
    '''
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None
    ):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger
        )

        self.infos = []

        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)

        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
            self.pad_to_max_shape = self.camera_image_config.get('PAD_TO_MAX_SHAPE', False)
            self.pad_value = int(self.camera_image_config.get('PAD_VALUE', 0))
            self.pad_align = self.camera_image_config.get('PAD_ALIGN', 'center')
        else:
            self.use_camera = False
            self.pad_to_max_shape = False
        
        self.map_config = self.dataset_cfg.get('MAP_CONFIG', None)
        
        if self.map_config is not None:
            self.use_map = self.map_config.get('USE_MAP', True)
            self.map_classes = self.map_config.CLASS_NAMES
        else:
            self.use_map = False

        self.infos = self.load_annotations(self.mode)

        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_resampling(self.infos)

    def load_annotations(self, mode):
        '''
        Load annotations for the desired run mode.

        Args:
            mode: run mode.

        Returns:
            data_infos: list of data samples in the dataset.
        '''
        info_path = self.root_path / self.dataset_cfg.INFO_PATH[mode][0]

        with open(info_path, 'r') as f:
            annotations = json.load(f)

        data_infos = []

        for key in annotations['data']:
            data_infos += annotations['data'][key]['scene_data']

        data_infos = self.normalize_info_paths(data_infos)
        
        self.full_infos = data_infos
        
        data_infos = data_infos[::self.dataset_cfg.LOAD_INTERVAL]

        self.metadata = annotations['metadata']

        data_infos = self.load_gt_bboxes(data_infos)

        data_infos = self.load_sweep_paths(data_infos)

        self.logger.info(f'Total samples of the SimBEV dataset used: {len(data_infos)}')

        return data_infos

    def normalize_info_paths(self, infos):
        """
        Normalize file paths from SimBEV annotations so they can work across
        different local mount points.
        """
        path_keys = (
            'LIDAR', 'GT_DET', 'GT_SEG', 'GT_SEG_VIZ',
            'RGB-', 'SEG-', 'IST-', 'DPT-', 'FLW-',
            'RAD_', 'GNSS', 'IMU'
        )

        for info in infos:
            for key, value in info.items():
                if not isinstance(value, str):
                    continue
                if not key.startswith(path_keys) and key not in path_keys:
                    continue
                info[key] = self.resolve_data_path(value)

        return infos

    def resolve_data_path(self, path_str):
        """
        Resolve absolute/relative paths from annotation files to the current
        DATA_PATH layout.
        """
        raw_path = Path(path_str)

        # Already valid as-is.
        if raw_path.exists():
            return str(raw_path)

        # Relative path in annotation: resolve against dataset root.
        if not raw_path.is_absolute():
            return str(self.root_path / raw_path)

        # Absolute path from another machine/mount point:
        # use stable SimBEV subfolders under the configured root path.
        marker_map = {
            'ground-truth': ['ground-truth', 'ground_truth'],
            'ground_truth': ['ground-truth', 'ground_truth'],
            'sweeps': ['sweeps', 'samples'],
            'samples': ['samples', 'sweeps'],
            'infos': ['infos']
        }
        for marker, mapped_markers in marker_map.items():
            if marker in raw_path.parts:
                idx = raw_path.parts.index(marker)
                suffix_parts = raw_path.parts[idx + 1:]
                for mapped_marker in mapped_markers:
                    mapped_parts = (mapped_marker,) + suffix_parts
                    mapped_path = self.root_path / Path(*mapped_parts)
                    if mapped_path.exists():
                        return str(mapped_path)
                mapped_parts = (mapped_markers[0],) + suffix_parts
                return str(self.root_path / Path(*mapped_parts))

        return str(raw_path)

    @staticmethod
    def _get_lidar_info_key(info):
        if 'LIDAR' in info:
            return 'LIDAR'
        if 'LIDAR_TOP' in info:
            return 'LIDAR_TOP'
        raise KeyError('No lidar key found in sample info. Expected LIDAR or LIDAR_TOP.')

    def _get_lidar_metadata_key(self):
        if 'LIDAR' in self.metadata:
            return 'LIDAR'
        if 'LIDAR_TOP' in self.metadata:
            return 'LIDAR_TOP'
        raise KeyError('No lidar key found in metadata. Expected LIDAR or LIDAR_TOP.')

    def _get_camera_names(self, info):
        rgb_cameras = sorted(
            [key[4:] for key in info.keys() if key.startswith('RGB-')]
        )
        camera_names = [name for name in rgb_cameras if name in self.metadata]
        if camera_names:
            return camera_names
        return [name for name in CAM_NAME if ('RGB-' + name) in info and name in self.metadata]

    def _get_camera_intrinsics(self, camera):
        camera_intrinsics_by_name = self.metadata.get('camera_intrinsics_by_name', None)
        if camera_intrinsics_by_name is not None and camera in camera_intrinsics_by_name:
            return np.array(camera_intrinsics_by_name[camera], dtype=np.float32)
        return np.array(self.metadata['camera_intrinsics'], dtype=np.float32)
    
    def load_gt_bboxes(self, infos):
        '''
        Load ground truth bounding boxes from file into the list of data
        samples.

        Args:
            infos: list of data samples in the dataset.
        
        Returns:
            infos: list of data samples updated with ground truth bounding
                boxes.
        '''
        for info in infos:
            gt_boxes = []
            gt_names = []
            gt_velocities = []
            
            num_lidar_pts = []
            num_radar_pts = []
            
            valid_flag = []

            # Load ground truth bounding boxes from file.
            gt_det_path = info['GT_DET']

            gt_det = self.safe_load_pickled_npy(gt_det_path)

            # Ego to global transformation.
            ego2global = np.eye(4).astype(np.float32)
            
            ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
            ego2global[:3, 3] = info['ego2global_translation']

            # Lidar to ego transformation.
            lidar2ego = np.eye(4).astype(np.float32)
            lidar_meta_key = self._get_lidar_metadata_key()
            lidar2ego[:3, :3] = Q(self.metadata[lidar_meta_key]['sensor2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = self.metadata[lidar_meta_key]['sensor2ego_translation']

            global2lidar = np.linalg.inv(ego2global @ lidar2ego)

            global2lidarrot = np.eye(4).astype(np.float32)
            
            global2lidarrot[:3, :3] = global2lidar[:3, :3]

            # Transform bounding boxes from the global coordinate system to
            # the lidar coordinate system.
            for det_object in gt_det:
                for tag in det_object['semantic_tags']:
                    if tag in OBJECT_CLASSES.keys():
                        global_bbox_corners = np.append(det_object['bounding_box'], np.ones((8, 1)), 1)
                        bbox_corners = (global2lidar @ global_bbox_corners.T)[:3].T

                        # Calculate the center of the bounding box.
                        center = ((bbox_corners[0] + bbox_corners[7]) / 2).tolist()

                        # Calculate the dimensions of the bounding box.
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[2]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[4]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[1]))

                        # Calculate the yaw angle of the bounding box.
                        diff = bbox_corners[0] - bbox_corners[2]
                        
                        gamma = np.arctan2(diff[1], diff[0])

                        center.append(-gamma)

                        gt_boxes.append(center)
                        gt_names.append(OBJECT_CLASSES[tag])
                        gt_velocities.append(
                            (global2lidarrot @ np.append(det_object['linear_velocity'], [1]))[:3].tolist()
                        )
                        
                        num_lidar_pts.append(det_object['num_lidar_pts'])
                        num_radar_pts.append(det_object['num_radar_pts'])
                        
                        valid_flag.append(det_object['valid_flag'])

            info['gt_boxes'] = np.array(gt_boxes)
            info['gt_names'] = np.array(gt_names)
            info['gt_velocity'] = np.array(gt_velocities)

            info['num_lidar_pts'] = np.array(num_lidar_pts)
            info['num_radar_pts'] = np.array(num_radar_pts)
            
            info['valid_flag'] = np.array(valid_flag)

        return infos

    @staticmethod
    @contextmanager
    def numpy_pickle_compat():
        """
        Temporary compatibility aliases for object arrays pickled with
        module path `numpy._core.*`.
        """
        added_keys = []
        try:
            if 'numpy._core' not in sys.modules:
                sys.modules['numpy._core'] = np.core
                added_keys.append('numpy._core')

            if 'numpy._core.multiarray' not in sys.modules:
                sys.modules['numpy._core.multiarray'] = np.core.multiarray
                added_keys.append('numpy._core.multiarray')

            yield
        finally:
            for key in added_keys:
                sys.modules.pop(key, None)

    def safe_load_pickled_npy(self, file_path):
        """
        Load object-array npy files with a fallback for numpy module rename
        compatibility.
        """
        try:
            return np.load(file_path, allow_pickle=True)
        except ModuleNotFoundError as err:
            if 'numpy._core' not in str(err):
                raise
            with self.numpy_pickle_compat():
                return np.load(file_path, allow_pickle=True)
    
    def load_sweep_paths(self, infos):
        '''
        Add sweep paths to the list of data samples.

        Args:
            infos: list of data samples in the dataset.
        
        Returns:
            infos: list of data samples updated with sweep paths.
        '''
        for index in range(len(infos)):
            info = infos[index]
            
            info['sweeps_lidar_paths'] = []
            info['sweeps_ego2global'] = []

            for i in range(self.dataset_cfg.MAX_SWEEPS):
                if info['frame'] - (i + 1) >= 0:
                    sweep_info = self.full_infos[self.dataset_cfg.LOAD_INTERVAL * index - (i + 1)]

                    sweep_lidar_key = self._get_lidar_info_key(sweep_info)
                    info['sweeps_lidar_paths'].append(sweep_info[sweep_lidar_key])

                    ego2global = np.eye(4).astype(np.float32)
            
                    ego2global[:3, :3] = Q(sweep_info['ego2global_rotation']).rotation_matrix
                    ego2global[:3, 3] = sweep_info['ego2global_translation']

                    info['sweeps_ego2global'].append(ego2global)
        
        return infos
    
    def balanced_resampling(self, infos):
        '''
        Class-balanced sampling of the dataset 
        (from https://arxiv.org/abs/1908.09492).

        Args:
            infos: list of data samples in the dataset.

        Returns:
            balanced_infos: balanced list of data samples.
        '''
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        balanced_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            balanced_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        
        self.logger.info(f'Total samples after balanced resampling: {len(balanced_infos)}')

        return balanced_infos

    def get_data_info(self, index):
        '''
        Package information for a data sample.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            data: packaged information for the sample.
        '''
        info = self.infos[index]

        lidar_info_key = self._get_lidar_info_key(info)
        data = dict(
            scene = info['scene'],
            frame = info['frame'],
            timestamp = info['timestamp'],
            gt_seg_path = info['GT_SEG'],
            gt_det_path = info['GT_DET'],
            lidar_path = info[lidar_info_key],
            sweeps_lidar_paths = info['sweeps_lidar_paths'],
            sweeps_ego2global = info['sweeps_ego2global'],
            gt_boxes = info['gt_boxes'],
            gt_names = info['gt_names'],
            gt_velocity = info['gt_velocity'],
            num_lidar_pts = info['num_lidar_pts'],
            num_radar_pts = info['num_radar_pts'],
            valid_flag = info['valid_flag']
        )

        # Ego to global transformation.
        ego2global = np.eye(4).astype(np.float32)
        
        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        
        data['ego2global'] = ego2global

        # Lidar to ego transformation.
        lidar2ego = np.eye(4).astype(np.float32)
        lidar_meta_key = self._get_lidar_metadata_key()
        lidar2ego[:3, :3] = Q(self.metadata[lidar_meta_key]['sensor2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = self.metadata[lidar_meta_key]['sensor2ego_translation']
        
        data['lidar2ego'] = lidar2ego

        if self.use_camera:
            data['image_paths'] = []
            data['camera_intrinsics'] = []
            data['camera2lidar'] = []
            data['lidar2camera'] = []
            data['lidar2image'] = []
            data['camera2ego'] = []

            camera_names = self._get_camera_names(info)
            if len(camera_names) == 0:
                raise KeyError('No valid RGB camera entries found in sample info for current metadata.')
            data['camera_names'] = camera_names
            for camera in camera_names:
                data['image_paths'].append(info['RGB-' + camera])

                # Camera intrinsics.
                camera_intrinsics = np.eye(4).astype(np.float32)

                camera_intrinsics[:3, :3] = self._get_camera_intrinsics(camera)
                
                data['camera_intrinsics'].append(camera_intrinsics)
                
                # Lidar to camera transformation.
                camera2lidar = np.eye(4).astype(np.float32)

                camera2lidar[:3, :3] = Q(self.metadata[camera]['sensor2lidar_rotation']).rotation_matrix
                camera2lidar[:3, 3] = self.metadata[camera]['sensor2lidar_translation']

                data['camera2lidar'].append(camera2lidar)

                lidar2camera = np.linalg.inv(camera2lidar)
                
                data['lidar2camera'].append(lidar2camera)

                # Lidar to image transformation.
                lidar2image = camera_intrinsics @ lidar2camera

                data['lidar2image'].append(lidar2image)

                # Camera to ego transformation.
                camera2ego = np.eye(4).astype(np.float32)

                camera2ego[:3, :3] = Q(self.metadata[camera]['sensor2ego_rotation']).rotation_matrix
                camera2ego[:3, 3] = self.metadata[camera]['sensor2ego_translation']

                data['camera2ego'].append(camera2ego)
        
        return data

    def pad_camera_images(self, input_dict):
        imgs = input_dict['camera_imgs']
        if len(imgs) == 0:
            return input_dict

        sizes = [img.size for img in imgs]
        max_w = max(size[0] for size in sizes)
        max_h = max(size[1] for size in sizes)
        if all(w == max_w and h == max_h for (w, h) in sizes):
            return input_dict

        if self.pad_align != 'center':
            raise ValueError(f'Unsupported PAD_ALIGN "{self.pad_align}". Only "center" is supported.')

        padded_images = []
        for idx, img in enumerate(imgs):
            cur_w, cur_h = img.size
            pad_left = (max_w - cur_w) // 2
            pad_top = (max_h - cur_h) // 2

            canvas = Image.new(img.mode, (max_w, max_h), color=(self.pad_value, self.pad_value, self.pad_value))
            canvas.paste(img, (pad_left, pad_top))
            padded_images.append(canvas)

            input_dict['camera_intrinsics'][idx][0, 2] += pad_left
            input_dict['camera_intrinsics'][idx][1, 2] += pad_top
            input_dict['lidar2image'][idx] = input_dict['camera_intrinsics'][idx] @ input_dict['lidar2camera'][idx]

        input_dict['camera_imgs'] = padded_images

        return input_dict

    def crop_image(self, input_dict):
        '''
        Crop images in the input dictionary to the desired dimensions.

        Args:
            input_dict: input dictionary containing images to be cropped.

        Returns:
            input_dict: input dictionary updated with cropped images.
        '''
        W, H = input_dict["ori_shape"]
        
        imgs = input_dict["camera_imgs"]
        
        input_dict['ori_imgs'] = [np.array(img) for img in imgs]
        
        img_process_infos = []
        crop_images = []
        
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                
                resize = np.random.uniform(*resize_lim)
                
                resize_dims = (int(W * resize), int(H * resize))
                
                newW, newH = resize_dims
                
                crop_h = newH - fH
                
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                
                resize = np.mean(resize_lim)
                
                resize_dims = (int(W * resize), int(H * resize))
                
                newW, newH = resize_dims
                
                crop_h = newH - fH
                
                crop_w = int(max(0, newW - fW) / 2)
                
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # Resize and crop image.
            img = img.resize(resize_dims)
            img = img.crop(crop)

            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        
        return input_dict
    
    def evaluation(self, results, class_names, eval_metric='iou', **kwargs):
        metrics = {}
        
        simbev_eval = SimBEVDetectionEval(results, self.class_names, eval_metric)

        metrics.update(simbev_eval.evaluate())

        return str(metrics), metrics
    
    def evaluation_map_segmentation(self, results):
        '''
        Evaluate BEV map segmentation results.

        Args:
            results: BEV map segmentation results from the model.
        
        Returns:
            metrics: evaluation metrics for BEV map segmentation results.
        '''
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device=device)

        data_processors = self.dataset_cfg.get('DATA_PROCESSOR', None)

        for cfg in data_processors:
            if cfg.NAME == 'load_simbev_bev_segmentation':
                xDim = cfg.bev_dim_x
                xRes = cfg.bev_res_x
        
        yDim = xDim
        yRes = xRes

        # Calculate the center-point coordinates of the BEV grid cells.
        xLim = xDim * xRes / 2
        yLim = yDim * yRes / 2
        
        cxLim = xLim - xRes / 2
        cyLim = yLim - yRes / 2

        x = torch.linspace(cxLim, -cxLim, xDim, device=device)
        y = torch.linspace(cyLim, -cyLim, yDim, device=device)

        xx, yy = torch.meshgrid(x, y, indexing='ij')

        coordinates = torch.stack([xx, yy], dim=2).reshape(-1, 2)

        distance = torch.linalg.norm(coordinates, dim=1)

        masks = torch.zeros(distance.shape[0], 4, dtype=torch.bool, device=device)

        masks[:, 0] = distance >= 0.0
        masks[:, 1] = distance <= 20.0
        masks[:, 2] = (distance <= 40.0) & (distance > 20.0)
        masks[:, 3] = distance > 40.0

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds, 4, device=device)
        fp = torch.zeros(num_classes, num_thresholds, 4, device=device)
        fn = torch.zeros(num_classes, num_thresholds, 4, device=device)

        confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

        for result in results:
            pred = result['masks_bev'].to(device)
            label = result['gt_masks_bev'].to(device)

            pred_labels = (pred >= 0.5)
            
            # Update the confusion matrix.
            for gt_class in range(num_classes):
                for pred_class in range(num_classes):
                    confusion_matrix[gt_class, pred_class] += \
                        ((label[gt_class] == 1) & (pred_labels[pred_class] == 1)).sum()

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            for i in range(4):
                mask_i = masks[:, i]
                
                tp[:, :, i] += (pred & label)[:, mask_i, :].sum(dim=1)
                fp[:, :, i] += (pred & ~label)[:, mask_i, :].sum(dim=1)
                fn[:, :, i] += (~pred & label)[:, mask_i, :].sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-6)
        
        metrics = {}
        
        for index, name in enumerate(self.map_classes):
            metrics[f'map/{name}/IoU@max'] = ious[index, :, 0].max().item()
            
            for threshold, iou in zip(thresholds, ious[index, :, 0]):
                metrics[f'map/{name}/IoU@{threshold.item():.2f}'] = iou.item()
        
        metrics['map/mean/IoU@max'] = ious[:, :, 0].max(dim=1).values.mean().item()

        for index, threshold in enumerate(thresholds):
            metrics[f'map/mean/IoU@{threshold.item():.2f}'] = ious[:, index, 0].mean().item()
        
        # Print IoU table.
        table_headings = ['Overall IoUs', '0-20m', '20-40m', '>40m']

        for i in range(4):
            print(f'\n{"-" * 40} {table_headings[i]} {"-" * 40}')
            print('\n\n')

            print(f'{"IoU":<12} {0.1:<8}{0.2:<8}{0.3:<8}{0.4:<8}{0.5:<8}{0.6:<8}{0.7:<8}{0.8:<8}{0.9:<8}')

            for index, name in enumerate(self.map_classes):
                print(f'{name:<12}', ''.join([f'{iou:<8.4f}' for iou in ious[index, :, i].tolist()]))

            print(f'{"mIoU":<12}', ''.join([f'{iou:<8.4f}' for iou in ious[:, :, i].mean(dim=0).tolist()]), '\n')

        print(f'\n{"-" * 40} {"Confusion Matrix"} {"-" * 40}')
        print('\n\n')

        print(f'{"":<12}', ''.join([f'{name:<12}' for name in self.map_classes]))

        for index, name in enumerate(self.map_classes):
            print(f'{name:<12}', ''.join([f'{confusion_matrix[index, j]:<12.0f}' for j in range(num_classes)]))
        
        print('\n\n')
        
        return metrics
    
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.get_data_info(index))

        lidar_loader = LoadSimBEVPointsFromFile(self.dataset_cfg.LIDAR_TRIM_STEP)
        lidar_sweeps_loader = LoadSimBEVPointsFromMultiSweeps(
            self.dataset_cfg.LIDAR_SWEEPS,
            self.dataset_cfg.LIDAR_TRIM_STEP,
            self.dataset_cfg.SIMBEV_TIMESTEP,
            self.training
        )

        info = lidar_sweeps_loader(lidar_loader(info))

        input_dict = {
            'points': info['points'],
            'frame_id': info['lidar_path'],
            'metadata': {
                'scene': info['scene'],
                'frame': info['frame'],
                'lidar2ego': info['lidar2ego'],
                'ego2global': info['ego2global'],
                'gt_seg_path': info['gt_seg_path'],
                'is_train': self.training
            }
        }

        info['gt_boxes'] = np.concatenate((info['gt_boxes'], info['gt_velocity'][:, :2]), axis=1)

        if self.use_map:
            input_dict['ref_from_car'] = np.linalg.inv(info['lidar2ego'])
            input_dict['car_from_global'] = np.linalg.inv(info['ego2global'])

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (
                    (info['num_lidar_pts'] + info['num_radar_pts'] + 1) > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT
                )
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

            gt_labels = []

            for class_name in input_dict['gt_names']:
                if class_name in self.class_names:
                    gt_labels.append(self.class_names.index(class_name))
                else:
                    gt_labels.append(-1)

            input_dict['metadata'].update({
                'gt_names': input_dict['gt_names'],
                'gt_boxes': input_dict['gt_boxes'],
                'gt_labels': np.array(gt_labels)
            })
        
        if self.use_camera:
            input_dict['image_paths'] = info['image_paths']
            input_dict['camera_intrinsics'] = info['camera_intrinsics']
            input_dict['camera2lidar'] = info['camera2lidar']
            input_dict['lidar2camera'] = info['lidar2camera']
            input_dict['lidar2image'] = info['lidar2image']
            input_dict['camera2ego'] = info['camera2ego']

            images = []

            for name in input_dict['image_paths']:
                images.append(Image.fromarray(np.array(Image.open(name))[:,:,::-1]))
            
            input_dict['camera_imgs'] = images
            if self.pad_to_max_shape:
                input_dict = self.pad_camera_images(input_dict)
            input_dict['ori_shape'] = input_dict['camera_imgs'][0].size

            input_dict = self.crop_image(input_dict)
        
        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            
            gt_boxes[np.isnan(gt_boxes)] = 0
            
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
            
        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)


class SimBEVDetectionEval:
    '''
    Class for evaluating 3D object detection results on the SimBEV dataset.

    Args:
        results: results from the model.
        classes: list of object classes in the dataset.
        mode: evalution mode, can be 'iou' or 'distance'.
    '''
    def __init__(self, results, classes, mode='iou'):
        self.results = results
        self.classes = classes
        self.mode = mode

        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]

        if self.mode == 'iou':
            self.thresholds = iou_thresholds
        elif self.mode == 'distance':
            self.thresholds = distance_thresholds
        else:
            raise ValueError(f'Unsupported evaluation mode {self.mode}.')

    def evaluate(self):
        '''
        Evaluate 3D object detection results.
        '''
        num_classes = len(self.classes)
        num_thresholds = len(self.thresholds)

        # Dictionary to store Average Precision (AP), Average Translation
        # Error (ATE), Average Orientation Error (AOE), Average Scale Error
        # (ASE), and Average Velocity Error (AVE) for each class and IoU
        # threshold.
        det_metrics = {
            item: torch.zeros((num_classes, num_thresholds)) for item in ['AP', 'ATE', 'AOE', 'ASE', 'AVE']
        }

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print('\n')
        
        for k, threshold in enumerate(self.thresholds):
            print(f'Calculating metrics for threshold {threshold}...')

            # Dictionaries to store True Positive (TP) and False Positive (FP)
            # values, scores, ATE, AOE, ASE, AVE, and the total number of
            # ground truth boxes for each class.
            tps = {i: torch.empty((0, )) for i in range(num_classes)}
            fps = {i: torch.empty((0, )) for i in range(num_classes)}

            scores = {i: torch.empty((0, )) for i in range(num_classes)}

            ate = {i: torch.empty((0, )) for i in range(num_classes)}
            aoe = {i: torch.empty((0, )) for i in range(num_classes)}
            ase = {i: torch.empty((0, )) for i in range(num_classes)}
            ave = {i: torch.empty((0, )) for i in range(num_classes)}

            num_gt_boxes = {i: 0 for i in range(num_classes)}

            # Iterate over predictions for each sample.
            for result in self.results:
                boxes_3d = torch.from_numpy(result['boxes_lidar']).to(torch.float32)
                scores_3d = torch.from_numpy(result['score']).to(torch.float32)
                labels_3d = torch.from_numpy(result['pred_labels'] - 1).to(torch.float32)
                gt_boxes_3d = torch.from_numpy(result['metadata']['gt_boxes']).to(torch.float32)
                gt_labels_3d = torch.from_numpy(result['metadata']['gt_labels']).to(torch.float32)

                if self.mode == 'iou':
                    if boxes_3d.shape[0] > 0:
                        boxes_3d_corners = boxes_to_corners_3d(boxes_3d)
                    else:
                        boxes_3d_corners = torch.empty((0, 8, 3))

                    if gt_boxes_3d.shape[0] > 0:
                        gt_boxes_3d_corners = boxes_to_corners_3d(gt_boxes_3d)
                    else:
                        gt_boxes_3d_corners = torch.empty((0, 8, 3))
                else:
                    boxes_3d_centers = boxes_3d[:, :3]

                    gt_boxes_3d_centers = gt_boxes_3d[:, :3]

                for cls in range(num_classes):
                    pred_mask = labels_3d == cls
                    
                    gt_mask = gt_labels_3d == cls

                    pred_boxes = boxes_3d[pred_mask]
                    
                    if self.mode == 'iou':
                        pred_box_corners = boxes_3d_corners[pred_mask]
                    else:
                        pred_box_centers = boxes_3d_centers[pred_mask]
                    
                    pred_scores = scores_3d[pred_mask]
                    
                    gt_boxes = gt_boxes_3d[gt_mask]

                    if self.mode == 'iou':
                        gt_box_corners = gt_boxes_3d_corners[gt_mask]
                    else:
                        gt_box_centers = gt_boxes_3d_centers[gt_mask]

                    # Sort predictions by confidence score in descending
                    # order.
                    sorted_indices = torch.argsort(-pred_scores)

                    pred_boxes = pred_boxes[sorted_indices]

                    if self.mode == 'iou':
                        pred_box_corners = pred_box_corners[sorted_indices]
                    else:
                        pred_box_centers = pred_box_centers[sorted_indices]
                    
                    pred_scores = pred_scores[sorted_indices]

                    if self.mode == 'iou':
                        pred_box_corners = pred_box_corners.to(device)
                        gt_box_corners = gt_box_corners.to(device)
                    else:
                        pred_box_centers = pred_box_centers.to(device)
                        gt_box_centers = gt_box_centers.to(device)
                    
                    if self.mode == 'iou':
                        # Calculate Intersection over Union (IoU) between
                        # predicted and ground truth bounding boxes.
                        if len(pred_box_corners) == 0:
                            ious = torch.zeros((0, len(gt_box_corners)), device=device)
                        elif len(gt_box_corners) == 0:
                            ious = torch.zeros((len(pred_box_corners), 0), device=device)
                        else:
                            _, ious = box3d_overlap(pred_box_corners, gt_box_corners)
                    else:
                        # Calculate Euclidean distance between predicted and
                        # ground truth bounding box centers.
                        dists = torch.cdist(pred_box_centers, gt_box_centers)

                    # Tensor to keep track of ground truth boxes that have
                    # been assigned to a prediction.
                    assigned_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

                    tp = torch.zeros(len(pred_boxes))
                    fp = torch.zeros(len(pred_boxes))                  

                    ate_local = []
                    aoe_local = []
                    ase_local = []
                    ave_local = []

                    for i, pred_box in enumerate(pred_boxes):                        
                        matched = False
                        matched_gt_idx = -1
                        
                        if self.mode == 'iou':
                            # Among the ground truth bounding boxes that have not
                            # been matched to a prediction yet, find the one with
                            # the highest IoU value.
                            available_ious = ious[i] * ~assigned_gt

                            if available_ious.shape[0] > 0:
                                iou_max, max_gt_idx = available_ious.max(dim=0)
                                max_gt_idx = max_gt_idx.item()
                            else:
                                iou_max = 0
                                max_gt_idx = -1

                            if iou_max >= threshold:
                                matched = True
                                matched_gt_idx = max_gt_idx
                        else:
                            # Among the ground truth bounding boxes that have not
                            # been matched to a prediction yet, find the one with
                            # the smallest Euclidean distance.
                            available_dists = 10000 - ((10000 - dists[i]) * ~assigned_gt)    

                            if available_dists.shape[0] > 0:
                                dist_min, min_gt_idx = available_dists.min(dim=0)
                                min_gt_idx = min_gt_idx.item()
                            else:
                                dist_min = 10000
                                min_gt_idx = -1
                            
                            if dist_min <= threshold:
                                matched = True
                                matched_gt_idx = min_gt_idx
                        
                        if matched:
                            tp[i] = 1

                            assigned_gt[matched_gt_idx] = True

                            # Calculate ATE, which is the Euclidean distance
                            # between the predicted and ground truth bounding
                            # box centers.
                            ate_local.append(
                                torch.linalg.vector_norm(pred_boxes[i, :3] - gt_boxes[matched_gt_idx, :3])
                            )

                            # Calculate AOE, which is the smallest yaw angle
                            # between the predicted and ground truth bounding
                            # boxes.
                            diff_angle = (
                                gt_boxes[matched_gt_idx, 6] - pred_boxes[i, 6] + np.pi
                            ) % (2 * np.pi) - np.pi

                            # Ensure the angle difference is between -pi and
                            # pi.
                            if diff_angle > np.pi:
                                diff_angle = diff_angle - 2 * np.pi

                            aoe_local.append(abs(diff_angle))

                            # Calculate ASE, which is defined as 1 - IOU after
                            # the predicted and ground truth bounding boxes
                            # are translated and rotated to have the same
                            # center and orientation.
                            pred_wlh = pred_boxes[i, 3:6]
                            gt_wlh = gt_boxes[matched_gt_idx, 3:6]

                            min_wlh = torch.minimum(pred_wlh, gt_wlh)

                            pred_vol = torch.prod(pred_wlh)
                            gt_vol = torch.prod(gt_wlh)
                            
                            intersection = torch.prod(min_wlh)

                            union = pred_vol + gt_vol - intersection

                            ase_local.append(1 - intersection / union)

                            # Calculate AVE, which is the L2 norm of the
                            # difference between the predicted and ground
                            # truth bounding box velocities.
                            ave_local.append(
                                torch.linalg.vector_norm(pred_boxes[i, -2:] - gt_boxes[matched_gt_idx, -2:])
                            )
                        else:
                            fp[i] = 1
                    
                    tps[cls] = torch.cat((tps[cls], tp))
                    fps[cls] = torch.cat((fps[cls], fp))

                    scores[cls] = torch.cat((scores[cls], pred_scores))

                    ate[cls] = torch.cat((ate[cls], torch.Tensor(ate_local)))
                    aoe[cls] = torch.cat((aoe[cls], torch.Tensor(aoe_local)))
                    ase[cls] = torch.cat((ase[cls], torch.Tensor(ase_local)))
                    ave[cls] = torch.cat((ave[cls], torch.Tensor(ave_local)))

                    num_gt_boxes[cls] += len(gt_boxes)

            for cls in range(num_classes):
                # Sort TP and FP values by confidence score in descending
                # order.
                sorted_indices = torch.argsort(-scores[cls])

                tps[cls] = tps[cls][sorted_indices]
                fps[cls] = fps[cls][sorted_indices]

                tps[cls] = torch.cumsum(tps[cls], dim=0, dtype=torch.float32)
                fps[cls] = torch.cumsum(fps[cls], dim=0, dtype=torch.float32)

                recalls = tps[cls] / num_gt_boxes[cls]
                precisions = tps[cls] / (tps[cls] + fps[cls])

                # Add the (0, 1) point to the precision-recall curve.
                recalls = torch.cat((torch.Tensor([0.0]), recalls))
                precisions = torch.cat((torch.Tensor([1.0]), precisions))

                # AP is the area under the precision-recall curve.
                det_metrics['AP'][cls, k] = torch.trapz(precisions, recalls)

                for item, value in zip(['ATE', 'AOE', 'ASE', 'AVE'], [ate, aoe, ase, ave]):
                    det_metrics[item][cls, k] = value[cls].mean()

        metrics = {}

        mean_metrics = {}

        print('\n')

        for item in ['AP', 'ATE', 'AOE', 'ASE', 'AVE']:
            for index, name in enumerate(self.classes):
                metrics[f'det/{name}/{item}@max'] = det_metrics[item][index].max().item()
                metrics[f'det/{name}/{item}@mean'] = det_metrics[item][index].nanmean().item()

                for threshold, value in zip(self.thresholds, det_metrics[item][index]):
                    metrics[f'det/{name}/{item}@{threshold:.2f}'] = value.item()
        
            for index, threshold in enumerate(self.thresholds):
                metrics[f'det/mean/{item}@{threshold:.2f}'] = det_metrics[item][:, index].nanmean().item()
            
            if self.mode == 'iou':
                print(f'{item:<12} {0.1:<8}{0.2:<8}{0.3:<8}{0.4:<8}{0.5:<8}{0.6:<8}{0.7:<8}{0.8:<8}{0.9:<8} {"mean":<8}')
            else:
                print(f'{item:<12} {0.5:<8}{1.0:<8}{2.0:<8}{4.0:<8} {"mean":<8}')

            for index, name in enumerate(self.classes):
                print(
                    f'{name:<12}',
                    ''.join([f'{value:<8.4f}' for value in det_metrics[item][index].tolist()]),
                    f'{det_metrics[item][index].nanmean().item():<8.4f}'
                )
            
            print(
                f'm{item:<11}',
                ''.join([f'{value:<8.4f}' for value in det_metrics[item].nanmean(dim=0).tolist()]),
                '\n'
            )

            if self.mode == 'iou':
                mean_metrics[f'm{item}'] = det_metrics[item][:, 2:].nanmean().item()
            else:
                mean_metrics[f'm{item}'] = det_metrics[item].nanmean().item()

            metrics[f'det/m{item}'] = mean_metrics[f'm{item}']

            print(f'm{item}: ', mean_metrics[f'm{item}'], '\n')

        mATE = max(0.0, 1 - mean_metrics['mATE'])
        mAOE = max(0.0, 1 - mean_metrics['mAOE'])
        mASE = max(0.0, 1 - mean_metrics['mASE'])
        mAVE = max(0.0, 1 - mean_metrics['mAVE'])

        SimBEVDetectionScore = (4 * mean_metrics['mAP'] + mATE + mAOE + mASE + mAVE) / 8

        metrics['det/SDS'] = SimBEVDetectionScore

        print('SDS: ', SimBEVDetectionScore, '\n')
        
        return metrics
