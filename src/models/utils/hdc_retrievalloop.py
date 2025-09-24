# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import TestLoop, ValLoop, autocast

from mmpretrain.registry import LOOPS
import gc
import os
import shutil
import copy

@LOOPS.register_module()
class HDCRetrievalValLoop(ValLoop):

    def __init__(self,
                 runner,
                 dataloader,
                 evaluator,
                 i2t: bool = True,
                 fp16: bool = False,
                 load_cpu: bool = False,
                 fast_datainfo: bool = False,
                 cache_dir: str = None) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.load_device = torch.device("cpu") if load_cpu else torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.fast_datainfo = fast_datainfo
        self.cache_dir = cache_dir
        self.i2t = i2t

    def _get_features_info(self):
        """Get feature shapes and dtypes from first batch."""
        dataloader_len = len(self.dataloader)
        feats_local_shape = None
        first_batch_size = None
        first_batch_dtype = {}

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor
                    
                    if idx == 0 or idx == dataloader_len - 1:
                        data_batch = data_preprocessor(data_batch, False)
                        feats = self.runner.model._run_forward(data_batch, mode='tensor')
                        if idx == 0:
                            feats_local_shape = {k: list(v.shape) for k, v in feats.items() if isinstance(v, torch.Tensor)}
                            first_batch_size = {k: v.shape[0] for k, v in feats.items() if isinstance(v, torch.Tensor)}
                            first_batch_dtype = {k: v.dtype for k, v in feats.items() if isinstance(v, torch.Tensor)}
                        else:
                            for k, v in feats.items():
                                feats_local_shape[k][0] += v.shape[0]
                    else:
                        if self.fast_datainfo:
                            for k in feats_local_shape:
                                feats_local_shape[k][0] += first_batch_size[k]
                        else:
                            data_batch = data_preprocessor(data_batch, False)
                            feats = self.runner.model._run_forward(data_batch, mode='tensor')
                            for k, v in feats.items():
                                feats_local_shape[k][0] += v.shape[0]

        return {
            'shapes': feats_local_shape,
            'dtypes': first_batch_dtype,
        }

    def _get_features(self, feats_info):
        """Get or generate features, with optional caching."""
        if self.cache_dir:
            feats_file = os.path.join(self.cache_dir, 'feats.pt')
            samples_file = os.path.join(self.cache_dir, 'data_samples.pt')

            # Try loading cached features
            if os.path.exists(feats_file) and os.path.exists(samples_file):
                print(f"Loading cached features from {self.cache_dir}")
                feats_local = torch.load(feats_file, map_location=self.load_device)
                data_samples_local = torch.load(samples_file, map_location=self.load_device)
                return feats_local, data_samples_local

        # If cache does not exist, proceed to generate features
        feats_local = {
            k: torch.empty(size=feats_info['shapes'][k],
                           dtype=feats_info['dtypes'][k],
                           device=self.load_device)
            for k in feats_info['shapes']
        }
        
        data_samples_local = []
        offset = {k: 0 for k in feats_local}

        # Collect features batch by batch
        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_val_iter', batch_idx=idx, data_batch=data_batch)
                
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(data_batch, mode='tensor')
                    
                    for k, v in feats.items():
                        batch_size = v.shape[0]
                        feats_local[k][offset[k]:offset[k] + batch_size] = v
                        offset[k] += batch_size

                    data_samples_local.extend(data_batch['data_samples'])
                
                self.runner.call_hook(
                    'after_val_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)
        
        # Cache features and data samples if cache_dir is provided
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(feats_local, feats_file)
            torch.save(data_samples_local, samples_file)
            print(f"Saved features to {self.cache_dir}")

        return feats_local, data_samples_local

    def _evaluate_single_direction(self, feats_info, cal_t2i, cal_i2t):
        """Run evaluation for either i2t or t2i direction."""
        # Run prediction and evaluation
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size

        feats_local, data_samples_local = self._get_features(feats_info)

        with torch.no_grad():
            data_samples, = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
                cal_t2i=cal_t2i,
                cal_i2t=cal_i2t
            )

            # del feats_local, data_samples_local
            # torch.cuda.empty_cache()
            # gc.collect()

            self.evaluator.process(data_samples, None)
            metrics = self.evaluator.evaluate(img_size if cal_i2t else text_size)
            
        # del data_samples
        # torch.cuda.empty_cache()
        # gc.collect()

        prefix = 'i2t' if cal_i2t else 't2i'
        return {f'{prefix}/{k}': v for k, v in metrics.items()}

    def run(self) -> dict:
        """Launch val."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # Calculate data shapes first
        feats_info = self._get_features_info()
        
        # Run i2t and t2i evaluations separately with memory cleanup
        if self.i2t:
            metrics = self._evaluate_single_direction(
                feats_info,
                cal_t2i=False, 
                cal_i2t=True)
        else:
            metrics = self._evaluate_single_direction(
                feats_info,
                cal_t2i=True, 
                cal_i2t=False)

       # feats_local, data_samples_local = self._get_features(feats_info)
        # metrics = self._evaluate_single_direction(
        #     feats_local,
        #     data_samples_local,
        #     cal_t2i=True,
        #     cal_i2t=False
        # )
        # t2i_metrics = copy.deepcopy(metrics)

        # del feats_local, data_samples_local, metrics
        # torch.cuda.empty_cache()
        # gc.collect()

        # metrics = {**i2t_metrics, **t2i_metrics}
        metrics = {**metrics}

        #  Clear cache_dir after run if specified
        if self.cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics
    

@LOOPS.register_module()
class HDCRetrievalTestLoop(TestLoop):

    def __init__(self,
                 runner,
                 dataloader,
                 evaluator,
                 i2t: bool = True,
                 fp16: bool = False,
                 load_cpu: bool = False,
                 fast_datainfo: bool = False,
                 cache_dir: str = None) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.load_device = torch.device("cpu") if load_cpu else torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.fast_datainfo = fast_datainfo
        self.cache_dir = cache_dir
        self.i2t = i2t
    def _get_features_info(self):
            """Get feature shapes and dtypes from first batch."""
            dataloader_len = len(self.dataloader)
            feats_local_shape = None
            first_batch_size = None
            first_batch_dtype = {}

            for idx, data_batch in enumerate(self.dataloader):
                with torch.no_grad():
                    with autocast(enabled=self.fp16):
                        if is_model_wrapper(self.runner.model):
                            data_preprocessor = self.runner.model.module.data_preprocessor
                        else:
                            data_preprocessor = self.runner.model.data_preprocessor
                        
                        if idx == 0 or idx == dataloader_len - 1:
                            data_batch = data_preprocessor(data_batch, False)
                            feats = self.runner.model._run_forward(data_batch, mode='tensor')
                            if idx == 0:
                                feats_local_shape = {k: list(v.shape) for k, v in feats.items()}
                                first_batch_size = {k: v.shape[0] for k, v in feats.items()}
                                first_batch_dtype = {k: v.dtype for k, v in feats.items()}
                            else:
                                for k, v in feats.items():
                                    feats_local_shape[k][0] += v.shape[0]
                        else:
                            if self.fast_datainfo:
                                for k in feats_local_shape:
                                    feats_local_shape[k][0] += first_batch_size[k]
                            else:
                                data_batch = data_preprocessor(data_batch, False)
                                feats = self.runner.model._run_forward(data_batch, mode='tensor')
                                for k, v in feats.items():
                                    feats_local_shape[k][0] += v.shape[0]

            return {
                'shapes': feats_local_shape,
                'dtypes': first_batch_dtype,
            }

    def _get_features(self, feats_info):
        """Get or generate features, with optional caching."""
        if self.cache_dir:
            feats_file = os.path.join(self.cache_dir, 'feats.pt')
            samples_file = os.path.join(self.cache_dir, 'data_samples.pt')

            # Try loading cached features
            if os.path.exists(feats_file) and os.path.exists(samples_file):
                print(f"Loading cached features from {self.cache_dir}")
                feats_local = torch.load(feats_file, map_location=self.load_device)
                data_samples_local = torch.load(samples_file, map_location=self.load_device)
                return feats_local, data_samples_local

        # If cache does not exist, proceed to generate features
        feats_local = {
            k: torch.empty(size=feats_info['shapes'][k],
                           dtype=feats_info['dtypes'][k],
                           device=self.load_device)
            for k in feats_info['shapes']
        }
        
        data_samples_local = []
        offset = {k: 0 for k in feats_local}

        # Collect features batch by batch
        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_test_iter', batch_idx=idx, data_batch=data_batch)
                
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(data_batch, mode='tensor')
                    
                    for k, v in feats.items():
                        batch_size = v.shape[0]
                        feats_local[k][offset[k]:offset[k] + batch_size] = v
                        offset[k] += batch_size

                    data_samples_local.extend(data_batch['data_samples'])
                
                self.runner.call_hook(
                    'after_test_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)
        
        # Cache features and data samples if cache_dir is provided
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(feats_local, feats_file)
            torch.save(data_samples_local, samples_file)
            print(f"Saved features to {self.cache_dir}")

        return feats_local, data_samples_local

    def _evaluate_single_direction(self, feats_info, cal_t2i, cal_i2t):
        """Run evaluation for either i2t or t2i direction."""
        # Run prediction and evaluation
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size

        feats_local, data_samples_local = self._get_features(feats_info)

        with torch.no_grad():
            data_samples, = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
                cal_t2i=cal_t2i,
                cal_i2t=cal_i2t
            )

            del feats_local, data_samples_local
            torch.cuda.empty_cache()
            gc.collect()

            self.evaluator.process(data_samples, None)
            metrics = self.evaluator.evaluate(img_size if cal_i2t else text_size)
        
        del data_samples
        torch.cuda.empty_cache()
        gc.collect()

        prefix = 'i2t' if cal_i2t else 't2i'
        return {f'{prefix}/{k}': v for k, v in metrics.items()}

    def run(self) -> dict:
        """Launch val."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # Calculate data shapes first
        feats_info = self._get_features_info()
        if self.i2t:
            metrics = self._evaluate_single_direction(
                feats_info,
                cal_t2i=False, 
                cal_i2t=True)
        else:
            metrics = self._evaluate_single_direction(
                feats_info,
                cal_t2i=True, 
                cal_i2t=False)
        
        # feats_local, data_samples_local = self._get_features(feats_info)
        # metrics = self._evaluate_single_direction(
        #     feats_local,
        #     data_samples_local,
        #     cal_t2i=True,
        #     cal_i2t=False
        # )
        # t2i_metrics = copy.deepcopy(metrics)

        # del feats_local, data_samples_local, metrics
        # torch.cuda.empty_cache()
        # gc.collect()

        # metrics = {**i2t_metrics, **t2i_metrics}
        metrics = {**metrics}

         # Clear cache_dir after run if specified
        if self.cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics