#!/usr/bin/env python3
"""
datasets/base.py
Abstract interface that every dataset adapter must implement.
The rest of the pipeline (preprocessing, patch extraction, training)
only talks to this interface — never to dataset-specific code.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class DatasetAdapter(ABC):

    # Subclasses must set these class-level attributes
    num_classes:    int        # number of emotion categories
    emotion_labels: List[str]  # readable names, index = class id
    dataset_name:   str        # e.g. 'CK+', 'Oulu', 'AFEW'
    cv_strategy:    str        # 'group_kfold' | 'fixed_split'

    def __init__(self, args):
        self.args = args
        self.base_dir = args.chemin_depot_global   # e.g. /sandbox1/aramadan

    # Paths

    def frames_dir(self, subject: str, video: str) -> str:
        """Absolute path to the folder containing PNG/JPG frames."""
        return f'{self.base_dir}/frames/{self.dataset_name}/{subject}/{video}'

    def landmark_pkl(self, subject: str, video: str) -> str:
        return (f'{self.base_dir}/landmark/{self.dataset_name}/'
                f'{subject}/{video}_data.pkl')

    def patches_pkl(self, subject: str, video: str) -> str:
        return (f'{self.base_dir}/patches/{self.dataset_name}/'
                f'{subject}/{video}_patches.pkl')

    def bbox_pkl(self, subject: str, video: str) -> str:
        return (f'{self.base_dir}/bboxes/{self.dataset_name}/'
                f'{subject}/{video}_bbox.pkl')

    def landmark_dir(self, subject: str = '') -> str:
        return f'{self.base_dir}/landmark/{self.dataset_name}/{subject}'

    def patches_dir(self, subject: str = '') -> str:
        return f'{self.base_dir}/patches/{self.dataset_name}/{subject}'

    def labels_pkl_path(self) -> str:
        return f'{self.base_dir}/labels/{self.dataset_name}_labels.pkl'

    # Required interface

    @abstractmethod
    def get_all_samples(self) -> List[Tuple[str, str, int]]:
        """
        Discover every (subject, video, label) triple available on disk.
        Used by preprocessing_exp_faciale.py and extract_patches.py.

        Returns
        -------
        list of (subject_id, video_id, label_int)
            subject_id : folder name under frames/{dataset}/
            video_id   : folder name under frames/{dataset}/{subject}/
            label_int  : integer class index
        """
        raise NotImplementedError

    @abstractmethod
    def get_folds(self) -> List[Dict]:
        """
        Build train/test splits appropriate for this dataset.

        Returns
        -------
        list of dicts, each with keys:
            'fold_id' : int  (1-based)
            'train'   : list of (subject, video, label)
            'test'    : list of (subject, video, label)
        """
        raise NotImplementedError