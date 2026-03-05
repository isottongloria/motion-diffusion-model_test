import random
import numpy as np
import torch, os, pickle, math
from .load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample
from .dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, dataset_name=dataset_name, **kwargs)

        self.w_vectorizer = w_vectorizer


    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']

        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.data_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        
        all_captions = [text]
        all_captions = all_captions * 3  #?

        clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
        # return torch.from_numpy(clip_poses).float(), basename, clip_text
        m_length = clip_poses.shape[0]
        if m_length < self.min_motion_length:
            idx = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        elif m_length > self.max_motion_length:
            idx = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[idx:idx + m_length]
        m_length = clip_poses.shape[0]

        # Text
        tokens = text.split(' ')
        max_text_len = 40
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        return text, torch.from_numpy(clip_poses).float(), m_length, name, None, None, "_".join(tokens), all_captions, None, src


def sample(input,count):
    ss=float(len(input))/count
    return [ input[int(math.floor(i*ss))] for i in range(count) ]