from __future__ import print_function, absolute_import
import os.path as osp
import os
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class PRW_BUformat(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(self.__class__, self).__init__(root, split_id=split_id)
        self.name="PRW_BUformat"
        self.num_cams = 6
        self.is_video = False
        print('in PRW_BUformat')
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        print("create new dataset")
        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        
        # get mars dataset
        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # totally 1261 person (482+?) with 6 camera views each
        # id 1~482 are for training
        # id 483~933 are for testing
        identities = [[{} for _ in range(6)] for _ in range(1503)]

        def register(subdir):
            pids = set()
            vids = []
            person_list = os.listdir(os.path.join(self.root, subdir)); person_list.sort()
            person_list=[person for person in person_list if ((person[0] != '.') and (person[0] != '@')) ]
            for person_id in person_list:
                videos = os.listdir(os.path.join(self.root, subdir, person_id)); videos.sort()
                videos=[video for video in videos if video[0] != '.' and video[0] != '@' ]
                for video_id in videos:
                    video_path = os.path.join(self.root, subdir, person_id, video_id)
                    video_id = int(video_id) - 1
                    fnames = os.listdir(video_path)
                    fnames=[fname for fname in fnames if fname[0] != '.' and fname[0] != '@' ]
                    frame_list = []
                    for fname in fnames:
                        pid = int(person_id)
                        cam = int(fname.split('_')[1][1]) - 1
                        assert -2 <= pid <= 933
                        assert 0 <= cam <= 5
                        pids.add(pid)
                        newname = ('{:04d}_{:02d}_{:05d}_{:04d}.jpg'.format(pid, cam, video_id, len(frame_list)))
                        frame_list.append(newname)
                        shutil.copy(osp.join(video_path, fname), osp.join(images_dir, newname))
                    identities[pid][cam][video_id] = frame_list
                    vids.append(frame_list)
            return pids, vids

        print("begin to preprocess PRW_BUformat dataset")
        print("################################")
        print("################################")
        print("COPY TO IMAGES")
        print("################################")
        print("################################")
        trainval_pids, _ = register('train')
        gallery_pids, gallery_vids = register('gallery_Detected_')
        # gallery_pids, gallery_vids = register('gallery_Detected_')
        query_pids, query_vids = register('query')
        # assert query_pids <= gallery_pids
        # assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': self.name, 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities,
                'query': query_vids,
                'gallery': gallery_vids}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'train': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)) ,
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

