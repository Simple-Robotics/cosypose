import json
from pathlib import Path


class BOPObjectDataset:
    def __init__(self, ds_dir, mesh_units='mm', ignore_symmetric=False, train_classes=None):
        assert mesh_units in ("mm", "m") # 'mm' for tless, 'm' for bracket assembly
        ds_dir = Path(ds_dir)
        infos_file = ds_dir / 'models_info.json'
        infos = json.loads(infos_file.read_text())
        objects = []
        for obj_id, bop_info in infos.items():
            # just load one model (nut: obj_id = obj_000005)
            # print("obj_id", train_classes, obj_id)
            if train_classes and str(obj_id) not in train_classes:
                print("obj_id", obj_id)
                continue
            obj_id = int(obj_id)
            obj_label = f'obj_{obj_id:06d}'
            mesh_path = (ds_dir / obj_label).with_suffix('.ply').as_posix()
            obj = dict(
                label=obj_label,
                category=None,
                mesh_path=mesh_path,
                mesh_units=mesh_units, # mm for tless
            )
            # TODO: for bracket assembly
            # set everything to false
            is_symmetric = False
            if not ignore_symmetric:
                for k in ('symmetries_discrete', 'symmetries_continuous'):
                    obj[k] = bop_info.get(k, [])
                    if len(obj[k]) > 0:
                        is_symmetric = True
            obj['is_symmetric'] = is_symmetric
            obj['diameter'] = bop_info['diameter']
            scale = 0.001 if obj['mesh_units'] == 'mm' else 1.0
            obj['diameter_m'] = bop_info['diameter'] * scale
            objects.append(obj)

        self.objects = objects
        print("self.objects", self.objects)
        self.ds_dir = ds_dir

    def __getitem__(self, idx):
        return self.objects[idx]

    def __len__(self):
        return len(self.objects)
