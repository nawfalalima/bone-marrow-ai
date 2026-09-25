import os
import torch

path = os.path.join('model', 'bone_marrow_resnet18.pth')
print('cwd', os.getcwd())
print('exists', os.path.exists(path))
obj = torch.load(path, map_location='cpu')
print('type', type(obj))
if isinstance(obj, dict):
    print('num_keys', len(obj))
    print('keys_sample', list(obj.keys())[:10])
    for key, value in obj.items():
        if isinstance(value, dict):
            print('nested_dict_keys', list(value.keys())[:10])
            break
        if isinstance(value, (list, tuple)):
            print('list_len', len(value))
            break
        if hasattr(value, 'shape'):
            print('tensor_shape', tuple(value.shape))
            break
else:
    print('repr', repr(obj)[:500])
