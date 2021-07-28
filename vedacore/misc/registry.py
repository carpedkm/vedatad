# adapted from https://github.com/open-mmlab/mmcv
import inspect

from .utils import is_str


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """
    _instance = None

    def __init__(self):
        self._module_dict = dict()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(items={self._module_dict})'
        return format_str

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, cls_name, module_name='module'):
        print('[TRYING TO GET]', cls_name)
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        if module_name not in self._module_dict:
            raise KeyError(f'{module_name} is not in registry')
        dd = self._module_dict[module_name] 
        if cls_name not in dd:
            raise KeyError(f'{cls_name} is not registered in {module_name}')
        print('Returning the class :', dd[cls_name])
        return dd[cls_name]

    def _register_module(self, cls, module_name): # input : class
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, ' f'but got {type(cls)}') # if the argument cls is not type class -> raise this error

        cls_name = cls.__name__ # class name
        self._module_dict.setdefault(module_name, dict()) # empty dictionary when first declared
        dd = self._module_dict[module_name] # dictionary key with the module_name given to be stored.
        if cls_name in dd:
            raise KeyError(f'{cls_name} is already registered '
                           f'in {module_name}')
        dd[cls_name] = cls

    def register_module(self, module_name='module'): # wrapper function in dataset_wrapper calls this

        def _register(cls):
            self._register_module(cls, module_name) # calls this function
            return cls

        return _register # return the function that returns the class (cls)


registry = Registry()


def build_from_cfg(cfg, registry, module_name='module', default_args=None):
    print('[BUILD from] >> ', module_name)
    if not isinstance(cfg, dict): # cfg should be dict, since it's been made as dict
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'typename' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "typename", but got {cfg}')
    if not isinstance(registry, Registry): 
        raise TypeError('registry must be a registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy() # copy the cfg to args -> dictionary type args
    obj_type = args.pop('typename')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type, module_name)
    else:
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def build_from_module(cfg, module, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'typename' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "typename", but got {cfg}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()
    obj_type = args.pop('typename')
    if is_str(obj_type):
        obj_cls = getattr(module, obj_type)
    else:
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)
