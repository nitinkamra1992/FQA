import argparse
import sys
from gettext import gettext

if (sys.version_info[0] < 3):  # for python2
    import imp
else:                          # for python3
    import importlib.util

# ############################ Functions ###############################


def expand_dict(b):
    ''' Expand a dictionary b with flattened keys.

    Args:
        b: A dictionary with nested keys

    Returns:
        c: b with its flattened keys expanded back into nested dictionaries.
    '''

    def merge_dicts(d1, d2):
        ''' Merge two dictionaries in the way required by expand_dict()
        '''
        for key in d2:
            if key in d1:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merge_dicts(d1[key], d2[key])
                else:
                    d1[key] = d2[key]
            else:
                d1[key] = d2[key]
        return d1

    c = {}
    for key in b:
        split_keys = key.split('.')
        split_keys.reverse()
        child = {}
        child[split_keys[0]] = b[key]
        for i in split_keys[1:]:
            parent_dict = {}
            parent_dict[i] = child
            child = parent_dict
        merge_dicts(c, child)
    return c


def flatten_keys(d):
    ''' DFS on dictionary to generate a list of (key, value) with flat keys.

    Args:
        d: A dictionary d
        lst: A list of flattened (key, value) pairs from d
    '''
    lst = []
    for key in d:
        if (isinstance(d[key], dict)):
            lst.extend([(key + '.' + k, v) for k, v in flatten_keys(d[key])])
        else:
            lst.append((key, d[key]))
    return lst


def args_to_dict(args, expand=True):
        ''' Converts args namespace to a dictionary.

        Args:
            args: The namespace to convert to dictionary. Typically taken as
                output from parse_args() method.
            expand: Boolean to choose if flattened args should be expanded
                into nested dictionaries.

        Returns:
            dictionary: A dictionary of key value pairs from args.
        '''
        if expand:
            return expand_dict(vars(args))
        else:
            return vars(args).copy()  # vars(args) accesses args internally


# ######################## Custom Data Types ###########################


def pytuple(s):
    assert type(s) == str, 'Input {} cannot be parsed into a tuple'.format(s)
    try:
        L = [el.strip() for el in s.strip('()[]').split(',')]
        for i in range(len(L)):
            try:
                L[i] = int(L[i])
            except ValueError as e:
                try:
                    L[i] = float(L[i])
                except ValueError as e:
                    pass
        return tuple(L)
    except:
        raise argparse.ArgumentTypeError('Input {} cannot be parsed into a tuple'.format(s))


def pylist(s):
    assert type(s) == str, 'Input {} cannot be parsed into a list'.format(s)
    try:
        L = [el.strip() for el in s.strip('[]()').split(',')]
        for i in range(len(L)):
            try:
                L[i] = int(L[i])
            except ValueError as e:
                try:
                    L[i] = float(L[i])
                except ValueError as e:
                    pass
        return list(L)
    except:
        raise argparse.ArgumentTypeError('Input {} cannot be parsed into a list'.format(s))


def pybool(v):
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        raise argparse.ArgumentTypeError('Input {} cannot be parsed into a list'.format(v))


# ############################# Classes ################################


class CustomArgumentParser(argparse.ArgumentParser):
    ''' Provides a wrapper around ArgumentParser to parse a configuration file
        along with other command-line arguments and override any configuration
        args with those provided via command-line. Supports nested structures
        (see help for methods below), and works for python3.5+.
    '''

    def __init__(self, *args, **kwargs):
        ''' Initializer wrapper. Sets conflict_handler argument to 'resolve'.
        '''
        kwargs['conflict_handler'] = 'resolve'
        super(CustomArgumentParser, self).__init__(*args, **kwargs)
        print('Setting conflict_handler argument to \'resolve\'')

    def parse_args(self, *args, **kwargs):
        ''' Wrapper for ArgumentParser.parse_args() method.

        Requires:
            The command-line arguments to the program must supply a '-c' or a
            '--configfile' argument with the location of the configuration
            file. Configuration file should have a python dictionary named
            'config' which contains arguments. Any key inside 'config' or
            inside any nested dictionary in config must be of type string.
            The command-line args can modify inner elements of a dictionary by
            using the dot(.) operator.

        Example:
            ### Configuration file: config.py ###
            config = {
                'arg1': 4,
                'arg2': {
                    'obj1': [3,4],
                    'obj2': 'foo'
                }
            }
            ### End: config.py ###

            ```bash
            python3 <main_program.py> -c config.py --arg1 3 --arg2.obj2 bar
                [other arguments added to the parser]
            ```

        Returns:
            args: A namespace of all known arguments from the command-line
                and those from the configuration file.
                Override priority: Command-line > configuration file > defaults
        '''
        args, unknown = self.parse_known_args()
        if unknown:
            msg = gettext('unrecognized arguments: ')
            self.error(msg + ' '.join(unknown))
        return args

    def parse_known_args(self, *args, **kwargs):
        ''' Wrapper for ArgumentParser.parse_known_args() method.

        Requires:
            The command-line arguments to the program must supply a '-c' or a
            '--configfile' argument with the location of the configuration
            file. Configuration file should have a python dictionary named
            'config' which contains arguments. Any key inside 'config' or
            inside any nested dictionary in config must be of type string.
            The command-line args can modify inner elements of a dictionary by
            using the dot(.) operator.

        Example:
            ### Configuration file: config.py ###
            config = {
                'arg1': 4,
                'arg2': {
                    'obj1': [3,4],
                    'obj2': 'foo'
                }
            }
            ### End: config.py ###

            ```bash
            python3 <main_program.py> -c config.py --arg1 3 --arg2.obj2 bar
                [other arguments]
            ```

        Returns:
            args: A namespace of all known arguments from the command-line
                and those from the configuration file.
                Override priority: Command-line > configuration file > defaults
            unknown: A list of remaining arguments.
        '''
        # Parse the configfile first
        (args, unknown) = \
            super(CustomArgumentParser, self).parse_known_args()

        if sys.version_info[0] < 3:  # python2
            conf_mod = imp.load_source('cfg', args.configfile)
            config = conf_mod.config
        else:                        # python3
            spec = importlib.util.spec_from_file_location('cfg', args.configfile)
            conf_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_mod)
            config = conf_mod.config

        # Flatten all keys in config and add them to the parser as arguments
        flat_config = flatten_keys(config)
        for k, v in flat_config:
            if v is None:
                self.add_argument('--' + k, default=None, type=str)
            else:
                if type(v) == tuple:
                    self.add_argument('--' + k, default=v, type=pytuple)
                elif type(v) == list:
                    self.add_argument('--' + k, default=v, type=pylist)
                elif type(v) == bool:
                    self.add_argument('--' + k, default=v, type=pybool)
                else:
                    self.add_argument('--' + k, default=v, type=type(v))

        # Parse all args again
        (args, unknown) = \
            super(CustomArgumentParser, self).parse_known_args()

        return args, unknown


# ############################ Test code ###############################

if __name__ == '__main__':
    # Create a custom argument parser
    parser = CustomArgumentParser(description='Custom Argument Parser')

    # Add arguments (including a --configfile)
    parser.add_argument('name')
    parser.add_argument('-c', '--configfile')
    parser.add_argument('-d', '--datafile')
    parser.add_argument('--model', type=pylist)
    parser.add_argument('--epochs', default=1, type=int)

    # Test parse_known_args()
    args, unknown = parser.parse_known_args()
    print('\nargs:')
    print(args)
    print('\nunknown:')
    print(unknown)

    # Test parse_args()
    args = parser.parse_args()
    print('\nargs:')
    print(args)

    # Test args_to_dict
    print('\nExpanded args: \n{}'.format(args_to_dict(args, expand=True)))
    print('\nUnexpanded args: \n{}'.format(args_to_dict(args, expand=False)))
