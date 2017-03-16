import os

import out


class Options(object):
    """
    Options parser.

    :param name: name
    """

    def __init__(self, name):
        self._name = name
        self._options = []

    def add_option(self, name, desc='no description available'):
        """
        Add a boolean option.

        :param name: name of options
        :param desc: description of option
        """
        self._options.append({'has_value': False,
                              'name': name.lower(),
                              'description': desc,
                              'required': False,

                              'value': False})

    def add_value_option(self, name, value_type,
                         desc='no description available', required=False,
                         default=None):
        """
        Add an option with a value.

        :param name: name of option
        :param value_type: type of option, should be callable
        :param desc: description of option
        :param required: option is required
        :param default: default value
        """
        self._options.append({'has_value': True,
                              'name': name.lower(),
                              'description': desc,
                              'required': required,

                              'value': default,
                              'value_type': value_type,
                              'specified': False})

    def _get_option(self, name):
        for option in self._options:
            if name == option['name']:
                return option
        raise RuntimeError('option "{}" not found'.format(name))

    def parse(self, args):
        """
        Parse arguments.

        :param args: arguments
        """
        self._options = sorted(self._options, key=lambda x: x['name'])
        self._parse_help(args)
        self._parse_args(args)
        self._parse_required()

    def _parse_required(self):
        missing = []
        for option in self._options:
            if option['required'] and not option['specified']:
                missing.append(option['name'])
        if len(missing) == 1:
            raise RuntimeError('missing option "{}"'.format(missing[0]))
        elif len(missing) > 1:
            missing_string = ', '.join(['"{}"'.format(x) for x in missing])
            raise RuntimeError('missing options {}'.format(missing_string))

    def _parse_args(self, args):
        it = iter(args)
        for arg in it:
            option = self._get_option(arg)
            if option['has_value']:
                option['value'] = option['value_type'](next(it))
                option['specified'] = True
            else:
                option['value'] = True

    def _parse_help(self, args):
        if 'help' in args:
            out.section('options for task')
            out.kv('name', self._name)
            for option in self._options:
                out.section(option['name'])
                out.kv('description', option['description'])
                out.kv('type', 'value' if option['has_value'] else 'bool')
                out.kv('required', 'yes' if option['required'] else 'no')
                out.section_end()
            out.section_end()
            exit()

    def __getitem__(self, name):
        return self._get_option(name)['value']

    def fp(self, groups=None, ignore=None):
        """
        Create a file path that identifies the current setting.

        :param groups: list of groups of option names to group by
        :param ignore: list of names to ignore
        :return: `FilePath` instance
        """
        ignore = [] if ignore is None else ignore
        groups = [] if groups is None else groups
        postfix = sorted(set([x['name'] for x in self._options])
                          - set(sum(groups, [])) - set(ignore))
        groups = map(self._names_to_str, groups)
        postfix = self._names_to_str(postfix)
        return FilePath([self._name] + groups + [postfix])

    def _value_to_str(self, val):
        if type(val) == bool:
            return 'y' if val else 'n'
        else:
            return str(val)

    def _names_to_str(self, names):
        names = sorted(names)
        options = map(self._get_option, names)
        strs = ['{}={}'.format(x['name'], self._value_to_str(x['value']))
                for x in options]
        return ','.join(strs)


class FilePath(object):
    """
    File path with added functionality.

    :param fp: file path in either string or list form
    """

    def __init__(self, fp):
        if type(fp) == str:
            self._parts = filter(None, fp.split(os.path.sep))
        elif type(fp) == list:
            self._parts = filter(None, fp)
        elif type(fp) == FilePath:
            self._parts = fp._parts
        else:
            raise ValueError('unknown type "{}"'.format(str(type(fp))))

    def __str__(self):
        return os.path.sep.join(self._parts)

    def __add__(self, other):
        return FilePath(self._parts + FilePath(other)._parts)

    def __radd__(self, other):
        return FilePath.__add__(other, self)

    def __and__(self, other):
        other = FilePath(other)
        if len(self._parts) != len(other._parts):
            raise ValueError('cannot combine file paths of different lengths')
        result_parts = []
        for p1, p2 in zip(self._parts, other._parts):
            result_parts.append(p1 if p1 == p2 else p1 + '_and_' + p2)
        return FilePath(result_parts)



