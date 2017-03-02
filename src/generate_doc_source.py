import os

documentation = [
    {'package': 'core',
     'title': 'Core Modules',
     'description': 'Important modules that make up the system',
     'subpackages': []},
    # {'package': 'learn',
    #  'title': 'Learning Tasks',
    #  'description': 'Scripts to perform learning tasks',
    #  'subpackages': []}
]


def walk_dir(dir_path):
    """
    Get all the files in a directory.

    :param dir_path: path of directory
    :return: files in directory
    """
    files = []
    for file in os.listdir(dir_path):
        if file == '.' or file == '..':
            continue
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            files += [file_path]
    return files


def filter_files(files, exts):
    """
    Filter files according to a list of extensions.

    :param files: list of files
    :param exts: accepted extensions
    :return: list of accepted files
    """
    return filter(lambda f: os.path.splitext(f)[1] in exts, files)


line1 = '=' * 100
line2 = '-' * 100
line3 = '^' * 100

template_index = '''
CGPCM Documentation
{line1:s}

.. toctree::
   :maxdepth: 2

   {modules:s}


Indices and Tables
{line1:s}

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
'''

template_module = '''
{module:s}
{line2:s}
.. automodule:: {module:s}
'''

template_package = '''
{title:s}
{line1:s}
{description:s}
{modules:s}
'''

template_experiments = '''
Experiments on Branch ``{branch:s}``
{line1:s}
{experiments:s}
'''

template_experiment = '''
{label:s} ({timestamp:s})
{line2:s}
Reason
{line3:s}
{reason:s}

Outcome
{line3:s}
{outcome:s}
'''


class Documentation(object):
    """
    Generator of documentation source.
    """

    def __init__(self, source_dir='source'):
        self._source_dir = source_dir

    def generate(self):
        """
        Generate documentation source.
        """
        self._write_index()
        self._write_packages()

    def _write_packages(self):
        for package in documentation:
            with open(os.path.join(self._source_dir,
                                   package['package'] + '.rst'), 'w') as f:
                f.write(self._render_package(package))

    def _write_index(self):
        with open(os.path.join(self._source_dir, 'index.rst'), 'w') as f:
            f.write(template_index.format(
                modules='\n   '.join(map(lambda x: x['package'],
                                         documentation)),
                line1=line1
            ))

    def _files_to_modules(self, files):
        files = filter_files(files, ['.py'])
        files = map(lambda x: os.path.splitext(x)[0], files)
        modules = filter(lambda x: not x.endswith('__init__'), files)
        modules = map(self._path_to_import, modules)
        return sorted(modules)

    def _import_to_path(self, package):
        return package.replace('.', os.path.sep)

    def _path_to_import(self, package):
        return package.replace(os.path.sep, '.')

    def _render_package(self, package):
        # Gather the packages and the subpackages to be included
        packages = [package['package']] \
                   + map(lambda x: '{}.{}'.format(package['package'], x),
                         package['subpackages'])

        # Extract all modules
        paths = map(self._import_to_path, packages)
        modules = sum(map(lambda x: self._files_to_modules(walk_dir(x)),
                          paths), [])

        # Render the modules
        return template_package.format(
            description=package['description'],
            title=package['title'],
            modules=''.join(
                map(lambda x: template_module.format(module=x, line2=line2),
                    modules)
            ),
            line1=line1
        )


doc = Documentation()
doc.generate()
